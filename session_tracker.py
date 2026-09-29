#!/usr/bin/env python3
"""
Point-in-time-safe, streaming session/level tracker.

Mirrors trading_agent/indicators.py's VWAP pattern: a bar-by-bar
accumulator (`update(bar)` called once per 1-minute bar, in order) rather
than a full-dataframe scan, so a level literally cannot have a value until
its source window has actually started, and never gets backfilled from
data that hasn't happened yet in replay time. This is deliberately NOT
built on trading_agent/poi_scanner.py's compute_session_levels, which
silently fabricates a level's value (falls back to PDH/PDL if Asia hasn't
started, falls back to Asia H/L if London hasn't started) with no flag
marking that it happened — that's fine for a pre-session snapshot tool,
but wrong for a tracker meant to be queried continuously through the day.

Tracks, as of the current bar and nothing later:
  - PDH/PDL (frozen at the prior Globex day's close)
  - Asia High/Low (frozen at asia_end_et)
  - Developing London High/Low (grows through london_start_et-london_end_et)
  - VWAP (reused as-is from trading_agent.indicators.VWAP)
  - Weekly floor pivots (precomputed once via weekly_levels.py, looked up
    per-week — safe because compute_pivots for week N only ever depends on
    week N-1's already-completed data)
  - A 7-state machine per level: far -> approaching -> testing ->
    {breaking -> broken_holding -> {lost|reclaimed}} | rejecting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from session_config import SessionConfig
from trading_agent.indicators import VWAP
from weekly_levels import WeeklyPivots, _session_date

ET = ZoneInfo("America/New_York")

_TERMINAL_INTERACTION_STATES = {
    "testing", "breaking", "broken_holding", "rejecting", "reclaimed", "lost",
}


SCANNER_CONFIG_VERSION = "v1"  # FROZEN 2026-09-27 — do not retune these defaults based on
                                # forward-outcome/MFE-MAE data. Any change is a new, separately
                                # tagged version (v2, ...), not an edit to v1's numbers, so past
                                # scan_runs stay interpretable against what actually produced them.


@dataclass
class ScannerConfig:
    version:                    str   = SCANNER_CONFIG_VERSION
    approach_ticks:             int   = 40   # ~10pts NQ-scale — wide, many levels are "approaching" often
    touch_ticks:                int   = 8    # matches poi_scanner.py's zone_buffer_ticks convention
    hold_confirm_bars:          int   = 3    # bars on the far side before BREAKING -> BROKEN_HOLDING
    confluence_tolerance_ticks: int   = 8
    volatility_percentile:      float = 0.60  # short-window ATR >= its own 60th percentile
    vwap_extension_atr_mult:    float = 0.5   # |price-vwap| >= 0.5x ATR
    momentum_expansion_bars:    int   = 2     # consecutive bars of growing MACD histogram magnitude


@dataclass
class LevelState:
    name: str
    price: Optional[float]
    is_developing: bool
    state: str                    # far|approaching|testing|breaking|broken_holding|rejecting|reclaimed|lost
    distance_pts: Optional[float]


@dataclass
class ConfluenceZone:
    center: float
    level_names: tuple[str, ...]
    count: int


class _LevelWatch:
    """Per-level 7-state machine, updated on every bar's close."""

    def __init__(self) -> None:
        self.state = "far"
        self.side: Optional[str] = None          # current side of price relative to the level
        self.pre_break_side: Optional[str] = None
        self.hold_bars = 0

    def update(self, price: float, level_price: Optional[float],
               approach_dist: float, touch_dist: float, hold_confirm_bars: int) -> None:
        if level_price is None:
            # The level doesn't exist right now — either it hasn't formed yet, or
            # (for a developing session level) it was just reset at a new day/session
            # boundary. Either way, any stale state from before is meaningless — a
            # level with no price can't be "testing" or "broken_holding".
            self.state, self.side, self.pre_break_side, self.hold_bars = "far", None, None, 0
            return
        dist = price - level_price
        adist = abs(dist)
        cur_side = "above" if dist >= 0 else "below"

        if self.state == "far":
            if adist <= approach_dist:
                self.state, self.side = "approaching", cur_side

        elif self.state == "approaching":
            if adist > approach_dist:
                self.state, self.side = "far", None
            elif adist <= touch_dist:
                self.state = "testing"
                self.side = cur_side
            else:
                self.side = cur_side

        elif self.state == "testing":
            if cur_side != self.side:
                self.pre_break_side = self.side
                self.side = cur_side
                self.state, self.hold_bars = "breaking", 1
            elif adist > touch_dist:
                self.state = "rejecting"

        elif self.state == "rejecting":
            if adist <= touch_dist:
                self.state, self.side = "testing", cur_side
            elif adist > approach_dist:
                self.state, self.side = "far", None

        elif self.state == "breaking":
            if cur_side == self.pre_break_side:
                # snapped back before the break was confirmed
                self.state, self.side, self.hold_bars = "rejecting", cur_side, 0
            else:
                self.hold_bars += 1
                if self.hold_bars >= hold_confirm_bars:
                    self.state = "broken_holding"

        elif self.state == "broken_holding":
            if cur_side == self.pre_break_side:
                # price has come back through, against the break direction
                self.state = "lost" if self.pre_break_side == "below" else "reclaimed"
                self.side = cur_side

        elif self.state in ("lost", "reclaimed"):
            if adist <= touch_dist:
                self.state, self.side = "testing", cur_side
            elif adist > approach_dist:
                self.state, self.side = "far", None


class SessionTracker:
    """One instance per instrument. Call update(bar) once per 1-minute bar,
    strictly in time order."""

    def __init__(self, config: Optional[SessionConfig] = None,
                 scanner_config: Optional[ScannerConfig] = None,
                 weekly_pivots: Optional[list[WeeklyPivots]] = None,
                 tick_size: float = 0.25):
        self.config = config or SessionConfig()
        self.scanner_config = scanner_config or ScannerConfig()
        self._weekly_pivots = sorted(weekly_pivots or [], key=lambda p: p.week_start)
        self.tick_size = tick_size

        self._vwap = VWAP()
        self._vwap_date: Optional[date] = None

        self._globex_day_key: Optional[date] = None
        self._day_start_date: Optional[date] = None
        self._cur_day_high: Optional[float] = None
        self._cur_day_low: Optional[float] = None
        self.pdh: Optional[float] = None
        self.pdl: Optional[float] = None

        self._asia_high: Optional[float] = None
        self._asia_low: Optional[float] = None
        self.asia_high: Optional[float] = None
        self.asia_low: Optional[float] = None
        self._asia_developing = False

        self._london_high: Optional[float] = None
        self._london_low: Optional[float] = None
        self.london_high: Optional[float] = None
        self.london_low: Optional[float] = None
        self._london_developing = False

        self._level_watches: dict[str, _LevelWatch] = {}
        self._last_close: Optional[float] = None
        self._last_timestamp: Optional[datetime] = None

    # ------------------------------------------------------------------

    def update(self, bar) -> None:
        ts_et = bar.timestamp.astimezone(ET)
        cur_date = ts_et.date()
        t = ts_et.time()
        day_key = _session_date(ts_et)

        if self._vwap_date != cur_date:
            self._vwap.reset()
            self._vwap_date = cur_date
        self._vwap.update(bar)

        if day_key != self._globex_day_key:
            if self._globex_day_key is not None:
                self.pdh, self.pdl = self._cur_day_high, self._cur_day_low
            self._globex_day_key = day_key
            self._day_start_date = cur_date
            self._cur_day_high, self._cur_day_low = bar.high, bar.low

            in_asia = t >= self.config.asia_start_et
            self._asia_high = bar.high if in_asia else None
            self._asia_low = bar.low if in_asia else None
            self.asia_high, self.asia_low = None, None
            self._asia_developing = True

            self._london_high, self._london_low = None, None
            self.london_high, self.london_low = None, None
            self._london_developing = False
        else:
            self._cur_day_high = max(self._cur_day_high, bar.high)
            self._cur_day_low = min(self._cur_day_low, bar.low)

            if self._asia_developing:
                if cur_date == self._day_start_date:
                    if t >= self.config.asia_start_et:
                        self._asia_high = bar.high if self._asia_high is None else max(self._asia_high, bar.high)
                        self._asia_low = bar.low if self._asia_low is None else min(self._asia_low, bar.low)
                else:
                    # calendar date advanced past midnight -> Asia freezes
                    self.asia_high, self.asia_low = self._asia_high, self._asia_low
                    self._asia_developing = False

            if cur_date == day_key:
                if self.config.london_start_et <= t < self.config.london_end_et:
                    self._london_developing = True
                    self._london_high = bar.high if self._london_high is None else max(self._london_high, bar.high)
                    self._london_low = bar.low if self._london_low is None else min(self._london_low, bar.low)
                elif t >= self.config.london_end_et and self._london_developing:
                    self.london_high, self.london_low = self._london_high, self._london_low
                    self._london_developing = False

        self._last_close = bar.close
        self._last_timestamp = bar.timestamp

        approach_dist = self.scanner_config.approach_ticks * self.tick_size
        touch_dist = self.scanner_config.touch_ticks * self.tick_size
        for name, price, _ in self._raw_levels():
            watch = self._level_watches.setdefault(name, _LevelWatch())
            watch.update(bar.close, price, approach_dist, touch_dist,
                         self.scanner_config.hold_confirm_bars)

    # ------------------------------------------------------------------

    def _weekly_pivot_row(self) -> Optional[WeeklyPivots]:
        if not self._weekly_pivots or self._last_timestamp is None:
            return None
        cur_date = self._last_timestamp.astimezone(ET).date()
        applicable = None
        for pv in self._weekly_pivots:
            if pv.week_start <= cur_date:
                applicable = pv
            else:
                break
        return applicable

    def _raw_levels(self) -> list[tuple[str, Optional[float], bool]]:
        """(name, price, is_developing) for every tracked level."""
        levels: list[tuple[str, Optional[float], bool]] = [
            ("PDH", self.pdh, False),
            ("PDL", self.pdl, False),
            ("AsiaHigh", self.asia_high if self.asia_high is not None else self._asia_high,
             self._asia_developing),
            ("AsiaLow", self.asia_low if self.asia_low is not None else self._asia_low,
             self._asia_developing),
            ("LondonHigh", self.london_high if self.london_high is not None else self._london_high,
             self._london_developing),
            ("LondonLow", self.london_low if self.london_low is not None else self._london_low,
             self._london_developing),
            # VWAP deliberately excluded: price oscillates through it constantly by
            # definition, so treating it as a discrete "level" to test/break against
            # the same approach/touch tolerances as PDH/PDL etc. would make
            # level_interaction fire almost every scan. VWAP has its own dedicated
            # vwap_extension signal in opportunity_scanner.py instead.
        ]
        pv = self._weekly_pivot_row()
        if pv is not None:
            for name, price in [("WeeklyP", pv.p), ("WeeklyR1", pv.r1), ("WeeklyR2", pv.r2),
                                 ("WeeklyR3", pv.r3), ("WeeklyS1", pv.s1), ("WeeklyS2", pv.s2),
                                 ("WeeklyS3", pv.s3)]:
                levels.append((name, price, False))
        return levels

    # ------------------------------------------------------------------

    @property
    def vwap(self) -> Optional[float]:
        return self._vwap.value

    def level_states(self) -> list[LevelState]:
        """Reflects the bar most recently passed to update()."""
        out = []
        for name, price, is_dev in self._raw_levels():
            watch = self._level_watches.get(name)
            state = watch.state if watch is not None else "far"
            dist = (self._last_close - price) if (price is not None and self._last_close is not None) else None
            out.append(LevelState(name=name, price=price, is_developing=is_dev,
                                   state=state, distance_pts=dist))
        return out

    def confluence_zones(self) -> list[ConfluenceZone]:
        tol = self.scanner_config.confluence_tolerance_ticks * self.tick_size
        refs = sorted(
            ((name, price) for name, price, _ in self._raw_levels() if price is not None),
            key=lambda kv: kv[1],
        )
        clusters: list[dict] = []
        for name, price in refs:
            placed = False
            for c in clusters:
                centroid = sum(c["prices"]) / len(c["prices"])
                if abs(price - centroid) <= tol:
                    c["names"].append(name)
                    c["prices"].append(price)
                    placed = True
                    break
            if not placed:
                clusters.append({"names": [name], "prices": [price]})
        return [
            ConfluenceZone(center=sum(c["prices"]) / len(c["prices"]),
                            level_names=tuple(c["names"]), count=len(c["names"]))
            for c in clusters if len(c["names"]) >= 2
        ]
