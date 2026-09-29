#!/usr/bin/env python3
"""
OpportunitySnapshot — the objective, point-in-time representation of "what
did BladeTrade know at timestamp T?" for a scanner episode.

Deliberately a distinct type from market_snapshot.MarketSnapshot (Milestone
1's per-candidate-setup snapshot, singular level_name/setup_direction) —
this one is centered on a full scanner episode: every tracked level, every
active session, confluence, VWAP context, momentum, volatility, and the
scanner's own reasoning for why the episode fired.

No forward-looking field exists on this type at all. Not "present but
unpopulated" — structurally absent, so nothing downstream (chart, trend
analysis, vision, final synthesis) could read future/outcome data even by
mistake. Forward-excursion data lives only in generate_episode_review.py's
separate output, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from opportunity_scanner import Opportunity
from session_tracker import ConfluenceZone, LevelState, SessionTracker

_DEFAULT_LOOKBACK_BARS = 60


@dataclass(frozen=True)
class OpportunitySnapshot:
    episode_id: str
    timestamp: datetime
    symbol: str
    price: float
    sessions: frozenset[str]

    asia_high: Optional[float]
    asia_low: Optional[float]
    london_high: Optional[float]
    london_low: Optional[float]
    london_is_developing: bool

    levels: tuple[LevelState, ...]
    confluence_zones: tuple[ConfluenceZone, ...]

    vwap: Optional[float]
    vwap_distance_pts: Optional[float]
    vwap_distance_daily_atr: Optional[float]
    vwap_extension_flag: bool

    macd_15m_hist: Optional[float]
    macd_15m_bottom_recent: bool
    macd_15m_top_recent: bool
    rsi_60m: Optional[float]
    macd_4h_hist: Optional[float]
    rsi_4h: Optional[float]

    atr_60bar: Optional[float]
    atr_percentile: Optional[float]
    daily_atr: Optional[float]

    is_opportunity: bool
    is_new_episode: bool
    categories: tuple[str, ...]
    reasoning: tuple[str, ...]

    recent_bars: tuple[tuple, ...]  # (ts, open, high, low, close, volume), oldest->newest, all <= timestamp

    def to_dict(self) -> dict:
        """JSON-safe representation for persistence and prompts."""
        import dataclasses as _dc

        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif k == "recent_bars":
                d[k] = [[b[0].isoformat() if isinstance(b[0], datetime) else b[0], *b[1:]] for b in v]
            elif k == "levels":
                d[k] = [_dc.asdict(l) for l in v]
            elif k == "confluence_zones":
                d[k] = [_dc.asdict(z) for z in v]
            elif isinstance(v, (frozenset, tuple)):
                d[k] = sorted(v) if isinstance(v, frozenset) else list(v)
            else:
                d[k] = v
        return d


def build_opportunity_snapshot(
    tracker: SessionTracker,
    features: pd.DataFrame,
    scanner_features: pd.DataFrame,
    opp: Opportunity,
    bars_df: pd.DataFrame,
    lookback_bars: int = _DEFAULT_LOOKBACK_BARS,
) -> OpportunitySnapshot:
    """
    Build a snapshot as of opp.timestamp. `bars_df` may extend past that
    timestamp (e.g. the full historical file) — it is truncated here before
    anything is read out of it, so passing an untruncated dataframe is safe.
    """
    timestamp = opp.timestamp
    truncated = bars_df.loc[bars_df["timestamp"] <= timestamp]
    recent = truncated.tail(lookback_bars)
    recent_bars = tuple(
        (r.timestamp, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
        for r in recent.itertuples()
    )

    feat_row = features.loc[timestamp] if timestamp in features.index else None
    sf_row = scanner_features.loc[timestamp] if timestamp in scanner_features.index else None

    def _f(row, col):
        if row is None:
            return None
        v = row.get(col)
        return None if pd.isna(v) else float(v)

    # Opportunity has no price field of its own — use the last available bar's
    # close, matching what the scanner itself used at scan time.
    price = float(recent.iloc[-1]["close"]) if len(recent) else float("nan")

    vwap = tracker.vwap
    daily_atr = _f(sf_row, "daily_atr")
    vwap_distance_pts = (price - vwap) if vwap is not None else None
    vwap_distance_daily_atr = (
        vwap_distance_pts / daily_atr if vwap_distance_pts is not None and daily_atr not in (None, 0) else None
    )

    asia_high = next((l.price for l in opp.level_states if l.name == "AsiaHigh"), None)
    asia_low = next((l.price for l in opp.level_states if l.name == "AsiaLow"), None)
    london_high = next((l.price for l in opp.level_states if l.name == "LondonHigh"), None)
    london_low = next((l.price for l in opp.level_states if l.name == "LondonLow"), None)
    london_dev = next((l.is_developing for l in opp.level_states if l.name == "LondonHigh"), False)

    return OpportunitySnapshot(
        episode_id=opp.episode_id or "",
        timestamp=timestamp,
        symbol=opp.symbol,
        price=price,
        sessions=opp.sessions,
        asia_high=asia_high, asia_low=asia_low,
        london_high=london_high, london_low=london_low,
        london_is_developing=london_dev,
        levels=opp.level_states,
        confluence_zones=opp.confluence_zones,
        vwap=vwap, vwap_distance_pts=vwap_distance_pts,
        vwap_distance_daily_atr=vwap_distance_daily_atr,
        vwap_extension_flag="vwap_extension" in opp.categories,
        macd_15m_hist=_f(feat_row, "15m_hist"),
        macd_15m_bottom_recent=bool(feat_row is not None and feat_row.get("15m_macd_bottom_recent")),
        macd_15m_top_recent=bool(feat_row is not None and feat_row.get("15m_macd_top_recent")),
        rsi_60m=_f(feat_row, "60m_rsi"),
        macd_4h_hist=_f(feat_row, "4h_hist"),
        rsi_4h=_f(feat_row, "4h_rsi"),
        atr_60bar=_f(sf_row, "atr"),
        atr_percentile=_f(sf_row, "atr_percentile"),
        daily_atr=daily_atr,
        is_opportunity=opp.is_opportunity,
        is_new_episode=opp.is_new_episode,
        categories=opp.categories,
        reasoning=opp.reasoning,
        recent_bars=recent_bars,
    )
