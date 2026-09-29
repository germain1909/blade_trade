#!/usr/bin/env python3
"""
Opportunity Scanner — deterministic, no LLM. Runs every 15 minutes against
a SessionTracker and asks: "is something sufficiently interesting happening
that deeper analysis would be justified?"

Four signal checks run every scan, independent of any level (so the
scanner isn't only about levels): momentum reversal, momentum expansion,
volatility expansion, VWAP extension. Two level-driven checks layer on
top: direct level interaction (always counts on its own), and "approaching
a level with supporting context" (approaching alone never counts — this is
what stops the scanner firing constantly on every near-miss).

See RESEARCH_NOTES.md / the approved architecture plan for the full
rationale and worked examples.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from mtf_indicators import rolling_percentile
from session_tracker import ConfluenceZone, LevelState, ScannerConfig, SessionTracker
from weekly_levels import _session_date

ET = ZoneInfo("America/New_York")

_INTERACTION_STATES = {"testing", "breaking", "broken_holding", "rejecting", "reclaimed", "lost"}

_ATR_SERIES_WINDOW = 2000  # ~1.5 days of 1m bars — trailing range for the volatility "percentile"


@dataclass(frozen=True)
class Opportunity:
    timestamp: datetime
    symbol: str
    sessions: frozenset[str]
    is_opportunity: bool
    categories: tuple[str, ...]
    reasoning: tuple[str, ...]
    level_states: tuple[LevelState, ...]
    confluence_zones: tuple[ConfluenceZone, ...]
    episode_id: Optional[str]
    is_new_episode: bool


def compute_scanner_features(df: pd.DataFrame, atr_period: int = 60) -> pd.DataFrame:
    """Precompute, once, over the whole 1m dataframe: a short-window ATR and
    its position (0-1) within its own trailing range — reused
    mtf_indicators.rolling_percentile for consistency with how MACD extremes
    are already judged elsewhere in this codebase. `atr_period` is in
    1-minute bars — 60 (roughly an hour of true range) rather than a
    classic 14, since 14 minutes is too short a window to give a meaningful
    "how extended is price from VWAP" comparison; at 14 the vwap_extension
    signal fired on almost every scan. Indexed by timestamp,
    same as mtf_indicators.compute_mtf_features."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=max(atr_period // 2, 1)).mean()
    atr_pctile = rolling_percentile(atr, _ATR_SERIES_WINDOW)
    out = pd.DataFrame({"atr": atr, "atr_percentile": atr_pctile})
    out.index = df["timestamp"]
    out["daily_atr"] = _daily_atr_series(df)
    return out


def _daily_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR on completed Globex daily bars (same session_date
    grouping weekly_levels.py already uses), forward-filled onto the
    1-minute index via merge_asof(direction="backward") — a day's ATR only
    becomes visible once that day has actually closed, so "today" always
    sees yesterday's-and-earlier value, never its own still-forming range.
    This is the appropriate scale for comparing against VWAP's distance
    (VWAP is a session-cumulative average, not a short-window one)."""
    et = df["timestamp"].dt.tz_convert(ET)
    session_date = et.apply(_session_date)
    daily = (
        df.assign(_sd=session_date)
        .groupby("_sd")
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"),
             timestamp=("timestamp", "last"))
        .reset_index(drop=True)
        .sort_values("timestamp")
    )
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=max(period // 2, 1)).mean()

    daily_atr = pd.DataFrame({"ts": daily["timestamp"].reset_index(drop=True), "daily_atr": atr.reset_index(drop=True)})
    base = pd.DataFrame({"ts": df["timestamp"].sort_values().reset_index(drop=True)})
    merged = pd.merge_asof(base, daily_atr.sort_values("ts"), on="ts", direction="backward")
    return pd.Series(merged["daily_atr"].values, index=merged["ts"])


def _momentum_expansion(features: pd.DataFrame, timestamp: datetime, bars: int) -> bool:
    vals = []
    for k in range(bars + 1):
        t = timestamp - pd.Timedelta(minutes=15 * k)
        if t in features.index:
            v = features.loc[t].get("15m_hist")
            vals.append(None if pd.isna(v) else float(v))
        else:
            vals.append(None)
    vals = vals[::-1]  # oldest -> newest
    if any(v is None for v in vals) or vals[-1] == 0:
        return False
    mags = [abs(v) for v in vals]
    same_sign = all((v > 0) == (vals[-1] > 0) for v in vals)
    growing = all(mags[i] < mags[i + 1] for i in range(len(mags) - 1))
    return same_sign and growing


def _episode_id(level_states: list[LevelState], categories: set[str]) -> Optional[str]:
    if not categories:
        return None
    driving = [
        l for l in level_states
        if l.state in _INTERACTION_STATES or l.state == "approaching"
    ]
    if driving:
        closest = min(driving, key=lambda l: abs(l.distance_pts) if l.distance_pts is not None else float("inf"))
        key = f"{closest.name}:{closest.state}"
    else:
        key = "|".join(sorted(categories))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def scan(
    tracker: SessionTracker,
    features: pd.DataFrame,
    scanner_features: pd.DataFrame,
    price: float,
    timestamp: datetime,
    symbol: str,
    seen_episodes: set[str],
    cfg: Optional[ScannerConfig] = None,
) -> Opportunity:
    cfg = cfg or tracker.scanner_config

    ts_et = timestamp.astimezone(ET)
    sessions = tracker.config.active_sessions(ts_et.time())

    feat_row = features.loc[timestamp] if timestamp in features.index else None
    sf_row = scanner_features.loc[timestamp] if timestamp in scanner_features.index else None

    momentum_reversal = bool(feat_row is not None and (
        feat_row.get("15m_macd_bottom_recent") or feat_row.get("15m_macd_top_recent")
    ))
    momentum_expansion = _momentum_expansion(features, timestamp, cfg.momentum_expansion_bars)

    atr = float(sf_row["atr"]) if sf_row is not None and pd.notna(sf_row.get("atr")) else None
    atr_pctile = float(sf_row["atr_percentile"]) if sf_row is not None and pd.notna(sf_row.get("atr_percentile")) else None
    volatility_expansion = atr_pctile is not None and atr_pctile >= cfg.volatility_percentile

    daily_atr = float(sf_row["daily_atr"]) if sf_row is not None and pd.notna(sf_row.get("daily_atr")) else None
    vwap = tracker.vwap
    vwap_extension = bool(vwap is not None and daily_atr is not None and daily_atr > 0
                           and abs(price - vwap) / daily_atr >= cfg.vwap_extension_atr_mult)

    level_states = tracker.level_states()
    zones = tracker.confluence_zones()

    categories: set[str] = set()
    reasoning: list[str] = []

    def _dist(l: LevelState) -> str:
        return f"{l.distance_pts:+.2f}" if l.distance_pts is not None else "?"

    interacting = [l for l in level_states if l.state in _INTERACTION_STATES]
    if interacting:
        categories.add("level_interaction")
        for l in interacting:
            reasoning.append(f"{l.name} is {l.state} ({_dist(l)}pts)")

    approaching = [l for l in level_states if l.state == "approaching"]
    context = momentum_reversal or momentum_expansion or volatility_expansion or vwap_extension or bool(zones)
    if approaching and context:
        categories.add("level_approach_with_context")
        for l in approaching:
            reasoning.append(f"{l.name} approaching ({_dist(l)}pts) with supporting context")

    if momentum_reversal:
        categories.add("momentum_reversal")
        reasoning.append("15m MACD substantial reversal seen recently")
    if momentum_expansion:
        categories.add("momentum_expansion")
        reasoning.append(f"15m MACD histogram expanding {cfg.momentum_expansion_bars}+ bars")
    if volatility_expansion:
        categories.add("volatility_expansion")
        reasoning.append(f"ATR at {atr_pctile:.0%} of its trailing range")
    if vwap_extension:
        categories.add("vwap_extension")
        reasoning.append(f"price {abs(price - vwap):.2f}pts from VWAP ({abs(price - vwap) / daily_atr:.2f}x daily ATR)")

    is_opportunity = len(categories) > 0
    episode_id = _episode_id(level_states, categories) if is_opportunity else None
    is_new_episode = bool(episode_id is not None and episode_id not in seen_episodes)
    if episode_id is not None:
        seen_episodes.add(episode_id)

    return Opportunity(
        timestamp=timestamp, symbol=symbol, sessions=sessions,
        is_opportunity=is_opportunity, categories=tuple(sorted(categories)),
        reasoning=tuple(reasoning), level_states=tuple(level_states),
        confluence_zones=tuple(zones), episode_id=episode_id, is_new_episode=is_new_episode,
    )
