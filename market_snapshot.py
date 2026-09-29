#!/usr/bin/env python3
"""
MarketSnapshot — a point-in-time bundle of everything a decision layer
(rule-based or LLM) needs to judge a candidate setup, without needing to
touch raw bars or indicator dataframes itself.

No-lookahead is enforced structurally, not by convention: `build_snapshot`
truncates `bars_df` to `<= timestamp` before reading anything out of it, so
a snapshot built at T cannot see a bar that closes after T even if the
caller passes in a dataframe that extends well past T. The `feat_row`
passed in (typically `features.loc[timestamp]` from
`mtf_indicators.compute_mtf_features`) is assumed to already carry that
same guarantee for higher-timeframe indicator values, via that function's
`pd.merge_asof(..., direction="backward")` alignment.

See test_no_lookahead.py for the regression test proving this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

_DEFAULT_LOOKBACK_BARS = 30
_ATR_PERIOD = 14


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: datetime
    symbol: str
    price: float
    recent_bars: tuple[tuple[Any, float, float, float, float, float], ...]  # (ts, o, h, l, c, v), oldest->newest, all <= timestamp

    session: str
    levels: tuple[tuple[str, float], ...]              # (name, price)
    distance_to_levels: tuple[tuple[str, float], ...]   # (name, price - level)

    vwap: Optional[float]
    macd_15m_hist: Optional[float]
    macd_15m_bottom_recent: bool
    macd_15m_top_recent: bool
    rsi_60m: Optional[float]
    macd_4h_hist: Optional[float]
    rsi_4h: Optional[float]
    higher_tf_trend: str
    atr: Optional[float]

    level_name: str
    setup_type: str        # "cross" | "fail"
    setup_direction: str   # "long" | "short"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        """JSON-safe representation for prompts and experiment_store persistence."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif k == "recent_bars":
                d[k] = [
                    [b[0].isoformat() if isinstance(b[0], datetime) else b[0], *b[1:]]
                    for b in v
                ]
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d


def _session_for(ts_et: datetime) -> str:
    """Rough Globex session bucket by ET hour. Approximation, not exact
    exchange session boundaries."""
    t = ts_et.time()
    if dtime(17, 0) <= t or t < dtime(3, 0):
        return "asia"
    if dtime(3, 0) <= t < dtime(8, 0):
        return "london"
    if dtime(8, 0) <= t < dtime(17, 0):
        return "ny"
    return "overnight"


def _higher_tf_trend(macd_4h_hist: Optional[float], rsi_4h: Optional[float]) -> str:
    if macd_4h_hist is None:
        return "unknown"
    if macd_4h_hist > 0:
        return "bullish" if (rsi_4h is None or rsi_4h >= 45) else "bullish_weak"
    if macd_4h_hist < 0:
        return "bearish" if (rsi_4h is None or rsi_4h <= 55) else "bearish_weak"
    return "neutral"


def _atr(bars: pd.DataFrame, period: int = _ATR_PERIOD) -> Optional[float]:
    """Simple ATR over the given (already-truncated) bar window. Short lookback
    on whatever timeframe `bars` is in (1m recent_bars by default here) — a
    placeholder measure of short-term volatility, not a session/daily ATR."""
    if len(bars) < 2:
        return None
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr = tr.dropna()
    if tr.empty:
        return None
    window = tr.tail(period)
    return round(float(window.mean()), 4)


def build_snapshot(
    *,
    timestamp: datetime,
    symbol: str,
    price: float,
    bars_df: pd.DataFrame,
    feat_row,
    levels: dict[str, float],
    level_name: str,
    setup_type: str,
    setup_direction: str,
    tags: tuple[str, ...] = (),
    lookback_bars: int = _DEFAULT_LOOKBACK_BARS,
) -> MarketSnapshot:
    """
    Build a MarketSnapshot as of `timestamp`.

    `bars_df` may span past `timestamp` — it is truncated here before anything
    is read out of it, so passing a dataframe that runs to the end of the
    week is safe and does not leak future bars into `recent_bars` or `atr`.

    `feat_row` should be the single row of indicator features at `timestamp`
    (e.g. `features.loc[timestamp]` from `mtf_indicators.compute_mtf_features`),
    which is itself no-lookahead by construction.
    """
    truncated = bars_df.loc[bars_df["timestamp"] <= timestamp]
    recent = truncated.tail(lookback_bars)
    recent_bars = tuple(
        (r.timestamp, float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
        for r in recent.itertuples()
    )

    ts_et = timestamp.astimezone(ET)
    session = _session_for(ts_et)

    distance = tuple((name, round(price - lvl, 4)) for name, lvl in levels.items())

    def _get(col: str):
        if feat_row is None:
            return None
        val = feat_row.get(col)
        return None if pd.isna(val) else val

    macd_15m_hist = _get("15m_hist")
    macd_4h_hist = _get("4h_hist")
    rsi_60m = _get("60m_rsi")
    rsi_4h = _get("4h_rsi")

    return MarketSnapshot(
        timestamp=timestamp,
        symbol=symbol,
        price=price,
        recent_bars=recent_bars,
        session=session,
        levels=tuple(levels.items()),
        distance_to_levels=distance,
        vwap=None,  # not currently computed upstream; reserved for a future fast-follow
        macd_15m_hist=float(macd_15m_hist) if macd_15m_hist is not None else None,
        macd_15m_bottom_recent=bool(feat_row is not None and feat_row.get("15m_macd_bottom_recent")),
        macd_15m_top_recent=bool(feat_row is not None and feat_row.get("15m_macd_top_recent")),
        rsi_60m=float(rsi_60m) if rsi_60m is not None else None,
        macd_4h_hist=float(macd_4h_hist) if macd_4h_hist is not None else None,
        rsi_4h=float(rsi_4h) if rsi_4h is not None else None,
        higher_tf_trend=_higher_tf_trend(
            float(macd_4h_hist) if macd_4h_hist is not None else None,
            float(rsi_4h) if rsi_4h is not None else None,
        ),
        atr=_atr(recent),
        level_name=level_name,
        setup_type=setup_type,
        setup_direction=setup_direction,
        tags=tuple(tags),
    )
