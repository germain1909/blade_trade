#!/usr/bin/env python3
"""
Multi-timeframe indicator helpers: 15m MACD, 60m RSI, 4hr MACD, 4hr RSI —
computed from 1-minute OHLC bars and aligned back onto the 1-minute index
with no lookahead (each higher-timeframe value only becomes visible once
that bar has actually closed).
"""

from __future__ import annotations

import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """Where does the current value sit (0-1) within its own trailing range?"""
    lo = s.rolling(window, min_periods=window // 2).min()
    hi = s.rolling(window, min_periods=window // 2).max()
    span = (hi - lo).replace(0, pd.NA)
    return (s - lo) / span


def macd_line_shape(macd_line: pd.Series, range_window: int = 100, smooth: int = 5,
                     extreme_pctile: float = 0.20, persist_bars: int = 2):
    """Overbought/oversold + rounding-turn detection on the raw MACD line
    (not the histogram), tuned to catch SUBSTANTIAL reversals — genuine
    multi-bar rounded arcs — rather than single-bar noise:

    - "extreme" is judged against a longer trailing range (`range_window`,
      default 100 15m bars ~ 1 day) with a tighter percentile cutoff, so it
      has to be extreme relative to real recent history, not just the last
      few hours of chop.
    - "rounding over" requires a smoothed version of the line (`smooth`
      bars) to keep moving the new direction for `persist_bars` bars in a
      row right after having moved the other way — a sustained curve, not
      a one-bar snap that could reverse right back.
    """
    pctile = rolling_percentile(macd_line, range_window)
    oversold = pctile <= extreme_pctile
    overbought = pctile >= (1 - extreme_pctile)

    smoothed = macd_line.rolling(smooth, min_periods=smooth).mean()
    d = smoothed.diff()

    rising_streak  = (d > 0).rolling(persist_bars).sum() == persist_bars
    falling_streak = (d < 0).rolling(persist_bars).sum() == persist_bars
    rounding_bottom = rising_streak & (d.shift(persist_bars) <= 0)
    rounding_top    = falling_streak & (d.shift(persist_bars) >= 0)

    return oversold, overbought, rounding_bottom, rounding_top


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """df must have a 'timestamp' column. Bins are labeled/closed on the RIGHT
    (bin timestamp = close time of that bar) so alignment never leaks the
    still-forming bar's value backward."""
    idx = df.set_index("timestamp")
    return (idx.resample(rule, label="right", closed="right")
               .agg({"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"})
               .dropna())


def compute_mtf_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed by 1-minute timestamp with columns:
    15m_hist, 15m_bull, 15m_cross_up, 15m_cross_down,
    60m_rsi, 60m_rsi_state,
    4h_hist, 4h_bull, 4h_cross_up, 4h_cross_down,
    4h_rsi, 4h_rsi_state
    """
    bars15 = resample_ohlc(df_1m, "15min")
    bars60 = resample_ohlc(df_1m, "60min")
    bars4h = resample_ohlc(df_1m, "4h")

    m15, s15, h15 = macd(bars15["close"])
    r60 = rsi(bars60["close"])
    _, _, h4h = macd(bars4h["close"])
    r4h = rsi(bars4h["close"])

    oversold15, overbought15, round_bot15, round_top15 = macd_line_shape(m15)

    f15 = pd.DataFrame({
        "15m_hist": h15,
        "15m_bull": h15 > 0,
        "15m_cross_up":   (h15 > 0) & (h15.shift(1) <= 0),
        "15m_cross_down": (h15 < 0) & (h15.shift(1) >= 0),
        "15m_macd_oversold":   oversold15,
        "15m_macd_overbought": overbought15,
        "15m_macd_rounding_bottom": round_bot15,
        "15m_macd_rounding_top":    round_top15,
        # the combined "at the bottom, turning up" / "at the top, turning down" signal
        "15m_macd_bottom": oversold15 & round_bot15,
        "15m_macd_top":    overbought15 & round_top15,
    })
    # "did this happen anytime in the last RECENT_BARS 15m bars" — decouples the MACD
    # confirmation from however long a level's approach window happens to last, since
    # price often blows through an approach zone in just 1-2 bars.
    RECENT_BARS = 8  # 8 x 15m = 2 hours
    f15["15m_macd_bottom_recent"] = f15["15m_macd_bottom"].rolling(RECENT_BARS, min_periods=1).max().astype(bool)
    f15["15m_macd_top_recent"]    = f15["15m_macd_top"].rolling(RECENT_BARS, min_periods=1).max().astype(bool)
    f60 = pd.DataFrame({
        "60m_rsi": r60,
        "60m_rsi_state": pd.cut(r60, [-1, 30, 70, 101],
                                 labels=["oversold", "neutral", "overbought"]),
    })
    f4h = pd.DataFrame({
        "4h_hist": h4h,
        "4h_bull": h4h > 0,
        "4h_cross_up":   (h4h > 0) & (h4h.shift(1) <= 0),
        "4h_cross_down": (h4h < 0) & (h4h.shift(1) >= 0),
        "4h_rsi": r4h,
        "4h_rsi_state": pd.cut(r4h, [-1, 30, 70, 101],
                                labels=["oversold", "neutral", "overbought"]),
    })

    base = pd.DataFrame({"ts": df_1m["timestamp"].sort_values().reset_index(drop=True)})
    merged = base
    for feat in (f15, f60, f4h):
        feat = feat.reset_index().rename(columns={"timestamp": "ts"})
        merged = pd.merge_asof(merged.sort_values("ts"), feat.sort_values("ts"), on="ts", direction="backward")

    return merged.set_index("ts")


def tags_for_row(row) -> list[str]:
    """Auto-tagged pattern labels present at a given 1-minute snapshot."""
    tags: list[str] = []

    if pd.notna(row.get("15m_bull")):
        tags.append("15m_macd_bull" if row["15m_bull"] else "15m_macd_bear")
    if row.get("15m_cross_up"):
        tags.append("15m_macd_cross_up")
    if row.get("15m_cross_down"):
        tags.append("15m_macd_cross_down")
    if row.get("15m_macd_oversold"):
        tags.append("15m_macd_oversold")
    if row.get("15m_macd_overbought"):
        tags.append("15m_macd_overbought")
    if row.get("15m_macd_bottom"):
        tags.append("15m_macd_bottom")
    if row.get("15m_macd_top"):
        tags.append("15m_macd_top")

    if pd.notna(row.get("60m_rsi_state")):
        tags.append(f"60m_rsi_{row['60m_rsi_state']}")

    if pd.notna(row.get("4h_bull")):
        tags.append("4h_macd_bull" if row["4h_bull"] else "4h_macd_bear")
    if row.get("4h_cross_up"):
        tags.append("4h_macd_cross_up")
    if row.get("4h_cross_down"):
        tags.append("4h_macd_cross_down")

    if pd.notna(row.get("4h_rsi_state")):
        tags.append(f"4h_rsi_{row['4h_rsi_state']}")

    if "15m_macd_bull" in tags and "4h_macd_bull" in tags:
        tags.append("mtf_macd_aligned_bull")
    if "15m_macd_bear" in tags and "4h_macd_bear" in tags:
        tags.append("mtf_macd_aligned_bear")

    return tags
