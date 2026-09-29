#!/usr/bin/env python3
"""
Current (still-forming) and most-recently-completed 4-hour candle context,
for MarketAnalysis. Reuses mtf_indicators.resample_ohlc (already exported,
already point-in-time-safe) for the completed side; the developing side is
a simple aggregate of 1-minute rows since the last 4h boundary through the
query timestamp -- point-in-time-safe by construction, since it only ever
aggregates rows <= timestamp. mtf_indicators.py itself is not modified.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from mtf_indicators import resample_ohlc


def completed_4h_candle(df_1m: pd.DataFrame, timestamp: datetime) -> Optional[dict]:
    """Most recently CLOSED 4h bar as of `timestamp` (bins are right-closed/
    right-labeled by resample_ohlc, so a bin's label is its close time --
    only bins whose label <= timestamp have actually closed)."""
    truncated = df_1m.loc[df_1m["timestamp"] <= timestamp]
    if truncated.empty:
        return None
    bars4h = resample_ohlc(truncated, "4h")
    closed = bars4h.loc[bars4h.index <= timestamp]
    if closed.empty:
        return None
    row = closed.iloc[-1]
    return {
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "closed_at": closed.index[-1].isoformat(),
    }


def developing_4h_candle(df_1m: pd.DataFrame, timestamp: datetime) -> Optional[dict]:
    """The still-forming 4h bar containing `timestamp`, aggregated from
    only the 1-minute bars <= timestamp within its (not-yet-closed) window."""
    truncated = df_1m.loc[df_1m["timestamp"] <= timestamp]
    if truncated.empty:
        return None
    # Find the 1-minute rows belonging to the same (still-open-or-just-closed) 4h
    # bin as `timestamp` -- right-closed/right-labeled, matching resample_ohlc's convention.
    binned = truncated.set_index("timestamp").resample("4h", label="right", closed="right")
    for label, group in binned:
        if group.empty:
            continue
        if group.index.min() <= timestamp <= label:
            return {
                "open": float(group["open"].iloc[0]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close_so_far": float(group["close"].iloc[-1]),
                "bars_so_far": int(len(group)),
                "window_closes_at": label.isoformat(),
            }
    return None
