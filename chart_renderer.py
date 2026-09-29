#!/usr/bin/env python3
"""
Point-in-time chart renderer — adapted from QuantHarness's graph_util.py
(generate_trend_image/generate_kline_image), fixing two problems in the
original: (1) it internally sliced a full dataframe (`data.iloc[-50:]`),
which is a lookahead risk if a caller ever passed it more than it should
see; this version only ever accepts an already-truncated bars sequence and
never reaches back into a larger source. (2) it wrote to fixed filenames
(`trend_graph.png`), unsafe under any concurrency; this version takes an
explicit, caller-chosen output path.

A structural assert double-checks the truncation invariant itself, not
just caller discipline: no bar timestamped after `snapshot_timestamp` may
be in `recent_bars`, or rendering aborts before a PNG is ever produced.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from session_tracker import LevelState
from trend_analysis import TrendAnalysis

_LEVEL_COLORS = {
    "PDH": "#8899AA", "PDL": "#8899AA",
    "AsiaHigh": "#C9A227", "AsiaLow": "#C9A227",
    "LondonHigh": "#4472C4", "LondonLow": "#4472C4",
}


def render(
    recent_bars: tuple[tuple, ...],
    snapshot_timestamp: datetime,
    levels: tuple[LevelState, ...] = (),
    trend: Optional[TrendAnalysis] = None,
    vwap: Optional[float] = None,
    out_path: Optional[str] = None,
) -> tuple[bytes, str]:
    """
    recent_bars: (ts, open, high, low, close, volume) tuples, oldest -> newest.
    Returns (png_bytes, base64_str). Writes to out_path if given.
    """
    if not recent_bars:
        raise ValueError("render() requires at least one bar")

    max_bar_ts = max(b[0] for b in recent_bars)
    if max_bar_ts > snapshot_timestamp:
        raise ValueError(
            f"point-in-time violation: recent_bars contains a bar at {max_bar_ts} "
            f"after the snapshot timestamp {snapshot_timestamp}"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(recent_bars)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)

    for i, (ts, o, h, l, c, v) in enumerate(recent_bars):
        color = "#2F9C8D" if c >= o else "#C85A42"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.plot([i, i], [o, c], color=color, linewidth=4)

    for lvl in levels:
        if lvl.price is None:
            continue
        color = _LEVEL_COLORS.get(lvl.name, "#999999")
        ax.axhline(lvl.price, color=color, linestyle="--", linewidth=1, alpha=0.8)
        ax.text(n - 1, lvl.price, f" {lvl.name}", color=color, fontsize=8, va="center")

    if vwap is not None:
        ax.axhline(vwap, color="#7A4DB3", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.text(0, vwap, "VWAP ", color="#7A4DB3", fontsize=8, va="center", ha="right")

    if trend is not None:
        xs = [0, n - 1]
        support_ys = [trend.support_price_now - trend.support_slope * (n - 1), trend.support_price_now]
        resist_ys = [trend.resistance_price_now - trend.resistance_slope * (n - 1), trend.resistance_price_now]
        ax.plot(xs, support_ys, color="#333333", linewidth=1.3, alpha=0.9)
        ax.plot(xs, resist_ys, color="#333333", linewidth=1.3, alpha=0.9)

    ax.set_xlim(-1, n)
    ax.set_title(f"Point-in-time chart, {n} bars ending {recent_bars[-1][0]}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(png_bytes)

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return png_bytes, b64
