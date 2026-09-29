#!/usr/bin/env python3
"""
Deterministic trendline/channel analysis — adapted from QuantHarness's
graph_util.py (fit_trendlines_high_low / optimize_slope / check_trend_line).

The math is copied faithfully (least-squares seed + iterative slope
optimization against high/low pivots). The interface is changed: this
takes an already-truncated bars array directly, rather than QuantHarness's
pattern of being handed a full dataframe and internally slicing
`data.iloc[-50:]` — that internal slicing is exactly the lookahead risk
this project avoids; a caller here cannot accidentally pass future bars
through it, because there is no dataframe-wide slicing step at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_BARS = 10


@dataclass(frozen=True)
class TrendAnalysis:
    support_slope: float
    support_price_now: float
    resistance_slope: float
    resistance_price_now: float
    channel_width_pts: float
    channel_width_atr_mult: float
    price_position_in_channel: float
    distance_to_support_pts: float
    distance_to_resistance_pts: float
    channel_direction: str  # "rising" | "falling" | "flat"
    compression: bool
    bars_used: int


def _check_trend_line(support: bool, pivot: int, slope: float, y: pd.Series) -> float:
    intercept = -slope * pivot + y.iloc[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y.values
    if support and diffs.max() > 1e-5:
        return -1.0
    if not support and diffs.min() < -1e-5:
        return -1.0
    return float((diffs ** 2.0).sum())


def _optimize_slope(support: bool, pivot: int, init_slope: float, y: pd.Series) -> tuple[float, float]:
    slope_unit = (y.max() - y.min()) / len(y)
    if slope_unit == 0:
        return init_slope, -init_slope * pivot + y.iloc[pivot]

    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step
    best_slope = init_slope
    best_err = _check_trend_line(support, pivot, init_slope, y)
    assert best_err >= 0.0, "initial slope violates support/resistance constraint"

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = _check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = _check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                # Both directions from this pivot immediately violate the
                # support/resistance constraint — happens on near-perfectly-linear
                # data (e.g. a dead/no-volume stretch) where the pivot lands away
                # from either endpoint. The unoptimized least-squares-anchored
                # line (already valid, since best_err succeeded above) is a fine
                # fallback; there's no valid direction to refine it in.
                return best_slope, -best_slope * pivot + y.iloc[pivot]
            get_derivative = False

        test_slope = best_slope - slope_unit * curr_step if derivative > 0.0 else best_slope + slope_unit * curr_step
        test_err = _check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return best_slope, -best_slope * pivot + y.iloc[pivot]


def _fit_trendlines_high_low(high: pd.Series, low: pd.Series, close: pd.Series):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close.values, 1)
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = int((high.values - line_points).argmax())
    lower_pivot = int((low.values - line_points).argmin())
    support_coefs = _optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = _optimize_slope(False, upper_pivot, coefs[0], high)
    return support_coefs, resist_coefs


def compute_trend_analysis(recent_bars: tuple[tuple, ...], daily_atr: float | None = None) -> TrendAnalysis | None:
    """
    recent_bars: tuple of (ts, open, high, low, close, volume), oldest -> newest,
    already truncated by the caller (e.g. OpportunitySnapshot.recent_bars).
    Returns None if there aren't enough bars for a meaningful fit.
    """
    if len(recent_bars) < _MIN_BARS:
        return None

    high = pd.Series([b[2] for b in recent_bars])
    low = pd.Series([b[3] for b in recent_bars])
    close = pd.Series([b[4] for b in recent_bars])
    n = len(close)

    (support_slope, support_intercept), (resist_slope, resist_intercept) = _fit_trendlines_high_low(high, low, close)

    support_now = support_slope * (n - 1) + support_intercept
    resistance_now = resist_slope * (n - 1) + resist_intercept
    support_start = support_intercept
    resistance_start = resist_intercept

    channel_width_pts = resistance_now - support_now
    width_start = resistance_start - support_start

    price_now = float(close.iloc[-1])
    price_position = ((price_now - support_now) / channel_width_pts) if channel_width_pts != 0 else 0.5

    avg_slope = (support_slope + resist_slope) / 2.0
    price_scale = max(abs(price_now), 1e-9)
    slope_eps = 1e-5 * price_scale  # points/bar below which the channel counts as flat
    if abs(avg_slope) < slope_eps:
        direction = "flat"
    else:
        direction = "rising" if avg_slope > 0 else "falling"

    compression = channel_width_pts < width_start

    channel_width_atr_mult = (channel_width_pts / daily_atr) if daily_atr else float("nan")

    return TrendAnalysis(
        support_slope=float(support_slope),
        support_price_now=float(support_now),
        resistance_slope=float(resist_slope),
        resistance_price_now=float(resistance_now),
        channel_width_pts=float(channel_width_pts),
        channel_width_atr_mult=float(channel_width_atr_mult),
        price_position_in_channel=float(price_position),
        distance_to_support_pts=float(price_now - support_now),
        distance_to_resistance_pts=float(resistance_now - price_now),
        channel_direction=direction,
        compression=bool(compression),
        bars_used=n,
    )
