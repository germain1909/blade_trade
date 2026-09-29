#!/usr/bin/env python3
"""
TradePlan configuration -- versioned like session_tracker.ScannerConfig.
FROZEN once used for a real experiment run. All defaults here are V1
defaults, not values optimized against the 12-episode pilot's outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

TRADE_PLAN_CONFIG_VERSION = "v1"


@dataclass
class TradePlanConfig:
    version: str = TRADE_PLAN_CONFIG_VERSION
    max_wait_bars: int = 120           # ~2h -- half the default thesis hold horizon; clipped further to
                                        # the parent thesis's own remaining horizon at plan-formation time
    max_hold_bars: int = 240           # ~4h hard ceiling on a held position (post-ENTERED)
    hold_confirm_bars: int = 3         # bars price must remain beyond a confirmation distance to trigger
    confirm_distance_ticks: int = 8    # matches Phase 1's touch_ticks convention
    touch_dist_ticks: int = 8          # retest / pullback "close enough" tolerance
    trigger_buffer_ticks: int = 2
    invalidation_buffer_ticks: int = 2
    target_r_multiple_min: float = 0.5
    target_r_multiple_max: float = 5.0
