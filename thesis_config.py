#!/usr/bin/env python3
"""
ActiveThesis configuration -- versioned like session_tracker.ScannerConfig.
FROZEN once used for a real experiment run: a deliberate change is a new
tagged version, never a silent edit to v1's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

THESIS_CONFIG_VERSION = "v1"  # intended_horizon_bars=360 (~6h, midpoint of the 4-8h target) is a V1
                                # default, not a value optimized against the 12-episode pilot's outcomes.


@dataclass
class ThesisConfig:
    version: str = THESIS_CONFIG_VERSION
    intended_horizon_bars: int = 360  # ~6 hours of 1-minute bars
