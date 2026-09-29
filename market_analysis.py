#!/usr/bin/env python3
"""
MarketAnalysis -- "what is happening right now." A thin, point-in-time-safe
aggregator over already-validated components (OpportunitySnapshot,
TrendAnalysis, VisionAnalysis) plus current/completed 4h candle context.
Not a rebuild of any of those -- this is the Phase 3 input type that
replaces OpportunitySnapshot as the primary point-in-time payload, while
reusing it as a nested field.

Built and persisted every 15-minute tick regardless of thesis state
(decided default) -- vision is only computed when about to feed a
thesis-related LLM call (same cost discipline as Phase 2), so `vision` is
Optional here even though `snapshot`/`trend` are always present when data
permits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from mtf_4h_candle_context import completed_4h_candle, developing_4h_candle
from opportunity_snapshot import OpportunitySnapshot
from trend_analysis import TrendAnalysis
from vision_analysis import VisionAnalysis


@dataclass(frozen=True)
class MarketAnalysis:
    market_analysis_id: str
    timestamp: datetime
    symbol: str
    snapshot: OpportunitySnapshot
    trend: Optional[TrendAnalysis]
    vision: Optional[VisionAnalysis]
    candle_4h_current: Optional[dict]
    candle_4h_completed: Optional[dict]
    chart_path: Optional[str]

    def to_dict(self) -> dict:
        return {
            "market_analysis_id": self.market_analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "snapshot": self.snapshot.to_dict(),
            "trend": dict(self.trend.__dict__) if self.trend is not None else None,
            "vision": dict(self.vision.__dict__) if self.vision is not None else None,
            "candle_4h_current": self.candle_4h_current,
            "candle_4h_completed": self.candle_4h_completed,
            "chart_path": self.chart_path,
        }


def build_market_analysis(
    market_analysis_id: str,
    snapshot: OpportunitySnapshot,
    trend: Optional[TrendAnalysis],
    df_1m: pd.DataFrame,
    vision: Optional[VisionAnalysis] = None,
    chart_path: Optional[str] = None,
) -> MarketAnalysis:
    """`df_1m` may extend past snapshot.timestamp -- both 4h-candle helpers
    truncate internally before reading anything, matching every other
    builder in this codebase (opportunity_snapshot.build_opportunity_snapshot,
    etc.)."""
    return MarketAnalysis(
        market_analysis_id=market_analysis_id,
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
        snapshot=snapshot,
        trend=trend,
        vision=vision,
        candle_4h_current=developing_4h_candle(df_1m, snapshot.timestamp),
        candle_4h_completed=completed_4h_candle(df_1m, snapshot.timestamp),
        chart_path=chart_path,
    )
