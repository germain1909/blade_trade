#!/usr/bin/env python3
"""
The closed candidate structure list for TradePlan/ActiveThesis target and
level references -- reuses level_relevance.relevant_levels (Phase 2 V2)
verbatim, plus VWAP and trendline references. No other name is ever
presented to or accepted from the model. Separate from
setup_definitions.compute_candidate_setups, which is about STATE-based
trigger-level eligibility per setup, not general relevance/naming.
"""

from __future__ import annotations

from typing import Optional

from level_relevance import relevant_levels
from opportunity_snapshot import OpportunitySnapshot
from trend_analysis import TrendAnalysis


def build_candidate_structures(
    snapshot: OpportunitySnapshot, trend: Optional[TrendAnalysis],
) -> tuple[dict, ...]:
    """Full enumerated {"name","price","state"} candidate list -- for
    schema-enum construction and for persistence (candidate_levels_json)."""
    out: list[dict] = []
    for l in relevant_levels(snapshot.levels, snapshot.daily_atr):
        out.append({"name": l.name, "price": l.price, "state": l.state})
    if snapshot.vwap is not None:
        out.append({"name": "VWAP", "price": snapshot.vwap, "state": "reference"})
    if trend is not None:
        out.append({"name": "TrendlineSupport", "price": trend.support_price_now, "state": "reference"})
        out.append({"name": "TrendlineResistance", "price": trend.resistance_price_now, "state": "reference"})
    return tuple(out)


def resolve_structure_price(
    name: str, snapshot: OpportunitySnapshot, trend: Optional[TrendAnalysis],
) -> Optional[float]:
    """Python resolves the exact price for a model-chosen NAME -- the model
    never states a price itself."""
    if name == "VWAP":
        return snapshot.vwap
    if name == "TrendlineSupport":
        return trend.support_price_now if trend is not None else None
    if name == "TrendlineResistance":
        return trend.resistance_price_now if trend is not None else None
    for l in snapshot.levels:
        if l.name == name:
            return l.price
    return None
