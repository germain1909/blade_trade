#!/usr/bin/env python3
"""
TradePlan -- an immutable, machine-checkable conditional plan formed under
an ActiveThesis. Pairs with prompts/trade_plan_v1.py. Once created, a
TradePlan row is NEVER edited; arming/disarming is tracked separately
(trade_plan_arming.py + the trade_plan_arming_events table), and the plan
resolves to exactly one terminal outcome via trade_plan_replay.py.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from active_thesis import ActiveThesis
from llm_provider import LLMProvider
from market_analysis import MarketAnalysis
from prompts import trade_plan_v1
from setup_definitions import (
    INVALIDATION_CONDITION_FOR,
    MOMENTUM_REFERENCE,
    SETUP_TYPES,
    TRIGGER_CONDITION_FOR,
    compute_candidate_setups,
)
from trade_plan_candidates import build_candidate_structures, resolve_structure_price
from trade_plan_config import TradePlanConfig


@dataclass(frozen=True)
class TradePlan:
    plan_id: str
    thesis_id: str
    thesis_version_id_formed: str
    market_analysis_id: str
    created_at: datetime
    symbol: str
    direction: str

    plan_status: str  # "CANDIDATE" | "NO_PLAN" -- the model's own output; "ARMED" is never a model value
    no_plan_reason: Optional[str]

    setup_type: Optional[str]
    reference_price: Optional[float]  # CANDIDATE-formation price; the distance anchor for MOMENTUM_CONTINUATION

    trigger_level_name: Optional[str]
    trigger_level_price: Optional[float]
    trigger_condition: Optional[str]

    invalidation_level_name: Optional[str]
    invalidation_level_price: Optional[float]
    invalidation_condition: Optional[str]

    target_choice_type: Optional[str]
    target_level_name: Optional[str]
    target_r_multiple: Optional[float]
    target_price_hint: Optional[float]  # resolved now for a LEVEL target; for R_MULTIPLE, final
                                         # target_price can only be computed once entry price is known
                                         # (trade_plan_replay.py does that at ENTERED time)

    max_wait_bars: int
    max_hold_bars: int

    thesis_restatement: Optional[str]
    candidate_levels_json: tuple[dict, ...]
    model_name: str
    provider: str
    prompt_version: str
    inference_settings_json: dict
    raw_response: str
    latency_s: float

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["created_at"] = self.created_at.isoformat()
        d["candidate_levels_json"] = list(self.candidate_levels_json)
        return d


def form_trade_plan(
    provider: LLMProvider, thesis: ActiveThesis, ma: MarketAnalysis, cfg: TradePlanConfig,
    tick_size: float = 0.25,
) -> TradePlan:
    """Invoked when thesis.state in {DEVELOPING, STRONG} and no live plan
    exists under this thesis. Returns a TradePlan with plan_status ==
    'CANDIDATE' or 'NO_PLAN' -- arming happens later, deterministically,
    never here."""
    bars_since_thesis_created = int((ma.timestamp - thesis.created_at).total_seconds() // 60)
    remaining_horizon_bars = max(0, thesis.intended_horizon_bars - bars_since_thesis_created)
    effective_max_wait_bars = min(cfg.max_wait_bars, remaining_horizon_bars)
    effective_max_hold_bars = min(cfg.max_hold_bars, remaining_horizon_bars)

    candidate_setups = compute_candidate_setups(ma.snapshot, ma.trend, thesis.direction, cfg, tick_size)
    setup_types_available = tuple(candidate_setups.keys())
    trigger_names_available = tuple(sorted({name for names in candidate_setups.values() for name in names}))

    prompt, target_structures = trade_plan_v1.build_prompt(
        ma, thesis.direction, thesis.thesis_statement, candidate_setups,
    )
    target_names = tuple(c["name"] for c in target_structures)
    schema = trade_plan_v1.build_schema(setup_types_available, trigger_names_available, target_names)

    t0 = time.time()
    parsed, raw = provider.complete_structured(
        system=trade_plan_v1.SYSTEM_PROMPT, user=prompt,
        schema=schema, schema_name="trade_plan_v1",
    )
    latency = time.time() - t0

    plan_status = str(parsed.get("plan_status", "NO_PLAN")).strip().upper()
    if plan_status not in ("CANDIDATE", "NO_PLAN"):
        plan_status = "NO_PLAN"

    common_kwargs = dict(
        plan_id=str(uuid.uuid4()), thesis_id=thesis.thesis_id,
        thesis_version_id_formed=f"{thesis.thesis_id}:{thesis.version}",
        market_analysis_id=ma.market_analysis_id, created_at=ma.timestamp,
        symbol=ma.symbol, direction=thesis.direction,
        candidate_levels_json=target_structures,
        model_name=provider.config.model, provider="OpenAICompatProvider",
        prompt_version=trade_plan_v1.PROMPT_VERSION,
        inference_settings_json={"temperature": provider.config.temperature, "base_url": provider.config.base_url},
        raw_response=raw, latency_s=latency,
    )

    if plan_status == "NO_PLAN":
        return TradePlan(
            plan_status="NO_PLAN", no_plan_reason=parsed.get("no_plan_reason"),
            setup_type=None, reference_price=None,
            trigger_level_name=None, trigger_level_price=None, trigger_condition=None,
            invalidation_level_name=None, invalidation_level_price=None, invalidation_condition=None,
            target_choice_type=None, target_level_name=None, target_r_multiple=None, target_price_hint=None,
            max_wait_bars=effective_max_wait_bars, max_hold_bars=effective_max_hold_bars,
            thesis_restatement=None, **common_kwargs,
        )

    setup_type = str(parsed.get("setup_type", "")).strip().upper()
    trigger_level_name = parsed.get("trigger_level_name")
    if setup_type not in SETUP_TYPES:
        raise ValueError(f"trade_plan returned an unrecognized setup_type: {parsed!r}")
    valid_triggers = candidate_setups.get(setup_type, ())
    if trigger_level_name not in valid_triggers:
        raise ValueError(
            f"trade_plan chose trigger_level_name={trigger_level_name!r} which is not a valid "
            f"candidate for setup_type={setup_type!r} (valid: {valid_triggers})"
        )

    is_momentum = setup_type == "MOMENTUM_CONTINUATION"
    reference_price = ma.snapshot.price if is_momentum else None
    trigger_level_price = None if is_momentum else resolve_structure_price(trigger_level_name, ma.snapshot, ma.trend)
    if not is_momentum and trigger_level_price is None:
        raise ValueError(f"could not resolve a price for trigger_level_name={trigger_level_name!r}")

    # v1 simplification (approved plan §7): invalidation always keyed to the same
    # structural reference as the trigger.
    invalidation_level_name = trigger_level_name
    invalidation_level_price = trigger_level_price

    target_choice_type = str(parsed.get("target_choice_type", "")).strip().upper()
    target_level_name = parsed.get("target_level_name")
    target_r_multiple = parsed.get("target_r_multiple")
    target_price_hint = None
    if target_choice_type == "LEVEL":
        if not target_level_name:
            raise ValueError(f"target_choice_type=LEVEL but no target_level_name given: {parsed!r}")
        target_price_hint = resolve_structure_price(target_level_name, ma.snapshot, ma.trend)
        target_r_multiple = None
    elif target_choice_type == "R_MULTIPLE":
        try:
            target_r_multiple = float(target_r_multiple)
        except (TypeError, ValueError):
            raise ValueError(f"target_choice_type=R_MULTIPLE but target_r_multiple is invalid: {parsed!r}")
        target_r_multiple = max(cfg.target_r_multiple_min, min(cfg.target_r_multiple_max, target_r_multiple))
        target_level_name = None
    else:
        raise ValueError(f"trade_plan returned an invalid target_choice_type: {parsed!r}")

    return TradePlan(
        plan_status="CANDIDATE", no_plan_reason=None,
        setup_type=setup_type, reference_price=reference_price,
        trigger_level_name=None if is_momentum else trigger_level_name,
        trigger_level_price=trigger_level_price,
        trigger_condition=TRIGGER_CONDITION_FOR[setup_type],
        invalidation_level_name=None if is_momentum else invalidation_level_name,
        invalidation_level_price=invalidation_level_price,
        invalidation_condition=INVALIDATION_CONDITION_FOR[setup_type],
        target_choice_type=target_choice_type, target_level_name=target_level_name,
        target_r_multiple=target_r_multiple, target_price_hint=target_price_hint,
        max_wait_bars=effective_max_wait_bars, max_hold_bars=effective_max_hold_bars,
        thesis_restatement=str(parsed.get("thesis_restatement", "") or ""),
        **common_kwargs,
    )
