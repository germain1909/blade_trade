#!/usr/bin/env python3
"""
ActiveThesis -- persistent, append-only, versioned multi-hour market
thesis. A review never mutates a prior row; it produces a new `version`
under the same `thesis_id`, or (on REVERSE) an entirely new `thesis_id`
with `parent_thesis_id` set. See RESEARCH_NOTES.md / the approved Phase 3
plan for the full state-machine rationale and anti-anchoring design.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bar_conditions import crossed_beyond
from opportunity_snapshot import OpportunitySnapshot

TERMINAL_STATES = {"INVALIDATED", "REVERSED", "COMPLETED", "EXPIRED"}
ACTIONABLE_STATES = {"DEVELOPING", "STRONG"}
_VALID_REVIEW_ACTIONS = {"MAINTAIN", "STRENGTHEN", "WEAKEN", "INVALIDATE", "REVERSE"}


@dataclass(frozen=True)
class ActiveThesis:
    thesis_id: str
    version: int
    parent_thesis_id: Optional[str]
    created_at: datetime
    last_reviewed_at: datetime
    symbol: str
    direction: str  # "LONG" | "SHORT"

    state: str  # CANDIDATE|DEVELOPING|STRONG|WEAKENING|INVALIDATED|REVERSED|COMPLETED|EXPIRED

    thesis_statement: str
    intended_horizon_bars: int
    confidence: float

    supporting_evidence: tuple[str, ...]
    opposing_evidence: tuple[str, ...]
    structural_basis: tuple[str, ...]
    important_levels: tuple[dict, ...]
    expected_market_behavior: str
    confirmation_still_needed: tuple[str, ...]
    invalidation_conditions: tuple[dict, ...]  # [{"name","condition","adverse_direction"}]
    structural_objectives: tuple[dict, ...]
    risks: tuple[str, ...]

    evidence_strengthened: tuple[str, ...]
    evidence_weakened: tuple[str, ...]
    evidence_disappeared: tuple[str, ...]
    new_evidence_against: tuple[str, ...]
    original_assumptions_still_hold: bool
    opposite_thesis_evidence_strength: float
    what_changed_since_previous: str

    review_action: str
    terminal_reason: Optional[str]

    market_analysis_id: str
    prior_thesis_version_id: Optional[str]

    model_name: str
    provider: str
    prompt_version: str
    inference_settings_json: dict
    raw_response: str
    latency_s: float

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["created_at"] = self.created_at.isoformat()
        d["last_reviewed_at"] = self.last_reviewed_at.isoformat()
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def next_state(current_state: str, review_action: str) -> str:
    """Pure state-machine transition. Does NOT apply the forced-invalidation
    circuit breaker (check_forced_invalidation) or horizon expiry
    (check_horizon_expired) -- callers must apply those separately, and
    they take precedence over whatever this function returns."""
    if review_action not in _VALID_REVIEW_ACTIONS:
        raise ValueError(f"unknown review_action: {review_action!r}")
    if current_state in TERMINAL_STATES:
        raise ValueError(f"cannot review a terminal thesis (state={current_state!r})")

    if review_action == "INVALIDATE":
        return "INVALIDATED"
    if review_action == "REVERSE":
        return "REVERSED"  # caller creates the new opposite-direction thesis separately
    if review_action == "MAINTAIN":
        return current_state
    if review_action == "STRENGTHEN":
        return {"CANDIDATE": "DEVELOPING", "DEVELOPING": "STRONG",
                "STRONG": "STRONG", "WEAKENING": "DEVELOPING"}[current_state]
    if review_action == "WEAKEN":
        return "WEAKENING" if current_state in ("CANDIDATE", "DEVELOPING", "STRONG") else current_state
    raise AssertionError("unreachable")


def check_forced_invalidation(
    thesis: ActiveThesis, snapshot: OpportunitySnapshot,
) -> Optional[str]:
    """Deterministic circuit breaker (anti-anchoring safeguard): if the
    thesis's OWN already-declared invalidation_conditions have fired
    against the CURRENT point-in-time snapshot, this returns a
    terminal_reason string that OVERRIDES whatever review_action the model
    returned -- INVALIDATE is forced regardless. Returns None if nothing
    has fired.

    Each invalidation_conditions entry: {"name": str, "condition":
    "CLOSE_BEYOND"|"TOUCH_BEYOND", "adverse_direction": "above"|"below"}.
    Uses the most recent bar in snapshot.recent_bars for high/low/close --
    already truncated to <= snapshot.timestamp by construction."""
    if not thesis.invalidation_conditions or not snapshot.recent_bars:
        return None
    last_bar = snapshot.recent_bars[-1]  # (ts, o, h, l, c, v)
    bar_high, bar_low, bar_close = last_bar[2], last_bar[3], last_bar[4]

    for cond in thesis.invalidation_conditions:
        name = cond.get("name")
        condition = cond.get("condition")
        adverse_direction = cond.get("adverse_direction")
        if not (name and condition and adverse_direction):
            continue
        price = _resolve_named_price(name, snapshot)
        if price is None:
            continue
        if crossed_beyond(adverse_direction, price, condition, bar_high, bar_low, bar_close):
            return f"forced invalidation: {name} crossed {adverse_direction} via {condition}"
    return None


def _resolve_named_price(name: str, snapshot: OpportunitySnapshot) -> Optional[float]:
    if name == "VWAP":
        return snapshot.vwap
    for l in snapshot.levels:
        if l.name == name:
            return l.price
    return None


def check_horizon_expired(thesis: ActiveThesis, current_timestamp: datetime) -> bool:
    """Python-enforced ceiling, not an LLM vote: has the thesis's own
    intended_horizon_bars elapsed since it was first created (version 0)?"""
    elapsed_minutes = (current_timestamp - thesis.created_at).total_seconds() / 60.0
    return elapsed_minutes >= thesis.intended_horizon_bars
