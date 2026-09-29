#!/usr/bin/env python3
"""
Deterministic arm/disarm consequence -- NEVER an LLM decision. A plan is
ARMED if and only if its parent thesis's most recently reviewed state is
STRONG; this is recomputed at every 15-minute ThesisReview, not a one-way
promotion. Recorded as an append-only event (ARMED/DISARMED), never as a
mutation of the immutable TradePlan row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class ArmingEvent:
    plan_id: str
    timestamp: datetime
    event: str  # "ARMED" | "DISARMED"
    thesis_version_id_at_event: str


def compute_arming_event(
    current_status: str, thesis_state: str, plan_id: str, timestamp: datetime, thesis_version_id: str,
) -> Optional[ArmingEvent]:
    """current_status: the plan's status immediately before this review
    ('CANDIDATE' or 'ARMED', reconstructed via current_plan_status below).
    Returns an event if a transition should occur this review, else None."""
    is_strong = thesis_state == "STRONG"
    if current_status == "CANDIDATE" and is_strong:
        return ArmingEvent(plan_id, timestamp, "ARMED", thesis_version_id)
    if current_status == "ARMED" and not is_strong:
        return ArmingEvent(plan_id, timestamp, "DISARMED", thesis_version_id)
    return None


def current_plan_status(events: list[ArmingEvent]) -> str:
    """Reconstructs a plan's current status from its append-only arming
    event history -- CANDIDATE if no events yet, else whatever the most
    recent event set."""
    if not events:
        return "CANDIDATE"
    latest = max(events, key=lambda e: e.timestamp)
    return "ARMED" if latest.event == "ARMED" else "CANDIDATE"
