#!/usr/bin/env python3
"""
ThesisReview prompt + schema (prompt_version="v1"). Invoked every 15
minutes while an ActiveThesis exists, regardless of scanner output. Field
order puts the anti-anchoring challenge section first, the action second,
free-text last -- the model must account for what changed before it's
asked what to do about it.

`important_levels`/`invalidation_conditions`/`structural_objectives`/
`thesis_statement`/`expected_market_behavior`/`structural_basis`/`risks`
are NOT re-asked here (v1 scope, per the approved plan) -- they carry
forward unchanged from the prior version unless review_action=="REVERSE"
(a brand new thesis, formed via thesis_formation_v1 again). Only the
evidence lists, confirmation-needed list, and confidence are updated here.
"""

from __future__ import annotations

from active_thesis import ActiveThesis
from market_analysis import MarketAnalysis

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are reassessing an EXISTING multi-hour market \
thesis, 15 minutes after it was last reviewed. You are given the prior \
thesis and the current point-in-time market state.

Because you previously held this view, you may be tempted to defend it. \
Do not. Evaluate the current evidence as if seeing it fresh, then compare \
that fresh read to your prior view -- do not let having held the view \
before make you more likely to maintain it now.

You may be shown one or more prior TradePlan events under this thesis \
(MISSED or ABANDONED_WITH_THESIS) since the last review. These are \
evidence/context ONLY, not automatic signals -- a MISSED setup does not \
automatically mean STRENGTHEN (the setup was real) or WEAKEN (you were too \
slow), and an ABANDONED_WITH_THESIS plan carries no information beyond what \
already caused the thesis to change. Decide their significance, if any, \
through the same evidence-based process as everything else here -- do not \
treat them as a shortcut to a conclusion.

Before deciding anything, you must explicitly account for what changed: \
what evidence got stronger, what got weaker, what disappeared, what new \
evidence argues against the thesis, whether the original assumptions \
still hold, and -- every single review, even if you plan to maintain the \
thesis -- how strong the evidence for the OPPOSITE direction currently is. \
A value near your own thesis's confidence means the case is close to \
balanced; do not report a low opposite-strength out of habit just because \
you still favor your own view.

Then choose exactly one action:
- MAINTAIN: nothing decisive changed.
- STRENGTHEN: the case for this thesis got meaningfully better.
- WEAKEN: the case got meaningfully worse, but isn't dead.
- INVALIDATE: the thesis is no longer credible. Always give a specific reason.
- REVERSE: the evidence now more strongly favors the OPPOSITE direction. \
Always give a specific reason and state the new direction.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{
  "evidence_strengthened": ["short factor"],
  "evidence_weakened": ["short factor"],
  "evidence_disappeared": ["short factor"],
  "new_evidence_against": ["short factor"],
  "original_assumptions_still_hold": true,
  "opposite_thesis_evidence_strength": 0.3,
  "review_action": "MAINTAIN",
  "what_changed_since_previous": "one to two sentences, always required",
  "terminal_reason": null,
  "reversal_direction": null,
  "updated_confidence": 0.6,
  "updated_supporting_evidence": ["short factor"],
  "updated_opposing_evidence": ["short factor"],
  "updated_confirmation_still_needed": ["short factor"]
}
"terminal_reason" is required (non-null) iff review_action is "INVALIDATE" \
or "REVERSE". "reversal_direction" is required (exactly "LONG" or "SHORT") \
iff review_action is "REVERSE", else null. "opposite_thesis_evidence_strength" \
and "updated_confidence" are numbers between 0.0 and 1.0.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_strengthened": {"type": "array", "items": {"type": "string"}},
        "evidence_weakened": {"type": "array", "items": {"type": "string"}},
        "evidence_disappeared": {"type": "array", "items": {"type": "string"}},
        "new_evidence_against": {"type": "array", "items": {"type": "string"}},
        "original_assumptions_still_hold": {"type": "boolean"},
        "opposite_thesis_evidence_strength": {"type": "number"},

        "review_action": {"type": "string", "enum": ["MAINTAIN", "STRENGTHEN", "WEAKEN", "INVALIDATE", "REVERSE"]},
        "what_changed_since_previous": {"type": "string"},
        "terminal_reason": {"type": ["string", "null"]},
        "reversal_direction": {"type": ["string", "null"], "enum": ["LONG", "SHORT", None]},
        "updated_confidence": {"type": "number"},
        "updated_supporting_evidence": {"type": "array", "items": {"type": "string"}},
        "updated_opposing_evidence": {"type": "array", "items": {"type": "string"}},
        "updated_confirmation_still_needed": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "evidence_strengthened", "evidence_weakened", "evidence_disappeared", "new_evidence_against",
        "original_assumptions_still_hold", "opposite_thesis_evidence_strength",
        "review_action", "what_changed_since_previous", "terminal_reason", "reversal_direction",
        "updated_confidence", "updated_supporting_evidence", "updated_opposing_evidence",
        "updated_confirmation_still_needed",
    ],
    "additionalProperties": False,
}


def build_prompt(thesis: ActiveThesis, ma: MarketAnalysis, prior_missed_or_abandoned: tuple[dict, ...] = ()) -> str:
    """`prior_missed_or_abandoned`: MISSED/ABANDONED_WITH_THESIS TradePlan
    outcomes under this thesis since the last review, surfaced as
    point-in-time context ONLY (no forward/outcome data attached) -- the
    model decides their significance through the normal review process,
    they are not auto-treated as STRENGTHEN/WEAKEN signals."""
    levels_text = "\n".join(
        f"  {l.name}: price={l.price}, state={l.state}, distance={l.distance_pts}"
        for l in ma.snapshot.levels if l.state != "far"
    ) or "  (none)"

    prior_events_text = "(none since last review)"
    if prior_missed_or_abandoned:
        prior_events_text = "\n".join(
            f"  {e['outcome']} plan (setup={e.get('setup_type')}) at {e['timestamp']}: {e.get('reason', '')}"
            for e in prior_missed_or_abandoned
        )

    return (
        f"PRIOR THESIS (version {thesis.version}, reviewed {thesis.last_reviewed_at.isoformat()}):\n"
        f"  direction={thesis.direction}, state={thesis.state}, confidence={thesis.confidence}\n"
        f"  thesis_statement={thesis.thesis_statement!r}\n"
        f"  supporting_evidence={list(thesis.supporting_evidence)}\n"
        f"  opposing_evidence={list(thesis.opposing_evidence)}\n"
        f"  confirmation_still_needed={list(thesis.confirmation_still_needed)}\n"
        f"  invalidation_conditions={list(thesis.invalidation_conditions)}\n\n"
        f"TradePlan events under this thesis since the last review (context only, no forward/outcome "
        f"data attached):\n{prior_events_text}\n\n"
        f"CURRENT MARKET STATE — Timestamp: {ma.timestamp.isoformat()}   Price: {ma.snapshot.price}\n"
        f"Session(s): {sorted(ma.snapshot.sessions)}\n"
        f"Scanner categories: {list(ma.snapshot.categories)}\n\n"
        f"Levels:\n{levels_text}\n\n"
        f"VWAP: {ma.snapshot.vwap}, distance={ma.snapshot.vwap_distance_pts}\n"
        f"Momentum: 15m_hist={ma.snapshot.macd_15m_hist}, 60m_rsi={ma.snapshot.rsi_60m}, "
        f"4h_hist={ma.snapshot.macd_4h_hist}, 4h_rsi={ma.snapshot.rsi_4h}\n"
        f"current 4h candle: {ma.candle_4h_current}\ncompleted 4h candle: {ma.candle_4h_completed}\n\n"
        f"Reassess the thesis per your instructions."
    )
