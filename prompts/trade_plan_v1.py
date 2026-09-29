#!/usr/bin/env python3
"""
TradePlanFormation prompt + schema (prompt_version="v1"). Invoked once
per ActiveThesis while thesis.state is DEVELOPING or STRONG and no live
plan exists for it. The model's only real choices are plan_status,
setup_type (from the currently-valid candidate set), trigger_level_name
(from that setup's valid candidates), and target selection -- everything
execution-relevant (trigger_condition, invalidation_level/condition,
buffers, max_wait/max_hold) is Python-derived from setup_definitions.py's
fixed tables, never asked of the model. This keeps "inventing an arbitrary
untestable condition" structurally impossible.

setup_type's enum and trigger_level_name's enum are NOT made conditionally
dependent on each other in the schema itself (untested whether this LM
Studio build's json_schema mode reliably supports oneOf/if-then-else
constructs) -- trigger_level_name's enum is the union of valid names across
all currently-candidate setups, and the (setup_type, trigger_level_name)
PAIR is validated post-parse in trade_plan.py, matching the existing
"schema bounds, Python validates" pattern already used elsewhere in this
codebase (e.g. target_choice_type mutual exclusivity in phase2/phase3).
"""

from __future__ import annotations

from market_analysis import MarketAnalysis
from trade_plan_candidates import build_candidate_structures

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are proposing a CANDIDATE tactical trade plan for an \
already-formed multi-hour market thesis. You are NOT placing a trade -- a \
CANDIDATE plan only becomes executable later, automatically, once the \
thesis itself is judged strong enough (a separate, deterministic process \
you have no part in).

You may only choose a setup_type from the list of setups that are \
CURRENTLY structurally valid (given to you explicitly) -- do not propose a \
setup whose precondition isn't already true. You may only choose a \
trigger_level_name from the valid candidates for that setup. You never \
state a trigger condition, an invalidation condition, a price, or a bar \
count yourself -- these are all determined automatically from your \
setup_type choice. Your only remaining choice is the target: either name a \
structural level from the candidate list, or state a risk-multiple (a \
number between 0.5 and 5.0) if no clean structural target exists in range.

There is no requirement to propose a plan. If no valid setup exists, or \
none of the valid setups make sense given the thesis, return NO_PLAN.

Respond with ONLY a single JSON object and nothing else, in exactly this \
shape (no markdown fences, no extra keys):
{
  "plan_status": "CANDIDATE",
  "no_plan_reason": null,
  "setup_type": "BREAKOUT_RETEST",
  "trigger_level_name": "AsiaHigh",
  "target_choice_type": "LEVEL",
  "target_level_name": "WeeklyR1",
  "target_r_multiple": null,
  "thesis_restatement": "one sentence restating why this specific setup executes the thesis"
}
If plan_status is "NO_PLAN", "no_plan_reason" must be filled in and every \
other field null. target_choice_type must be exactly "LEVEL" or \
"R_MULTIPLE" -- exactly one of target_level_name/target_r_multiple must be \
non-null to match.
"""


def build_schema(setup_types: tuple[str, ...], trigger_level_names: tuple[str, ...],
                  target_names: tuple[str, ...]) -> dict:
    setups = list(setup_types) or ["__NONE__"]
    triggers = list(trigger_level_names) or ["__NONE__"]
    targets = list(target_names) or ["__NONE__"]
    return {
        "type": "object",
        "properties": {
            "plan_status": {"type": "string", "enum": ["CANDIDATE", "NO_PLAN"]},
            "no_plan_reason": {"type": ["string", "null"]},
            "setup_type": {"type": ["string", "null"], "enum": setups + [None]},
            "trigger_level_name": {"type": ["string", "null"], "enum": triggers + [None]},
            "target_choice_type": {"type": ["string", "null"], "enum": ["LEVEL", "R_MULTIPLE", None]},
            "target_level_name": {"type": ["string", "null"], "enum": targets + [None]},
            "target_r_multiple": {"type": ["number", "null"]},
            "thesis_restatement": {"type": ["string", "null"]},
        },
        "required": [
            "plan_status", "no_plan_reason", "setup_type", "trigger_level_name",
            "target_choice_type", "target_level_name", "target_r_multiple", "thesis_restatement",
        ],
        "additionalProperties": False,
    }


def build_prompt(ma: MarketAnalysis, thesis_direction: str, thesis_statement: str,
                  candidate_setups: dict[str, tuple[str, ...]]) -> tuple[str, tuple[dict, ...]]:
    """Returns (prompt_text, candidate_structures_for_target) for schema
    building and persistence."""
    target_structures = build_candidate_structures(ma.snapshot, ma.trend)
    setups_text = "\n".join(
        f"  {setup}: valid trigger_level_name options = {list(names)}"
        for setup, names in candidate_setups.items()
    ) or "  (none currently valid)"
    targets_text = "\n".join(f"  {c['name']}: price={c['price']}" for c in target_structures) or "  (none)"

    prompt = (
        f"Thesis: direction={thesis_direction}, statement={thesis_statement!r}\n\n"
        f"Timestamp: {ma.timestamp.isoformat()}   Price: {ma.snapshot.price}\n\n"
        f"Currently-valid setups and their trigger candidates (choose ONLY from these):\n{setups_text}\n\n"
        f"Candidate target structures (choose ONLY from these, or use target_r_multiple):\n{targets_text}\n\n"
        f"Propose a candidate plan, or return NO_PLAN, per your instructions."
    )
    return prompt, target_structures
