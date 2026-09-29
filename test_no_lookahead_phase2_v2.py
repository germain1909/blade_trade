#!/usr/bin/env python3
"""
Structural/quality tests for Phase 2 V2 (level_relevance.py,
output_quality.py, and a live schema round-trip). Must pass before
run_phase2_v2.py is run against real episodes.

Run directly: python3 test_no_lookahead_phase2_v2.py
(Tests 1-2 are pure/offline. Test 3 requires LM Studio running.)
"""

from __future__ import annotations

from llm_provider import LLMConfig, OpenAICompatProvider
from level_relevance import relevant_levels, relevant_reasoning
from output_quality import validate_and_normalize_v2
from session_tracker import LevelState

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-4b"


def test_relevant_levels_is_pure_and_atr_scaled() -> None:
    levels = (
        LevelState(name="Near", price=100.0, is_developing=False, state="testing", distance_pts=5.0),
        LevelState(name="FarButWithinATR", price=140.0, is_developing=False, state="broken_holding", distance_pts=45.0),
        LevelState(name="TooFar", price=200.0, is_developing=False, state="broken_holding", distance_pts=105.0),
        LevelState(name="StaleAndFar", price=50.0, is_developing=False, state="lost", distance_pts=-90.0),
        LevelState(name="Untouched", price=999.0, is_developing=False, state="far", distance_pts=899.0),
    )
    daily_atr = 50.0  # cutoff at 1.0x = 50pts

    kept = relevant_levels(levels, daily_atr)
    kept_names = {l.name for l in kept}
    assert kept_names == {"Near", "FarButWithinATR"}, f"unexpected relevance filter result: {kept_names}"

    # Purity: same input, same output.
    kept2 = relevant_levels(levels, daily_atr)
    assert kept == kept2

    # No daily_atr -> no-op except the existing state!='far' rule (never guesses a cutoff).
    kept_no_atr = relevant_levels(levels, None)
    assert {l.name for l in kept_no_atr} == {"Near", "FarButWithinATR", "TooFar", "StaleAndFar"}

    excluded = {l.name for l in levels} - kept_names
    reasoning = (
        "Near is testing (+5.00pts)",
        "TooFar is broken_holding (+105.00pts)",
        "StaleAndFar is lost (-90.00pts)",
        "momentum_reversal seen recently",
    )
    filtered_reasoning = relevant_reasoning(reasoning, excluded)
    assert filtered_reasoning == ("Near is testing (+5.00pts)", "momentum_reversal seen recently"), filtered_reasoning
    print("PASS: relevant_levels/relevant_reasoning are pure and correctly ATR-scaled")


def test_output_quality_validator() -> None:
    long_summary = " ".join(["word"] * 20) + " this discusses real evidence for a directional lean"

    # "null"-string normalization
    parsed_null = {
        "setup": "null", "invalidation_concept": "NULL",
        "analysis_summary": long_summary,
        "directional_bull_case": ["a"], "directional_bear_case": ["b"],
        "directional_decisive_factors": ["c"],
        "tradeability_supporting_factors": ["d"], "tradeability_opposing_factors": ["e"],
        "tradeability_reason": "a perfectly good reason with enough words",
        "tradeability": "TRADE", "directional_confidence": 0.8,
    }
    normalized, flags = validate_and_normalize_v2(parsed_null)
    assert normalized["setup"] is None and normalized["invalidation_concept"] is None
    assert "NULL_STRING_NORMALIZED:setup" in flags
    assert "NULL_STRING_NORMALIZED:invalidation_concept" in flags
    assert not any(f.startswith("EMPTY_FACTOR_ARRAY") for f in flags)

    # Empty factor array despite substantive summary
    parsed_empty = dict(parsed_null)
    parsed_empty["setup"] = None
    parsed_empty["directional_bull_case"] = []
    normalized2, flags2 = validate_and_normalize_v2(parsed_empty)
    assert "EMPTY_FACTOR_ARRAY_WITH_SUBSTANTIVE_SUMMARY:directional_bull_case" in flags2

    # Trivial tradeability_reason
    parsed_trivial = dict(parsed_null)
    parsed_trivial["tradeability_reason"] = "ok"
    _, flags3 = validate_and_normalize_v2(parsed_trivial)
    assert "TRADEABILITY_REASON_MISSING_OR_TRIVIAL" in flags3

    # Directional/tradeability inconsistency: TRADE + low confidence + no factors
    parsed_inconsistent = dict(parsed_null)
    parsed_inconsistent["directional_confidence"] = 0.5
    parsed_inconsistent["directional_bull_case"] = []
    parsed_inconsistent["directional_bear_case"] = []
    _, flags4 = validate_and_normalize_v2(parsed_inconsistent)
    assert "DIRECTIONAL_TRADEABILITY_INCONSISTENCY" in flags4

    # Clean input -> no flags at all
    parsed_clean = dict(parsed_null)
    parsed_clean["setup"] = "example setup"
    parsed_clean["invalidation_concept"] = "example invalidation"
    _, flags5 = validate_and_normalize_v2(parsed_clean)
    assert flags5 == (), f"expected no flags on clean input, got {flags5}"

    print("PASS: output_quality.validate_and_normalize_v2 catches each targeted issue and stays quiet on clean input")


def test_live_v2_schema_round_trip() -> None:
    from prompts import phase2_decision_v2
    provider = OpenAICompatProvider(LLMConfig(model=MODEL, base_url=BASE_URL))
    # Minimal probe schema mirroring the v2 shape's key/value types, not the full prompt —
    # confirms json_schema mode handles this schema's structure (nested object array, enums,
    # multiple required arrays) before spending a full real prompt on it.
    parsed, raw = provider.complete_structured(
        system="You are a test assistant. Follow the schema exactly.",
        user='Return directional_view="LONG", tradeability="NO_TRADE", both confidences 0.5, '
             'empty arrays, tradeability_reason="test reason with enough words here", '
             'setup=null, key_levels=[], invalidation_concept=null, analysis_summary="test".',
        schema=phase2_decision_v2.JSON_SCHEMA,
        schema_name="phase2_decision_v2_probe",
    )
    assert parsed["directional_view"] in ("LONG", "SHORT")
    assert parsed["tradeability"] in ("TRADE", "NO_TRADE")
    for key in phase2_decision_v2.JSON_SCHEMA["required"]:
        assert key in parsed, f"schema-required key missing from parsed response: {key}"
    print(f"PASS: v2 JSON_SCHEMA round-trips through complete_structured (raw={raw[:80]!r}...)")


if __name__ == "__main__":
    test_relevant_levels_is_pure_and_atr_scaled()
    test_output_quality_validator()
    try:
        test_live_v2_schema_round_trip()
    except ConnectionError as exc:
        print(f"SKIPPED (LM Studio unreachable): {exc}")
    print("\nAll Phase 2 V2 structural/quality checks passed.")
