#!/usr/bin/env python3
"""
Model-facing level relevance filter — Phase 2 V2. Filters what appears in
PROMPT TEXT only; never modifies OpportunitySnapshot.levels/.reasoning
(the persisted, raw record is untouched — callers keep the full snapshot
and separately record what was actually shown to the model).

Root cause this addresses: session_tracker.py's `_LevelWatch` has no
distance-based exit from "broken_holding" (unlike "lost"/"reclaimed",
which do reset via approach_dist) — if price runs away without ever
crossing back through the level, it stays "broken_holding" forever
regardless of distance. Not fixed in session_tracker.py itself (frozen
Phase 1, out of scope) — fixed here, at the prompt-construction layer.

Threshold (1.0x daily_atr) is a design default chosen independently of the
12-episode pilot's outcomes — see RESEARCH_NOTES.md and the approved V2
plan for the reasoning. Not tuned to erase any specific previously-noticed
example.
"""

from __future__ import annotations

from typing import Optional

from session_tracker import LevelState

DEFAULT_ATR_MULTIPLE = 1.0


def relevant_levels(
    levels: tuple[LevelState, ...],
    daily_atr: Optional[float],
    atr_multiple: float = DEFAULT_ATR_MULTIPLE,
) -> tuple[LevelState, ...]:
    """A level is model-facing-relevant if it's not 'far' and its distance
    is within atr_multiple x daily_atr. If daily_atr is unavailable
    (insufficient warmup), this is a no-op (shows everything) rather than
    guessing a cutoff."""
    if daily_atr is None:
        return tuple(l for l in levels if l.state != "far")
    cutoff = atr_multiple * daily_atr
    return tuple(
        l for l in levels
        if l.state != "far" and l.distance_pts is not None and abs(l.distance_pts) <= cutoff
    )


def relevant_reasoning(reasoning: tuple[str, ...], excluded_names: set[str]) -> tuple[str, ...]:
    """Drops a reasoning line if it names a level excluded by relevant_levels,
    so filtered-out levels can't leak back in through the free-text reasoning
    list. Scanner reasoning lines always start with the level/category name
    (see opportunity_scanner.py's reasoning.append(f"{l.name} is ...") /
    f"{l.name} approaching ..." convention)."""
    if not excluded_names:
        return reasoning
    return tuple(line for line in reasoning if not any(line.startswith(name) for name in excluded_names))
