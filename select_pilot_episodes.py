#!/usr/bin/env python3
"""
Selects a pilot subset (~10-15) from the 85 frozen-v1 scanner episodes in
data/episode_review.json, for the first Phase 2 analysis pass.

Selection criteria is exclusively information available at scan time —
session(s) and trigger category(ies), walked in chronological order. It
NEVER reads forward_excursion_research_only. Deterministic and
reproducible: same input file, same output every time.

Algorithm: walk episodes oldest -> newest. For each, compute the set of
(session, category) pairs it represents. If it contributes at least one
pair not yet covered by an already-selected episode, select it and mark
those pairs covered. Stop once coverage plateaus (an episode contributes
nothing new) for MAX_PLATEAU consecutive candidates, or MAX_PILOT episodes
are selected, whichever comes first.

Usage:
    python3 select_pilot_episodes.py --in data/episode_review.json \\
        --out data/pilot_episodes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_PILOT = 14
# No early-plateau cutoff: with only 85 episodes total and 16 real (session,
# category) pairs among them (found by inspection), a mid-size run of
# non-contributing episodes in the middle of the chronological order doesn't
# mean coverage is exhausted — later episodes still add new pairs. MAX_PILOT
# is the only real stopping bound; "plateau" is reaching the end of all 85
# episodes with nothing left to add, i.e. len(episodes).
MAX_PLATEAU = 10_000


def _pairs(episode: dict) -> set[tuple[str, str]]:
    pit = episode["point_in_time"]
    sessions = pit["sessions"] or ["none"]
    categories = pit["categories"] or ["none"]
    return {(s, c) for s in sessions for c in categories}


def select(episodes: list[dict]) -> list[dict]:
    covered: set[tuple[str, str]] = set()
    selected: list[dict] = []
    plateau = 0

    for ep in episodes:
        if len(selected) >= MAX_PILOT:
            break
        pairs = _pairs(ep)
        new_pairs = pairs - covered
        if new_pairs:
            selected.append({**ep, "_new_pairs": sorted(new_pairs)})
            covered |= pairs
            plateau = 0
        else:
            plateau += 1
            if plateau >= MAX_PLATEAU:
                break

    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/episode_review.json")
    ap.add_argument("--out", dest="out_path", default="data/pilot_episodes.json")
    args = ap.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from {args.in_path}")

    selected = select(episodes)

    print(f"\nSelected {len(selected)} pilot episodes (criteria: session/category coverage only):\n")
    print(f"{'#':>2}  {'timestamp_et':<26} {'sessions':<20} {'categories':<45} new (session, category) pairs")
    print("-" * 130)
    for i, ep in enumerate(selected, 1):
        pit = ep["point_in_time"]
        print(
            f"{i:>2}  {pit['timestamp_et']:<26} {','.join(pit['sessions']):<20} "
            f"{','.join(pit['categories']):<45} {ep['_new_pairs']}"
        )

    all_pairs = set()
    for ep in selected:
        all_pairs |= set(tuple(p) for p in ep["_new_pairs"])
    print(f"\nTotal distinct (session, category) pairs covered: {len(all_pairs)}")

    out = {
        "meta": {
            "source": args.in_path,
            "selection_criteria": "chronological greedy walk, session/category coverage only, "
                                   "no forward-excursion data read",
            "pilot_count": len(selected),
        },
        "episodes": selected,
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
