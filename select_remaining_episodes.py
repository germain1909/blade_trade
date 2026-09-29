#!/usr/bin/env python3
"""
Selects the episodes for the V2 blinded evaluation: every one of the 85
frozen-v1 scanner episodes NOT already used in the V1 pilot (the 12 rows
in data/experiments.db's `analyses` table). No coverage/diversity
heuristic — using the full remaining population is the cleanest possible
choice, since any selective sub-sampling risks incidentally correlating
with something outcome-adjacent. Never reads forward_excursion_research_only.

Usage:
    python3 select_remaining_episodes.py --review data/episode_review.json \\
        --db data/experiments.db --out data/remaining_episodes.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="data/episode_review.json")
    ap.add_argument("--db", default="data/experiments.db")
    ap.add_argument("--out", default="data/remaining_episodes.json")
    args = ap.parse_args()

    review = json.loads(Path(args.review).read_text())
    all_episodes = review["episodes"]
    print(f"Loaded {len(all_episodes)} total frozen-v1 episodes from {args.review}")

    conn = sqlite3.connect(args.db)
    used_ids = {r[0] for r in conn.execute("SELECT DISTINCT episode_id FROM analyses")}
    conn.close()
    print(f"{len(used_ids)} episode_ids already used in the V1 pilot ({args.db}'s analyses table)")

    remaining = [ep for ep in all_episodes if ep["point_in_time"]["episode_id"] not in used_ids]
    remaining_ids = {ep["point_in_time"]["episode_id"] for ep in remaining}
    overlap = remaining_ids & used_ids
    assert not overlap, f"selection bug: {len(overlap)} episodes overlap with the already-used pilot set"

    print(f"Selected {len(remaining)} remaining episodes for the V2 blinded evaluation "
          f"(zero overlap with the pilot's 12 confirmed)")

    out = {
        "meta": {
            "source": args.review,
            "selection_criteria": "all frozen-v1 episodes not already used in the V1 pilot — no "
                                   "coverage/diversity heuristic, no forward-excursion data read",
            "total_frozen_episodes": len(all_episodes),
            "already_used_in_pilot": len(used_ids),
            "remaining_count": len(remaining),
        },
        "episodes": remaining,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
