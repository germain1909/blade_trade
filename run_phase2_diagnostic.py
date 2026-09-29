#!/usr/bin/env python3
"""
DIAGNOSTIC-ONLY pass over the 12 pilot episodes already persisted in
data/experiments.db's `analyses` table. Reuses each episode's exact
already-computed snapshot/trend/vision (no re-scanning, no new vision
calls) and asks a different question: forced LONG-vs-SHORT, with NO_TRADE
unavailable. See prompts/forced_direction_v1.py for why.

Persists to a separate `diagnostic_forced_direction` table. Never touches
or overwrites the real `analyses` rows.

Usage:
    python3 run_phase2_diagnostic.py --db data/experiments.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3

from experiment_store import ExperimentStore
from llm_provider import LLMConfig, OpenAICompatProvider
from phase2_diagnostic import run_forced_direction_diagnostic
from prompts import forced_direction_v1

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-4b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/experiments.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT episode_id, timestamp, symbol, snapshot_json, trend_analysis_json, vision_analysis_json "
        "FROM analyses ORDER BY timestamp"
    ).fetchall()
    conn.close()
    print(f"Loaded {len(rows)} pilot episodes from {args.db}'s analyses table\n")

    provider = OpenAICompatProvider(LLMConfig(base_url=BASE_URL, model=MODEL))
    store = ExperimentStore(db_path=args.db)

    print(f"{'timestamp':<26} {'forced_direction':<18} confidence")
    print("-" * 60)

    for row in rows:
        snapshot = json.loads(row["snapshot_json"])
        trend = json.loads(row["trend_analysis_json"]) if row["trend_analysis_json"] else None
        vision = json.loads(row["vision_analysis_json"])

        result = run_forced_direction_diagnostic(provider, snapshot, trend, vision)
        prompt_text = forced_direction_v1.build_prompt(snapshot, trend, vision)

        store.record_diagnostic_forced_direction(
            episode_id=row["episode_id"], timestamp=row["timestamp"], symbol=row["symbol"],
            snapshot=snapshot, trend=trend, vision=vision, final_prompt=prompt_text,
            result=result, model_name=provider.config.model, provider="OpenAICompatProvider",
            prompt_version=forced_direction_v1.PROMPT_VERSION,
            inference_settings={"temperature": provider.config.temperature, "base_url": provider.config.base_url},
            retry_count=provider.last_retry_count, parse_errors=provider.last_parse_error,
        )

        print(f"{row['timestamp']:<26} {result.forced_direction:<18} {result.directional_confidence:.2f}")
        print(f"    bull_case: {list(result.bull_case)}")
        print(f"    bear_case: {list(result.bear_case)}")
        print(f"    decisive_factors: {list(result.decisive_factors)}")
        print()

    store.close()
    print("Diagnostic pass complete. Persisted to diagnostic_forced_direction — original analyses rows untouched.")


if __name__ == "__main__":
    main()
