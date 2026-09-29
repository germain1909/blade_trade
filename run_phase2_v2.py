#!/usr/bin/env python3
"""
Phase 2 V2 blinded run — directional_view + tradeability in one frozen
record, over the episodes selected by select_remaining_episodes.py.
Reuses opportunity_snapshot/trend_analysis/chart_renderer/vision_analysis
completely unchanged from V1. Applies level_relevance's filter when
building the v2 prompt. Never reads/prints/references
forward_excursion_research_only or any bar beyond an episode's own
point-in-time cutoff.

Supports --limit for a smoke test on the first N (chronologically) episodes
before committing to the full run.

Usage:
    python3 run_phase2_v2.py --limit 3          # smoke test
    python3 run_phase2_v2.py                    # full run over all selected episodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from chart_renderer import render
from experiment_store import ExperimentStore
from llm_provider import LLMConfig, OpenAICompatProvider
from mtf_indicators import compute_mtf_features
from opportunity_scanner import compute_scanner_features, scan
from opportunity_snapshot import build_opportunity_snapshot
from phase2_analysis_v2 import run_phase2_analysis_v2
from prompts import phase2_decision_v2
from session_config import SessionConfig
from session_tracker import SCANNER_CONFIG_VERSION, ScannerConfig, SessionTracker
from trend_analysis import compute_trend_analysis
from vision_analysis import run_vision_analysis
from weekly_levels import build_weekly_bars, compute_pivots

ET = ZoneInfo("America/New_York")

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-4b"


def _print_episode_result(snapshot, trend, chart_path, vision, result, provider, retry_count, parse_error) -> None:
    print("=" * 100)
    print(f"Episode {snapshot.episode_id}  |  {snapshot.timestamp.isoformat()}  |  {snapshot.symbol}")
    print(f"Session(s): {sorted(snapshot.sessions)}   Price: {snapshot.price}")
    print(f"Scanner categories: {list(snapshot.categories)}")
    print("-" * 100)
    print(f"DIRECTIONAL VIEW: {result.directional_view}   confidence={result.directional_confidence:.2f}")
    print(f"  bull_case: {list(result.directional_bull_case)}")
    print(f"  bear_case: {list(result.directional_bear_case)}")
    print(f"  decisive_factors: {list(result.directional_decisive_factors)}")
    print(f"TRADEABILITY: {result.tradeability}   confidence={result.tradeability_confidence:.2f}")
    print(f"  reason: {result.tradeability_reason}")
    print(f"  supporting: {list(result.tradeability_supporting_factors)}")
    print(f"  opposing: {list(result.tradeability_opposing_factors)}")
    print(f"setup={result.setup!r}  invalidation_concept={result.invalidation_concept!r}")
    print(f"Summary: {result.analysis_summary}")
    if result.quality_flags:
        print(f"QUALITY FLAGS: {list(result.quality_flags)}")
    print(f"Chart: {chart_path}   latency={result.latency_s:.2f}s   retries={retry_count}   "
          f"parse_error={parse_error}   prompt_version={phase2_decision_v2.PROMPT_VERSION}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/NQZ5_1m.csv")
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--episodes", default="data/remaining_episodes.json")
    ap.add_argument("--charts-dir", default="data/phase2_v2_charts")
    ap.add_argument("--db", default="data/experiments.db")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N chronological episodes")
    args = ap.parse_args()

    episodes_data = json.loads(Path(args.episodes).read_text())
    episode_list = episodes_data["episodes"]
    if args.limit is not None:
        episode_list = episode_list[: args.limit]
    target_episode_ids = {ep["point_in_time"]["episode_id"] for ep in episode_list}
    target_timestamps = {pd.Timestamp(ep["point_in_time"]["timestamp"]) for ep in episode_list}
    print(f"Loaded {len(target_timestamps)} episodes to analyze from {args.episodes}"
          + (f" (--limit {args.limit})" if args.limit else ""))

    path = Path(args.file)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Computing indicators/scanner features (same frozen-v1 computation as the pilot)...")
    features = compute_mtf_features(df)
    scanner_features = compute_scanner_features(df)
    weeks = build_weekly_bars(df)
    weekly_pivots = compute_pivots(weeks)

    session_cfg = SessionConfig()
    scanner_cfg = ScannerConfig()
    assert scanner_cfg.version == SCANNER_CONFIG_VERSION == "v1", "scanner config version drifted from v1"

    tracker = SessionTracker(config=session_cfg, scanner_config=scanner_cfg,
                              weekly_pivots=weekly_pivots, tick_size=0.25)

    provider = OpenAICompatProvider(LLMConfig(base_url=BASE_URL, model=MODEL))
    store = ExperimentStore(db_path=args.db)
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    seen_episodes: set[str] = set()
    analyzed = 0
    results_summary = []

    print(f"\nReplaying {path} and analyzing target episodes as they're reached...\n")
    for row in df.itertuples():
        tracker.update(row)
        if row.timestamp.minute % 15 != 0:
            continue
        if row.timestamp not in target_timestamps:
            continue

        opp = scan(
            tracker=tracker, features=features, scanner_features=scanner_features,
            price=row.close, timestamp=row.timestamp, symbol=args.symbol.upper(),
            seen_episodes=seen_episodes, cfg=scanner_cfg,
        )
        if not opp.is_new_episode or opp.episode_id not in target_episode_ids:
            continue

        try:
            snapshot = build_opportunity_snapshot(tracker, features, scanner_features, opp, df, lookback_bars=60)
            trend = compute_trend_analysis(snapshot.recent_bars, daily_atr=snapshot.daily_atr)

            chart_path = str(charts_dir / f"{snapshot.timestamp.strftime('%Y%m%dT%H%M')}_{snapshot.episode_id}.png")
            _, image_b64 = render(
                snapshot.recent_bars, snapshot_timestamp=snapshot.timestamp,
                levels=snapshot.levels, trend=trend, vwap=snapshot.vwap, out_path=chart_path,
            )

            vision = run_vision_analysis(provider, snapshot, trend, image_b64)
            result, prompt_text, levels_shown = run_phase2_analysis_v2(provider, snapshot, trend, vision)

            retry_count = provider.last_retry_count
            parse_error = provider.last_parse_error

            _print_episode_result(snapshot, trend, chart_path, vision, result, provider, retry_count, parse_error)

            store.record_analysis_v2(
                snapshot=snapshot, trend=trend, chart_path=chart_path, vision=vision,
                levels_shown_to_model=levels_shown, final_prompt=prompt_text,
                result=result, model_name=provider.config.model, provider="OpenAICompatProvider",
                prompt_version=phase2_decision_v2.PROMPT_VERSION,
                inference_settings={
                    "temperature": provider.config.temperature, "max_tokens": provider.config.max_tokens,
                    "seed": provider.config.seed, "base_url": provider.config.base_url,
                },
                retry_count=retry_count, parse_errors=parse_error,
            )
            analyzed += 1
            results_summary.append(
                (snapshot.timestamp, sorted(snapshot.sessions), result.directional_view,
                 result.directional_confidence, result.tradeability, result.tradeability_confidence)
            )
        except ConnectionError as exc:
            print(f"ABORT: could not reach the local LLM — {exc}")
            store.close()
            return

    store.close()

    print("=" * 100)
    print(f"V2 run complete: {analyzed}/{len(target_timestamps)} episodes analyzed and persisted to {args.db}")
    print(f"{'timestamp':<26} {'sessions':<20} {'direction':<10} {'dconf':<7} {'tradeability':<12} tconf")
    for ts, sessions, direction, dconf, tradeability, tconf in results_summary:
        print(f"{ts.isoformat():<26} {','.join(sessions):<20} {direction:<10} {dconf:<7.2f} "
              f"{tradeability:<12} {tconf:.2f}")
    print("\nSTOP: per the approved V2 plan, freeze here. No outcome computation, no TradeIdea, "
          "no execution, no scanner/prompt tuning from here.")


if __name__ == "__main__":
    main()
