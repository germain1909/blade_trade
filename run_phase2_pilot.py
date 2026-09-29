#!/usr/bin/env python3
"""
Phase 2 pilot runner — analyzes ONLY the episodes selected by
select_pilot_episodes.py (10-15 of the 85 frozen-v1 scanner episodes), not
all 85. Per episode: build a point-in-time OpportunitySnapshot, compute a
deterministic TrendAnalysis, render a point-in-time chart, run the local
vision model constrained to prompts/vision_v1.py's schema, run the final
LONG/SHORT/NO_TRADE synthesis constrained to prompts/phase2_decision_v1.py's
schema, print every field, and persist to experiment_store.py's `analyses`
table.

This intentionally stops here. No TradeIdea, no ActiveThesis, no entry
triggers, no execution, no scanner/prompt tuning against outcomes — those
are out of scope until the pilot has been reviewed.

Usage:
    python3 run_phase2_pilot.py --file data/NQZ5_1m.csv --symbol NQ \\
        --pilot data/pilot_episodes.json --charts-dir data/phase2_charts
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
from phase2_analysis import run_phase2_analysis
from prompts import phase2_decision_v1
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
    print(f"Scanner reasoning: {list(snapshot.reasoning)}")
    print("-" * 100)
    print("Levels:")
    for l in snapshot.levels:
        if l.price is not None and l.state != "far":
            print(f"    {l.name}: price={l.price:.2f} state={l.state} dist={l.distance_pts:+.2f}")
    if snapshot.confluence_zones:
        print("Confluence zones:")
        for z in snapshot.confluence_zones:
            print(f"    center={z.center:.2f} levels={list(z.level_names)}")
    print(f"VWAP: {snapshot.vwap}  distance={snapshot.vwap_distance_pts}  "
          f"({snapshot.vwap_distance_daily_atr}x daily ATR)  extension_flag={snapshot.vwap_extension_flag}")
    print(f"Momentum: 15m_hist={snapshot.macd_15m_hist} bottom_recent={snapshot.macd_15m_bottom_recent} "
          f"top_recent={snapshot.macd_15m_top_recent} 60m_rsi={snapshot.rsi_60m} "
          f"4h_hist={snapshot.macd_4h_hist} 4h_rsi={snapshot.rsi_4h}")
    print(f"Volatility: atr_60bar={snapshot.atr_60bar} atr_percentile={snapshot.atr_percentile} "
          f"daily_atr={snapshot.daily_atr}")
    print("-" * 100)
    if trend is not None:
        print(f"Trend analysis: direction={trend.channel_direction} compression={trend.compression} "
              f"channel_width={trend.channel_width_pts:.2f}pts ({trend.channel_width_atr_mult:.2f}x daily ATR) "
              f"price_position={trend.price_position_in_channel:.2f} bars_used={trend.bars_used}")
    else:
        print("Trend analysis: insufficient bars for a fit")
    print(f"Chart: {chart_path}")
    print("-" * 100)
    print(f"Vision analysis: trend_structure={vision.trend_structure!r} pattern={vision.pattern!r} "
          f"compression={vision.compression} latency={vision.latency_s:.2f}s")
    print(f"    breakout={vision.breakout_structure!r}  rejection={vision.rejection_structure!r}")
    print(f"    supporting={list(vision.visual_supporting_factors)}")
    print(f"    risk={list(vision.visual_risk_factors)}")
    print("-" * 100)
    print(f"DECISION: {result.decision}   confidence={result.confidence:.2f}   setup={result.setup!r}")
    print(f"Thesis: {result.thesis}")
    if result.decision == "NO_TRADE":
        print(f"Reason: {result.reason}")
    print(f"Supporting: {list(result.supporting_factors)}")
    print(f"Opposing: {list(result.opposing_factors)}")
    print(f"Key levels: {list(result.key_levels)}")
    print(f"Invalidation concept: {result.invalidation_concept}")
    print(f"Summary: {result.analysis_summary}")
    print(f"Model: {provider.config.model}  provider=OpenAICompatProvider({provider.config.base_url})  "
          f"prompt_version={phase2_decision_v1.PROMPT_VERSION}  latency={result.latency_s:.2f}s  "
          f"retries={retry_count}  parse_error={parse_error}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/NQZ5_1m.csv")
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--pilot", default="data/pilot_episodes.json")
    ap.add_argument("--charts-dir", default="data/phase2_charts")
    ap.add_argument("--db", default="data/experiments.db")
    args = ap.parse_args()

    pilot = json.loads(Path(args.pilot).read_text())
    pilot_episode_ids = {ep["point_in_time"]["episode_id"] for ep in pilot["episodes"]}
    pilot_timestamps = {pd.Timestamp(ep["point_in_time"]["timestamp"]) for ep in pilot["episodes"]}
    print(f"Loaded {len(pilot_timestamps)} pilot episodes from {args.pilot}")

    path = Path(args.file)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Computing indicators/scanner features (same frozen-v1 computation as the review)...")
    features = compute_mtf_features(df)
    scanner_features = compute_scanner_features(df)
    weeks = build_weekly_bars(df)
    weekly_pivots = compute_pivots(weeks)

    session_cfg = SessionConfig()
    scanner_cfg = ScannerConfig()
    assert scanner_cfg.version == SCANNER_CONFIG_VERSION == "v1", "scanner config version drifted from v1"

    tracker = SessionTracker(config=session_cfg, scanner_config=scanner_cfg,
                              weekly_pivots=weekly_pivots, tick_size=0.25)

    try:
        provider = OpenAICompatProvider(LLMConfig(base_url=BASE_URL, model=MODEL))
    except ImportError as exc:
        print(f"ABORT: {exc}")
        return

    store = ExperimentStore(db_path=args.db)
    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    seen_episodes: set[str] = set()
    analyzed = 0
    results_summary = []

    print(f"\nReplaying {path} and analyzing pilot episodes as they're reached...\n")
    for row in df.itertuples():
        tracker.update(row)
        if row.timestamp.minute % 15 != 0:
            continue
        if row.timestamp not in pilot_timestamps:
            continue

        opp = scan(
            tracker=tracker, features=features, scanner_features=scanner_features,
            price=row.close, timestamp=row.timestamp, symbol=args.symbol.upper(),
            seen_episodes=seen_episodes, cfg=scanner_cfg,
        )
        if not opp.is_new_episode or opp.episode_id not in pilot_episode_ids:
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
            result = run_phase2_analysis(provider, snapshot, trend, vision)

            retry_count = provider.last_retry_count
            parse_error = provider.last_parse_error

            _print_episode_result(snapshot, trend, chart_path, vision, result, provider, retry_count, parse_error)

            store.record_analysis(
                snapshot=snapshot, trend=trend, chart_path=chart_path, vision=vision,
                final_prompt=phase2_decision_v1.build_prompt(snapshot, trend, vision),
                result=result, model_name=provider.config.model, provider="OpenAICompatProvider",
                prompt_version=phase2_decision_v1.PROMPT_VERSION,
                inference_settings={
                    "temperature": provider.config.temperature, "max_tokens": provider.config.max_tokens,
                    "seed": provider.config.seed, "base_url": provider.config.base_url,
                },
                retry_count=retry_count, parse_errors=parse_error,
            )
            analyzed += 1
            results_summary.append((snapshot.timestamp, sorted(snapshot.sessions), result.decision, result.confidence))
        except ConnectionError as exc:
            print(f"ABORT: could not reach the local LLM — {exc}")
            store.close()
            return

    store.close()

    print("=" * 100)
    print(f"Pilot complete: {analyzed}/{len(pilot_timestamps)} episodes analyzed and persisted to {args.db}")
    print(f"{'timestamp':<26} {'sessions':<20} {'decision':<10} confidence")
    for ts, sessions, decision, confidence in results_summary:
        print(f"{ts.isoformat():<26} {','.join(sessions):<20} {decision:<10} {confidence:.2f}")
    print("\nSTOP: per the approved Phase 2 plan, this pilot is the endpoint until reviewed. "
          "No further episodes, no TradeIdea, no execution, no scanner/prompt tuning from here.")


if __name__ == "__main__":
    main()
