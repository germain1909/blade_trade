#!/usr/bin/env python3
"""
Milestone 1 — Experiment A vs. B: does a local LLM TAKE/SKIP filter improve
on the existing deterministic MACD-turn gate?

Runs the identical candidate stream (every cross/fail event backtest_manual_week.py's
LevelWatcher detects, for the same file/week/levels) through two strategies:

  A — RuleBasedStrategy   (today's require_macd_turn boolean check)
  B — LLMStrategy         (a local model via LM Studio, judging the same
                           MarketSnapshot the rule-based gate inspects)

Every candidate — TAKE or SKIP, either experiment — is recorded in
data/experiments.db (see experiment_store.py) with its full snapshot,
decision, and outcome (real fill for TAKEs, forward-simulated for SKIPs),
so the two experiments can be compared after the fact.

Usage:
    python3 run_experiment.py --file data/MYM_combined_1m.csv --symbol MYM \\
        --week-start 2026-09-14 --week-end 2026-09-18 \\
        --level "L1:52605" --level "L2:52320" --level "L3:52016" --level "L4:51505"

    # LLM leg only, pointed at a non-default LM Studio port:
    python3 run_experiment.py --file ... --symbol MYM --week-start ... --week-end ... \\
        --level ... --only llm --llm-base-url http://localhost:1234/v1 --llm-model qwen3-vl-4b-instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_manual_week import Config, parse_level, run_week
from backtest_weekly_swing_fade import INSTRUMENT_SPECS
from decision_strategy import RuleBasedStrategy, LLMStrategy
from experiment_store import ExperimentStore
from llm_provider import LLMConfig, OpenAICompatProvider
from mtf_indicators import compute_mtf_features

ET = __import__("zoneinfo").ZoneInfo("America/New_York")


def _summarize(label: str, decisions: list[tuple], trades: list) -> dict:
    takes = [d for d in decisions if d[1].action == "TAKE"]
    skips = [d for d in decisions if d[1].action == "SKIP"]
    real_pnl = sum(t.pnl_usd for t in trades)
    wins = [t for t in trades if t.pnl_usd > 0]
    hypothetical_pnl = sum(o.pnl_usd for _, dec, o in skips if o is not None and o.pnl_usd is not None)

    print(f"\n{'='*90}")
    print(f"  Experiment {label}")
    print(f"{'='*90}")
    print(f"  Candidates evaluated : {len(decisions)}  ({len(takes)} TAKE / {len(skips)} SKIP)")
    print(f"  Real trades taken    : {len(trades)}  ({len(wins)}W / {len(trades)-len(wins)}L)")
    if trades:
        wr = len(wins) / len(trades) * 100
        gw = sum(t.pnl_usd for t in wins)
        gl = abs(sum(t.pnl_usd for t in trades if t.pnl_usd <= 0))
        pf = gw / gl if gl else float("inf")
        print(f"  Win rate             : {wr:.1f}%")
        print(f"  Real PnL             : ${real_pnl:+,.2f}   PF: {pf:.2f}")
    else:
        print(f"  Real PnL             : $+0.00 (no candidates were TAKEn)")
    print(f"  Hypothetical PnL of SKIPped candidates (if taken anyway): ${hypothetical_pnl:+,.2f}")

    return {
        "label": label, "candidates": len(decisions), "takes": len(takes), "skips": len(skips),
        "trades": len(trades), "real_pnl": real_pnl, "hypothetical_skip_pnl": hypothetical_pnl,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--week-start", required=True)
    ap.add_argument("--week-end", required=True)
    ap.add_argument("--level", action="append", required=True, dest="levels")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--cooldown-hours", type=float, default=4.0, dest="cooldown_hours")
    ap.add_argument("--only", choices=["rule", "llm"], default=None,
                     help="Run only one leg instead of both A and B")
    ap.add_argument("--llm-base-url", default="http://localhost:1234/v1", dest="llm_base_url")
    ap.add_argument("--llm-model", default="qwen3-vl-4b-instruct", dest="llm_model")
    ap.add_argument("--db", default="data/experiments.db")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    spec = INSTRUMENT_SPECS.get(args.symbol.upper(), INSTRUMENT_SPECS["NQ"])
    base_cfg_kwargs = dict(
        contracts=args.contracts, tick_size=spec["tick_size"],
        dollars_per_point=spec["dollars_per_point"], cooldown_hours=args.cooldown_hours,
        symbol_label=args.symbol.upper(),
    )
    levels = [parse_level(s) for s in args.levels]

    print(f"Loading {path} ...")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    week_start = pd.Timestamp(args.week_start, tz=ET).normalize()
    week_end = (pd.Timestamp(args.week_end, tz=ET) + pd.Timedelta(days=1)).normalize()

    print("Computing 15m MACD / 60m RSI / 4hr MACD+RSI ...")
    features = compute_mtf_features(df)

    et_ts = df["timestamp"].dt.tz_convert(ET)
    mask = (et_ts >= week_start) & (et_ts < week_end)
    week_bars = df.loc[mask].sort_values("timestamp")
    if week_bars.empty:
        print("No bars found in that week range.")
        return
    print(f"Week bars: {len(week_bars):,}   Levels: "
          f"{', '.join(f'{l.name}={l.price:.2f}' for l in levels)}")

    store = ExperimentStore(args.db)
    results = []

    legs = [("A_rulebased", "rule"), ("B_llm_filter", "llm")]
    if args.only:
        legs = [leg for leg in legs if leg[1] == args.only]

    for label, kind in legs:
        cfg = Config(**base_cfg_kwargs, require_macd_turn=True)

        if kind == "rule":
            strategy = RuleBasedStrategy(require_macd_turn=True)
            model_name = model_version = prompt_version = ""
            inference_params = {}
        else:
            llm_cfg = LLMConfig(base_url=args.llm_base_url, model=args.llm_model)
            try:
                provider = OpenAICompatProvider(llm_cfg)
            except ImportError as exc:
                print(f"\nSkipping {label}: {exc}")
                continue
            strategy = LLMStrategy(provider, prompt_version=llm_cfg.prompt_version)
            model_name = llm_cfg.model
            model_version = ""
            prompt_version = llm_cfg.prompt_version
            inference_params = {"temperature": llm_cfg.temperature, "max_tokens": llm_cfg.max_tokens,
                                 "base_url": llm_cfg.base_url}

        try:
            trades, _snapshots, decisions = run_week(week_bars, features, levels, cfg, strategy=strategy)
        except ConnectionError as exc:
            print(f"\nAborting {label}: {exc}")
            continue

        experiment_id = store.create_experiment(
            label=label, symbol=args.symbol.upper(),
            date_range_start=args.week_start, date_range_end=args.week_end,
            strategy_class=type(strategy).__name__,
            strategy_config=base_cfg_kwargs,
            model_name=model_name, model_version=model_version,
            prompt_version=prompt_version, inference_params=inference_params,
        )
        for snapshot, decision, outcome in decisions:
            store.record_decision(experiment_id=experiment_id, snapshot=snapshot,
                                   decision=decision, outcome=outcome)

        results.append(_summarize(label, decisions, trades))

    if len(results) == 2:
        print(f"\n{'='*90}")
        print(f"  A vs. B")
        print(f"{'='*90}")
        a, b = results
        print(f"  {a['label']:<14} real PnL ${a['real_pnl']:+,.2f}  ({a['trades']} trades)")
        print(f"  {b['label']:<14} real PnL ${b['real_pnl']:+,.2f}  ({b['trades']} trades)")
        diff = b["real_pnl"] - a["real_pnl"]
        print(f"  Difference (B - A): ${diff:+,.2f}")

    store.close()
    print(f"\nRecorded to {args.db}")


if __name__ == "__main__":
    main()
