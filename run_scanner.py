#!/usr/bin/env python3
"""
Historical Opportunity Scanner runner — Phase 1 of the BladeTrade +
QuantHarness architecture. No LLM, no TradeIdea, no execution: this only
answers "is the scanner well-calibrated" by replaying historical bars,
scanning every 15 minutes, and persisting every scan (interesting or not)
to data/experiments.db.

Usage:
    python3 run_scanner.py --file data/NQZ5_1m.csv --symbol NQ \\
        --start 2025-09-22 --end 2025-12-19

    # override session windows:
    python3 run_scanner.py --file data/MYM_combined_1m.csv --symbol MYM \\
        --start 2026-07-27 --end 2026-09-17 --london-end 08:30
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from experiment_store import ExperimentStore
from mtf_indicators import compute_mtf_features
from opportunity_scanner import compute_scanner_features, scan
from session_config import SessionConfig
from session_tracker import ScannerConfig, SessionTracker
from weekly_levels import build_weekly_bars, compute_pivots

ET = ZoneInfo("America/New_York")


def _parse_time(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tick-size", type=float, default=0.25, dest="tick_size")
    ap.add_argument("--london-start", default="00:00", dest="london_start")
    ap.add_argument("--london-end", default="09:00", dest="london_end")
    ap.add_argument("--ny-start", default="08:00", dest="ny_start")
    ap.add_argument("--ny-end", default="17:00", dest="ny_end")
    ap.add_argument("--db", default="data/experiments.db")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    session_cfg = SessionConfig(
        london_start_et=_parse_time(args.london_start),
        london_end_et=_parse_time(args.london_end),
        ny_start_et=_parse_time(args.ny_start),
        ny_end_et=_parse_time(args.ny_end),
    )
    scanner_cfg = ScannerConfig()

    print(f"Loading {path} ...")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} bars  ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")

    print("Computing 15m MACD / 60m RSI / 4hr MACD+RSI ...")
    features = compute_mtf_features(df)
    print("Computing ATR / volatility percentile ...")
    scanner_features = compute_scanner_features(df)

    print("Computing weekly floor pivots ...")
    weeks = build_weekly_bars(df)
    weekly_pivots = compute_pivots(weeks)
    unreliable = sum(1 for p in weekly_pivots if not p.reliable)
    if unreliable:
        print(f"  Warning: {unreliable} of {len(weekly_pivots)} week(s) have insufficient prior-week "
              f"bar density — their weekly levels may be unreliable.")

    start = pd.Timestamp(args.start, tz=ET).normalize()
    end = (pd.Timestamp(args.end, tz=ET) + pd.Timedelta(days=1)).normalize()
    et_ts = df["timestamp"].dt.tz_convert(ET)
    mask = (et_ts >= start) & (et_ts < end)
    run_bars = df.loc[mask].sort_values("timestamp")
    if run_bars.empty:
        print("No bars found in that range.")
        return
    print(f"Scanning {len(run_bars):,} bars  ({run_bars['timestamp'].iloc[0]} -> {run_bars['timestamp'].iloc[-1]})")

    tracker = SessionTracker(config=session_cfg, scanner_config=scanner_cfg,
                              weekly_pivots=weekly_pivots, tick_size=args.tick_size)

    store = ExperimentStore(args.db)
    scan_run_id = store.create_scan_run(
        symbol=args.symbol.upper(), date_range_start=args.start, date_range_end=args.end,
        session_config={k: str(v) for k, v in session_cfg.__dict__.items()},
        scanner_config=scanner_cfg.__dict__,
    )

    seen_episodes: set[str] = set()
    total_scans = 0
    opportunity_count = 0
    new_episode_count = 0
    category_counts: Counter = Counter()
    by_day: dict = defaultdict(lambda: [0, 0])  # date -> [scans, opportunities]

    # Bars before `start` are used for warmup (PDH/Asia/London/weekly context) but
    # not scanned or persisted — only the requested range counts as "the run."
    warmup_mask = et_ts < start
    all_bars_in_order = pd.concat([df.loc[warmup_mask], run_bars]).sort_values("timestamp")

    for row in all_bars_in_order.itertuples():
        tracker.update(row)
        if row.timestamp < start:
            continue
        if row.timestamp.minute % 15 != 0:
            continue

        opp = scan(
            tracker=tracker, features=features, scanner_features=scanner_features,
            price=row.close, timestamp=row.timestamp, symbol=args.symbol.upper(),
            seen_episodes=seen_episodes, cfg=scanner_cfg,
        )
        store.record_opportunity(scan_run_id=scan_run_id, opp=opp)

        total_scans += 1
        day = row.timestamp.astimezone(ET).date()
        by_day[day][0] += 1
        if opp.is_opportunity:
            opportunity_count += 1
            by_day[day][1] += 1
            category_counts.update(opp.categories)
            if opp.is_new_episode:
                new_episode_count += 1

    store.close()

    print(f"\n{'='*70}")
    print(f"  {args.symbol.upper()} Opportunity Scanner — {args.start} to {args.end}")
    print(f"{'='*70}")
    print(f"  Total scans          : {total_scans}")
    print(f"  Opportunities        : {opportunity_count} ({opportunity_count/total_scans*100:.1f}% of scans)")
    print(f"  New episodes         : {new_episode_count}")
    print(f"\n  By category:")
    for cat, n in category_counts.most_common():
        print(f"    {cat:<28} {n:>5}")
    print(f"\n  By day (scans / opportunities):")
    for day in sorted(by_day):
        s, o = by_day[day]
        print(f"    {day}  {s:>3} scans  {o:>3} opportunities")
    print(f"\nRecorded to {args.db}  (scan_run_id={scan_run_id})")


if __name__ == "__main__":
    main()
