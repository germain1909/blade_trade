#!/usr/bin/env python3
"""
9AM Breakout Strategy backtest.

Usage:
    python3 backtest_9am.py                          # MES/ES, opposite stop, 2R
    python3 backtest_9am.py --gold                   # Gold
    python3 backtest_9am.py --stop mid --rr 1.5      # midpoint stop, 1.5R target
"""

import argparse
from datetime import time as dtime
from pathlib import Path

from trading_agent.backtester import load_excel_bars
from trading_agent.nine_am_strategy import (
    NineAmConfig, run_nine_am_strategy, print_nine_am_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--stop", choices=["opposite", "mid"], default="opposite")
    ap.add_argument("--rr", type=float, default=2.0, help="Target R:R multiple")
    ap.add_argument("--contracts", type=int, default=1)
    args = ap.parse_args()

    if args.gold:
        paths = sorted(Path("data").glob("GC*_1m.csv"))
        label = "Gold"
        cfg = NineAmConfig(
            stop_mode=args.stop, target_rr=args.rr,
            contracts=args.contracts,
            tick_size=0.10, dollars_per_point=10.0, slippage_ticks=2,
        )
    else:
        paths = (sorted(Path("data").glob("MES*_1m.xlsx"))
                 + sorted(Path("data").glob("MES*_1m.csv"))
                 + sorted(Path("data").glob("ES*_1m.csv")))
        label = "MES/ES"
        cfg = NineAmConfig(
            stop_mode=args.stop, target_rr=args.rr,
            contracts=args.contracts,
        )

    if not paths:
        print("No data files found.")
        return

    print(f"Loading {len(paths)} file(s)...")
    bars = load_excel_bars(paths)
    print(f"Loaded {len(bars):,} bars  ({label})")

    trades = run_nine_am_strategy(bars, cfg)
    print_nine_am_summary(trades, cfg)


if __name__ == "__main__":
    main()
