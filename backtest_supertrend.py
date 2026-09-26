#!/usr/bin/env python3
"""
Adaptive Supertrend backtest on MES 1-minute data.

Usage:
    python3 backtest_supertrend.py              # flip-to-flip, 5m bars
    python3 backtest_supertrend.py --bar 30     # flip-to-flip, 30m bars
    python3 backtest_supertrend.py --atr        # ATR level TP/SL, 5m bars
    python3 backtest_supertrend.py --atr --bar 30
    python3 backtest_supertrend.py --bar 5 --manual --lb 10 --ct 1.5 --pw 0.7
"""

import argparse
from datetime import time as dtime
from pathlib import Path

from trading_agent.backtester import load_excel_bars
from trading_agent.supertrend_strategy import (
    SupertrendConfig,
    PivotSupertrendConfig,
    run_supertrend_strategy,
    run_supertrend_atr_strategy,
    run_supertrend_hold_strategy,
    run_pivot_supertrend_strategy,
    print_supertrend_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest Adaptive Supertrend")
    ap.add_argument("--bar", type=int, default=5, help="Bar period in minutes")
    ap.add_argument("--atr", action="store_true",
                    help="Use ATR percentage levels for TP/SL")
    ap.add_argument("--hold", action="store_true",
                    help="Hold to session end, ignore flips while in trade")
    ap.add_argument("--pivot", action="store_true",
                    help="Pivot rejection + Supertrend confluence")
    ap.add_argument("--zone-buf", type=float, default=5.0,
                    help="Pivot zone buffer in points")
    ap.add_argument("--min-rr", type=float, default=1.0,
                    help="Minimum reward:risk ratio")
    ap.add_argument("--manual", action="store_true", help="Use manual params")
    ap.add_argument("--lb", type=int, default=14, help="Lookback period")
    ap.add_argument("--ct", type=float, default=2.0, help="Threshold multiplier")
    ap.add_argument("--pw", type=float, default=0.5, help="Prediction weight")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--start-et", default="08:30", help="Session start (ET)")
    ap.add_argument("--end-et", default="16:00", help="Session end (ET)")
    args = ap.parse_args()

    s_h, s_m = map(int, args.start_et.split(":"))
    e_h, e_m = map(int, args.end_et.split(":"))

    cfg = SupertrendConfig(
        bar_minutes=args.bar,
        lookback=args.lb,
        threshold=args.ct,
        pred_weight=args.pw,
        auto_params=not args.manual,
        contracts=args.contracts,
        trade_start_et=dtime(s_h, s_m),
        trade_end_et=dtime(e_h, e_m),
    )

    paths = sorted(Path("data").glob("MES*_1m.xlsx")) + sorted(Path("data").glob("MES*_1m.csv"))
    # Also pick up ES data (same prices as MES)
    paths += sorted(Path("data").glob("ES*_1m.csv"))
    if not paths:
        print("No MES/ES data files found in data/")
        return

    print(f"Loading {len(paths)} file(s): {[p.name for p in paths]}")
    bars = load_excel_bars(paths)
    print(f"Loaded {len(bars):,} 1-minute bars")

    if args.pivot:
        mode = "pivot-rejection"
        print(f"Mode: {mode}  (zone_buf={args.zone_buf}  min_rr={args.min_rr})")
        pcfg = PivotSupertrendConfig(
            st_bar_minutes=args.bar,
            rej_bar_minutes=5,
            zone_buffer_pts=args.zone_buf,
            min_rr=args.min_rr,
            contracts=args.contracts,
            trade_start_et=dtime(s_h, s_m),
            trade_end_et=dtime(e_h, e_m),
        )
        trades = run_pivot_supertrend_strategy(bars, pcfg)
        # Wrap a SupertrendConfig for the summary printer
        cfg_for_print = SupertrendConfig(
            bar_minutes=args.bar, contracts=args.contracts,
            trade_start_et=dtime(s_h, s_m), trade_end_et=dtime(e_h, e_m),
        )
        print_supertrend_summary(trades, cfg_for_print)
    else:
        mode = ("hold-to-close" if args.hold
                else "ATR level TP/SL" if args.atr
                else "flip-to-flip")
        print(f"Mode: {mode}")

        if args.hold:
            trades = run_supertrend_hold_strategy(bars, cfg)
        elif args.atr:
            trades = run_supertrend_atr_strategy(bars, cfg)
        else:
            trades = run_supertrend_strategy(bars, cfg)

        print_supertrend_summary(trades, cfg)


if __name__ == "__main__":
    main()
