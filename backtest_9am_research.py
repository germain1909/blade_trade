#!/usr/bin/env python3
"""
9AM 1H OHLC Level Research
===========================

For each trading day:
  1. Build the 9AM-10AM 1-hour candle (Open, High, Low, Close)
  2. During the 10AM hour (10:00-10:59), track every touch of those levels
  3. Measure what happens after the touch — breakout or rejection?
  4. Track MFE/MAE for potential TP/SL sizing

Usage:
    python3 backtest_9am_research.py                    # MES data
    python3 backtest_9am_research.py --gold              # Gold data
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from trading_agent.backtester import load_excel_bars
from trading_agent.projectx_client import Bar

_ET = ZoneInfo("America/New_York")


@dataclass
class LevelTouch:
    date:           date
    level_name:     str       # "9H", "9L", "9O", "9C"
    level_price:    float
    touch_time:     datetime
    touch_price:    float
    direction:      str       # "from_above" or "from_below"
    # What happened after the touch (measured to end of 10AM hour)
    broke_through:  bool
    max_move_with:  float     # max pts in direction of the break/rejection
    max_move_against: float   # max pts against
    close_10am:     float     # price at end of 10AM hour
    result_pts:     float     # close_10am - touch_price (signed)
    # Extended: what happened by end of day
    close_eod:      float
    eod_pts:        float


def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


def run_research(bars: list[Bar], zone_buffer: float = 2.0) -> list[LevelTouch]:
    """Walk through bars, build 9AM OHLC, track touches during 10AM hour."""
    touches: list[LevelTouch] = []

    # Group bars by ET date
    daily_bars: dict[date, list[Bar]] = defaultdict(list)
    for bar in bars:
        d = _et(bar.timestamp).date()
        daily_bars[d].append(bar)

    for day in sorted(daily_bars.keys()):
        day_bars = daily_bars[day]

        # Build 9AM 1-hour candle (9:00-9:59)
        nine_bars = [b for b in day_bars
                     if dtime(9, 0) <= _et(b.timestamp).time() < dtime(10, 0)]
        if len(nine_bars) < 5:
            continue

        nine_o = nine_bars[0].open
        nine_h = max(b.high for b in nine_bars)
        nine_l = min(b.low for b in nine_bars)
        nine_c = nine_bars[-1].close

        levels = {
            "9O": nine_o,
            "9H": nine_h,
            "9L": nine_l,
            "9C": nine_c,
        }

        # Get 10AM hour bars (10:00-10:59)
        ten_bars = [b for b in day_bars
                    if dtime(10, 0) <= _et(b.timestamp).time() < dtime(11, 0)]
        if len(ten_bars) < 5:
            continue

        ten_close = ten_bars[-1].close

        # Get rest of day bars for EOD measurement
        rest_bars = [b for b in day_bars
                     if _et(b.timestamp).time() >= dtime(10, 0)]
        eod_close = rest_bars[-1].close if rest_bars else ten_close

        # Track touches at each level during 10AM hour
        touched_levels: set[str] = set()

        for i, bar in enumerate(ten_bars):
            for name, price in levels.items():
                if name in touched_levels:
                    continue

                # Check if bar touches the level (within buffer)
                touched_from_above = (bar.low <= price + zone_buffer
                                      and bar.high > price)
                touched_from_below = (bar.high >= price - zone_buffer
                                      and bar.low < price)

                if not touched_from_above and not touched_from_below:
                    continue

                direction = "from_above" if bar.close > price else "from_below"

                # Measure what happens after the touch
                remaining = ten_bars[i:]
                if not remaining:
                    continue

                if direction == "from_above":
                    # Came from above, might reject up or break down
                    max_up = max(b.high for b in remaining) - price
                    max_down = price - min(b.low for b in remaining)
                    broke = ten_close < price - zone_buffer
                else:
                    # Came from below, might reject down or break up
                    max_up = max(b.high for b in remaining) - price
                    max_down = price - min(b.low for b in remaining)
                    broke = ten_close > price + zone_buffer

                touches.append(LevelTouch(
                    date=day,
                    level_name=name,
                    level_price=price,
                    touch_time=bar.timestamp,
                    touch_price=bar.close,
                    direction=direction,
                    broke_through=broke,
                    max_move_with=max_up if direction == "from_below" else max_down,
                    max_move_against=max_down if direction == "from_below" else max_up,
                    close_10am=ten_close,
                    result_pts=ten_close - price if direction == "from_below" else price - ten_close,
                    close_eod=eod_close,
                    eod_pts=eod_close - price if direction == "from_below" else price - eod_close,
                ))
                touched_levels.add(name)

    return touches


def print_research(touches: list[LevelTouch], label: str = "") -> None:
    if not touches:
        print("No touches found.")
        return

    print(f"\n{'='*80}")
    print(f"  9AM OHLC LEVEL RESEARCH{' — ' + label if label else ''}")
    print(f"  {len(touches)} level touches across {len(set(t.date for t in touches))} trading days")
    print(f"{'='*80}")

    # Overall: breakout vs rejection
    breaks = [t for t in touches if t.broke_through]
    rejects = [t for t in touches if not t.broke_through]
    print(f"\n── BREAKOUT vs REJECTION ─────────────────────────────────────────")
    print(f"  Breakouts:   {len(breaks):>4}  ({len(breaks)/len(touches)*100:.1f}%)")
    print(f"  Rejections:  {len(rejects):>4}  ({len(rejects)/len(touches)*100:.1f}%)")

    # By level
    print(f"\n── BY LEVEL ─────────────────────────────────────────────────────")
    print(f"  {'Level':<6} {'N':>4}  {'Break%':>7}  {'Reject%':>8}  "
          f"{'AvgMFE':>8}  {'AvgMAE':>8}  {'Avg10AM':>8}  {'AvgEOD':>8}")
    print(f"  {'-'*70}")

    for name in ["9H", "9L", "9O", "9C"]:
        grp = [t for t in touches if t.level_name == name]
        if not grp:
            continue
        brk = sum(1 for t in grp if t.broke_through)
        rej = len(grp) - brk
        avg_mfe = sum(t.max_move_with for t in grp) / len(grp)
        avg_mae = sum(t.max_move_against for t in grp) / len(grp)
        avg_10 = sum(t.result_pts for t in grp) / len(grp)
        avg_eod = sum(t.eod_pts for t in grp) / len(grp)
        print(f"  {name:<6} {len(grp):>4}  {brk/len(grp)*100:>6.1f}%  {rej/len(grp)*100:>7.1f}%  "
              f"{avg_mfe:>+8.2f}  {avg_mae:>+8.2f}  {avg_10:>+8.2f}  {avg_eod:>+8.2f}")

    # By direction of approach
    print(f"\n── BY APPROACH DIRECTION ─────────────────────────────────────────")
    for direction in ["from_above", "from_below"]:
        grp = [t for t in touches if t.direction == direction]
        if not grp:
            continue
        brk = sum(1 for t in grp if t.broke_through)
        avg_10 = sum(t.result_pts for t in grp) / len(grp)
        avg_eod = sum(t.eod_pts for t in grp) / len(grp)
        label_d = "From above (potential support)" if direction == "from_above" else "From below (potential resistance)"
        print(f"  {label_d}")
        print(f"    N={len(grp)}  Break={brk/len(grp)*100:.1f}%  "
              f"Avg10AM={avg_10:+.2f}  AvgEOD={avg_eod:+.2f}")

    # Rejection trades: if we traded the rejection, what would happen?
    print(f"\n── REJECTION TRADE ANALYSIS ──────────────────────────────────────")
    print(f"  If we enter on rejection (close back through level), what's the P&L?")
    print(f"  {'Level':<6} {'N':>4}  {'AvgWin':>8}  {'AvgLoss':>8}  {'WR%':>6}  {'Exp':>8}")
    print(f"  {'-'*50}")

    for name in ["9H", "9L", "9O", "9C"]:
        grp = [t for t in touches if t.level_name == name and not t.broke_through]
        if not grp:
            continue
        winners = [t for t in grp if t.result_pts > 0]
        losers = [t for t in grp if t.result_pts <= 0]
        avg_w = sum(t.result_pts for t in winners) / len(winners) if winners else 0
        avg_l = abs(sum(t.result_pts for t in losers) / len(losers)) if losers else 0
        wr = len(winners) / len(grp) * 100 if grp else 0
        exp = (wr/100 * avg_w) - ((100-wr)/100 * avg_l)
        print(f"  {name:<6} {len(grp):>4}  {avg_w:>+8.2f}  {avg_l:>+8.2f}  {wr:>5.1f}%  {exp:>+8.2f}")

    # Breakout trades: if we traded the breakout, what would happen?
    print(f"\n── BREAKOUT TRADE ANALYSIS ───────────────────────────────────────")
    print(f"  If we enter on breakout (close through level), what's the P&L?")
    print(f"  {'Level':<6} {'N':>4}  {'AvgMFE':>8}  {'AvgMAE':>8}  {'Avg10AM':>8}  {'AvgEOD':>8}")
    print(f"  {'-'*50}")

    for name in ["9H", "9L", "9O", "9C"]:
        grp = [t for t in touches if t.level_name == name and t.broke_through]
        if not grp:
            continue
        avg_mfe = sum(t.max_move_with for t in grp) / len(grp)
        avg_mae = sum(t.max_move_against for t in grp) / len(grp)
        avg_10 = sum(t.result_pts for t in grp) / len(grp)
        avg_eod = sum(t.eod_pts for t in grp) / len(grp)
        print(f"  {name:<6} {len(grp):>4}  {avg_mfe:>+8.2f}  {avg_mae:>+8.2f}  {avg_10:>+8.2f}  {avg_eod:>+8.2f}")

    # MFE/MAE distribution for sizing
    print(f"\n── MFE/MAE DISTRIBUTION (all touches) ───────────────────────────")
    mfes = sorted([t.max_move_with for t in touches])
    maes = sorted([t.max_move_against for t in touches])
    for pct in [25, 50, 75, 90]:
        idx = int(len(mfes) * pct / 100)
        print(f"  P{pct:>2}  MFE={mfes[min(idx, len(mfes)-1)]:>8.2f}  "
              f"MAE={maes[min(idx, len(maes)-1)]:>8.2f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--zone-buf", type=float, default=2.0,
                    help="Points from level to count as a touch")
    args = ap.parse_args()

    if args.gold:
        paths = sorted(Path("data").glob("GC*_1m.csv"))
        label = "Gold (GC)"
        buf = args.zone_buf if args.zone_buf != 2.0 else 5.0
    else:
        paths = (sorted(Path("data").glob("MES*_1m.xlsx"))
                 + sorted(Path("data").glob("MES*_1m.csv"))
                 + sorted(Path("data").glob("ES*_1m.csv")))
        label = "MES/ES"
        buf = args.zone_buf

    if not paths:
        print("No data files found.")
        return

    print(f"Loading {len(paths)} file(s)...")
    bars = load_excel_bars(paths)
    print(f"Loaded {len(bars):,} bars  ({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})")

    touches = run_research(bars, zone_buffer=buf)
    print_research(touches, label)


if __name__ == "__main__":
    main()
