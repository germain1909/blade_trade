#!/usr/bin/env python3
"""
9AM Candle Analyzer — Pre-trade intelligence for manual 10AM trading.

Analyzes historical 9AM candle patterns and what happened at each level
during the 10AM hour. Segments by candle type, range size, and close
position to give actionable insights.

Usage:
    python3 analyze_9am.py                # full analysis on MES/ES
    python3 analyze_9am.py --gold         # gold analysis
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


def _et(ts):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


@dataclass
class DayProfile:
    date:       date
    nine_o:     float
    nine_h:     float
    nine_l:     float
    nine_c:     float
    nine_range: float
    candle_type: str      # "bullish", "bearish", "doji"
    range_class: str      # "narrow", "normal", "wide"
    close_pos:   str      # "upper", "middle", "lower" (where close is in range)

    # 10AM hour outcomes
    ten_high:    float
    ten_low:     float
    ten_close:   float
    broke_high:  bool
    broke_low:   bool
    first_break: str      # "high", "low", "neither"

    # What happened at each level
    high_touch:       bool
    high_reject:      bool     # touched and closed back inside
    high_break:       bool     # touched and closed through
    high_move_after:  float    # pts move in break direction after touch

    low_touch:        bool
    low_reject:       bool
    low_break:        bool
    low_move_after:   float

    # EOD
    eod_close:   float
    eod_vs_nine_h: float   # eod - nine_h (positive = above high)
    eod_vs_nine_l: float   # nine_l - eod (positive = below low)


def build_profiles(bars: list[Bar]) -> list[DayProfile]:
    daily_bars: dict[date, list[Bar]] = defaultdict(list)
    for bar in bars:
        d = _et(bar.timestamp).date()
        daily_bars[d].append(bar)

    # Compute ATR for range classification
    ranges = []
    profiles = []

    for day in sorted(daily_bars.keys()):
        day_bars = daily_bars[day]

        nine_bars = [b for b in day_bars
                     if dtime(9, 0) <= _et(b.timestamp).time() < dtime(10, 0)]
        ten_bars = [b for b in day_bars
                    if dtime(10, 0) <= _et(b.timestamp).time() < dtime(11, 0)]
        rest_bars = [b for b in day_bars
                     if _et(b.timestamp).time() >= dtime(10, 0)]

        if len(nine_bars) < 5 or len(ten_bars) < 5:
            continue

        nine_o = nine_bars[0].open
        nine_h = max(b.high for b in nine_bars)
        nine_l = min(b.low for b in nine_bars)
        nine_c = nine_bars[-1].close
        nine_range = nine_h - nine_l
        ranges.append(nine_range)

        # Candle type
        body = nine_c - nine_o
        if abs(body) < nine_range * 0.1:
            candle_type = "doji"
        elif body > 0:
            candle_type = "bullish"
        else:
            candle_type = "bearish"

        # Close position in range
        if nine_range > 0:
            pos = (nine_c - nine_l) / nine_range
        else:
            pos = 0.5
        close_pos = "upper" if pos > 0.66 else ("lower" if pos < 0.33 else "middle")

        # 10AM outcomes
        ten_high  = max(b.high for b in ten_bars)
        ten_low   = min(b.low for b in ten_bars)
        ten_close = ten_bars[-1].close
        eod_close = rest_bars[-1].close if rest_bars else ten_close

        broke_high = ten_high > nine_h
        broke_low  = ten_low < nine_l

        # Which broke first
        first_break = "neither"
        for b in ten_bars:
            if b.high > nine_h and first_break == "neither":
                first_break = "high"
                break
            if b.low < nine_l and first_break == "neither":
                first_break = "low"
                break

        # High level analysis
        high_touch = any(b.high >= nine_h - 2.0 for b in ten_bars)
        high_break_val = ten_close > nine_h
        high_reject_val = high_touch and not high_break_val

        high_move = 0.0
        for b in ten_bars:
            if b.high >= nine_h - 2.0:
                remaining = [bb for bb in ten_bars if bb.timestamp >= b.timestamp]
                if remaining:
                    high_move = max(bb.high for bb in remaining) - nine_h
                break

        # Low level analysis
        low_touch = any(b.low <= nine_l + 2.0 for b in ten_bars)
        low_break_val = ten_close < nine_l
        low_reject_val = low_touch and not low_break_val

        low_move = 0.0
        for b in ten_bars:
            if b.low <= nine_l + 2.0:
                remaining = [bb for bb in ten_bars if bb.timestamp >= b.timestamp]
                if remaining:
                    low_move = nine_l - min(bb.low for bb in remaining)
                break

        # Range classification (use rolling median)
        avg_range = sum(ranges[-20:]) / len(ranges[-20:]) if len(ranges) >= 20 else sum(ranges) / len(ranges)
        if nine_range < avg_range * 0.7:
            range_class = "narrow"
        elif nine_range > avg_range * 1.3:
            range_class = "wide"
        else:
            range_class = "normal"

        profiles.append(DayProfile(
            date=day, nine_o=nine_o, nine_h=nine_h, nine_l=nine_l, nine_c=nine_c,
            nine_range=nine_range, candle_type=candle_type, range_class=range_class,
            close_pos=close_pos,
            ten_high=ten_high, ten_low=ten_low, ten_close=ten_close,
            broke_high=broke_high, broke_low=broke_low, first_break=first_break,
            high_touch=high_touch, high_reject=high_reject_val, high_break=high_break_val,
            high_move_after=high_move,
            low_touch=low_touch, low_reject=low_reject_val, low_break=low_break_val,
            low_move_after=low_move,
            eod_close=eod_close,
            eod_vs_nine_h=eod_close - nine_h,
            eod_vs_nine_l=nine_l - eod_close,
        ))

    return profiles


def print_analysis(profiles: list[DayProfile], label: str = "") -> None:
    if not profiles:
        print("No data."); return

    n = len(profiles)
    print(f"\n{'='*80}")
    print(f"  9AM CANDLE PLAYBOOK{' — ' + label if label else ''}")
    print(f"  {n} trading days analyzed")
    print(f"{'='*80}")

    # ── Overall: what happens during 10AM ─────────────────────────
    print(f"\n── WHAT HAPPENS AT 10AM ──────────────────────────────────────────")
    bh = sum(1 for p in profiles if p.broke_high)
    bl = sum(1 for p in profiles if p.broke_low)
    both = sum(1 for p in profiles if p.broke_high and p.broke_low)
    neither = sum(1 for p in profiles if not p.broke_high and not p.broke_low)
    print(f"  Broke 9AM High:  {bh:>3} ({bh/n*100:.0f}%)")
    print(f"  Broke 9AM Low:   {bl:>3} ({bl/n*100:.0f}%)")
    print(f"  Broke Both:      {both:>3} ({both/n*100:.0f}%)")
    print(f"  Stayed Inside:   {neither:>3} ({neither/n*100:.0f}%)")

    fb_h = sum(1 for p in profiles if p.first_break == "high")
    fb_l = sum(1 for p in profiles if p.first_break == "low")
    fb_n = sum(1 for p in profiles if p.first_break == "neither")
    print(f"\n  First break direction:")
    print(f"    High first:    {fb_h:>3} ({fb_h/n*100:.0f}%)")
    print(f"    Low first:     {fb_l:>3} ({fb_l/n*100:.0f}%)")
    print(f"    Neither:       {fb_n:>3} ({fb_n/n*100:.0f}%)")

    # ── By candle type ────────────────────────────────────────────
    print(f"\n── BY 9AM CANDLE TYPE ────────────────────────────────────────────")
    for ctype in ["bullish", "bearish", "doji"]:
        grp = [p for p in profiles if p.candle_type == ctype]
        if not grp: continue
        g = len(grp)
        bh = sum(1 for p in grp if p.broke_high)
        bl = sum(1 for p in grp if p.broke_low)
        avg_eod_h = sum(p.eod_vs_nine_h for p in grp) / g
        avg_eod_l = sum(p.eod_vs_nine_l for p in grp) / g
        hr = sum(1 for p in grp if p.high_reject)
        hb = sum(1 for p in grp if p.high_break)
        lr = sum(1 for p in grp if p.low_reject)
        lb = sum(1 for p in grp if p.low_break)

        print(f"\n  {ctype.upper()} candle ({g} days, {g/n*100:.0f}%):")
        print(f"    Broke high: {bh/g*100:.0f}%  |  Broke low: {bl/g*100:.0f}%")
        print(f"    At 9AM HIGH → reject: {hr} ({hr/g*100:.0f}%)  break: {hb} ({hb/g*100:.0f}%)")
        print(f"    At 9AM LOW  → reject: {lr} ({lr/g*100:.0f}%)  break: {lb} ({lb/g*100:.0f}%)")
        print(f"    EOD vs 9H: {avg_eod_h:+.1f} pts  |  EOD vs 9L: {avg_eod_l:+.1f} pts")

        # The playbook
        if ctype == "bullish":
            if bh/g > 0.5:
                print(f"    → PLAY: High likely breaks. Look for long breakout above 9H.")
            if lr/g > 0.5:
                print(f"    → PLAY: Low tends to hold. Buy dips to 9L with tight stop below.")
        elif ctype == "bearish":
            if bl/g > 0.5:
                print(f"    → PLAY: Low likely breaks. Look for short breakout below 9L.")
            if hr/g > 0.5:
                print(f"    → PLAY: High tends to hold. Sell rips to 9H with tight stop above.")
        elif ctype == "doji":
            print(f"    → PLAY: Indecision. Wait for a clear break of either level before acting.")

    # ── By range size ─────────────────────────────────────────────
    print(f"\n── BY 9AM RANGE SIZE ─────────────────────────────────────────────")
    for rclass in ["narrow", "normal", "wide"]:
        grp = [p for p in profiles if p.range_class == rclass]
        if not grp: continue
        g = len(grp)
        bh = sum(1 for p in grp if p.broke_high)
        bl = sum(1 for p in grp if p.broke_low)
        both = sum(1 for p in grp if p.broke_high and p.broke_low)
        avg_range = sum(p.nine_range for p in grp) / g
        avg_mfe_h = sum(p.high_move_after for p in grp if p.high_touch) / max(1, sum(1 for p in grp if p.high_touch))
        avg_mfe_l = sum(p.low_move_after for p in grp if p.low_touch) / max(1, sum(1 for p in grp if p.low_touch))

        print(f"\n  {rclass.upper()} range ({g} days, avg {avg_range:.1f} pts):")
        print(f"    Broke high: {bh/g*100:.0f}%  |  Broke low: {bl/g*100:.0f}%  |  Both: {both/g*100:.0f}%")
        print(f"    Avg move after high touch: {avg_mfe_h:+.1f} pts")
        print(f"    Avg move after low touch:  {avg_mfe_l:+.1f} pts")

        if rclass == "narrow":
            print(f"    → PLAY: Narrow range = coiled. Breakout likely. Trade the first clean break.")
        elif rclass == "wide":
            print(f"    → PLAY: Wide range = levels more meaningful. Rejections more likely to hold.")

    # ── By close position ─────────────────────────────────────────
    print(f"\n── BY 9AM CLOSE POSITION ─────────────────────────────────────────")
    for cpos in ["upper", "middle", "lower"]:
        grp = [p for p in profiles if p.close_pos == cpos]
        if not grp: continue
        g = len(grp)
        bh = sum(1 for p in grp if p.broke_high)
        bl = sum(1 for p in grp if p.broke_low)
        fb_h = sum(1 for p in grp if p.first_break == "high")
        fb_l = sum(1 for p in grp if p.first_break == "low")

        print(f"\n  Close in {cpos.upper()} third ({g} days):")
        print(f"    Broke high: {bh/g*100:.0f}%  |  Broke low: {bl/g*100:.0f}%")
        print(f"    First break high: {fb_h/g*100:.0f}%  |  First break low: {fb_l/g*100:.0f}%")

        if cpos == "upper":
            print(f"    → Strong close. Bias long. High break more likely.")
        elif cpos == "lower":
            print(f"    → Weak close. Bias short. Low break more likely.")
        else:
            print(f"    → Neutral close. No directional bias. Wait for the break.")

    # ── Quick reference cheat sheet ───────────────────────────────
    print(f"\n── CHEAT SHEET ───────────────────────────────────────────────────")
    print(f"  Read the 9AM candle. Check these three things:")
    print(f"")
    print(f"  1. CANDLE TYPE (bullish/bearish/doji)")
    print(f"     Bullish → bias long, look to buy breaks of 9H or dips to 9L")
    print(f"     Bearish → bias short, look to sell breaks of 9L or rips to 9H")
    print(f"     Doji    → no bias, wait for the first clean break")
    print(f"")
    print(f"  2. RANGE SIZE (compare to recent days)")
    print(f"     Narrow  → breakout likely, trade the first break aggressively")
    print(f"     Wide    → levels more likely to hold, favor rejections")
    print(f"")
    print(f"  3. CLOSE POSITION (where did 9AM close in the range?)")
    print(f"     Upper third → bulls in control, high break favored")
    print(f"     Lower third → bears in control, low break favored")
    print(f"     Middle      → toss-up, wait for confirmation")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true")
    args = ap.parse_args()

    if args.gold:
        paths = sorted(Path("data").glob("GC*_1m.csv"))
        label = "Gold (GC)"
    else:
        paths = (sorted(Path("data").glob("MES*_1m.xlsx"))
                 + sorted(Path("data").glob("ES*_1m.csv")))
        label = "MES/ES"

    if not paths:
        print("No data."); return

    bars = load_excel_bars(paths)
    print(f"Loaded {len(bars):,} bars")
    profiles = build_profiles(bars)
    print_analysis(profiles, label)


if __name__ == "__main__":
    main()
