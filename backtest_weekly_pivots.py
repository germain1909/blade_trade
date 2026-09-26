#!/usr/bin/env python3
"""
Weekly Pivot Reversion Strategy — backtest

Classic floor-pivot fade: trade rejections off the weekly S1-S3 / R1-R3
levels computed in weekly_levels.py, targeting the next level in
(toward the pivot).

For each week with a reliable prior-week H/L/C (>= 50% of a full week's
bars), replay that week's 1-minute bars:
  - S1/S2/S3 are long-only zones (fade support, expect a bounce)
  - R1/R2/R3 are short-only zones (fade resistance, expect a rejection)
  - A level is "confirmed" when price touches the zone then closes back
    beyond it by `entry_confirm_ticks` (a rejection close, not just a wick)
  - Stop sits beyond the zone's touch extreme; target is the next level
    inward, or an R-multiple if there's no level beyond (S3/R3)

Usage:
    python3 backtest_weekly_pivots.py                       # NQZ5 (default)
    python3 backtest_weekly_pivots.py --file data/GCZ5_1m.csv --symbol GC
    python3 backtest_weekly_pivots.py --contracts 2 --start-balance 2000
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from weekly_levels import (
    FULL_WEEK_BARS, WeekBar, WeeklyPivots,
    load_bars, build_weekly_bars, compute_pivots, _session_date,
)

ET = ZoneInfo("America/New_York")

INSTRUMENT_SPECS = {
    "NQ":  dict(tick_size=0.25, dollars_per_point=20.0),
    "MNQ": dict(tick_size=0.25, dollars_per_point=2.0),
    "ES":  dict(tick_size=0.25, dollars_per_point=50.0),
    "MES": dict(tick_size=0.25, dollars_per_point=5.0),
    "GC":  dict(tick_size=0.10, dollars_per_point=100.0),
    "MGC": dict(tick_size=0.10, dollars_per_point=10.0),
}


@dataclass
class Config:
    zone_buffer_ticks:     int   = 10
    entry_confirm_ticks:   int   = 4
    max_zone_bars:         int   = 60     # 1m bars a level stays "watching" before expiry
    stop_buffer_ticks:     int   = 8
    target_r_multiple:     float = 1.5    # used only for the outermost levels (S3/R3)
    contracts:             int   = 1
    tick_size:             float = 0.25
    dollars_per_point:     float = 20.0
    slippage_ticks:        int   = 1
    max_trades_per_week:   int   = 4
    daily_loss_limit_usd:  float = -1000.0
    daily_profit_limit_usd: float = 3000.0
    trade_start_et:        dtime = dtime(8, 0)
    trade_end_et:          dtime = dtime(16, 0)


# Level name -> (side, target-inward level name or None)
_LEVEL_CHAIN = {
    "S3": ("long",  "S2"),
    "S2": ("long",  "S1"),
    "S1": ("long",  "P"),
    "R1": ("short", "P"),
    "R2": ("short", "R1"),
    "R3": ("short", "R2"),
}


@dataclass
class LevelState:
    name:      str
    price:     float
    side:      str
    target_name: Optional[str]
    target_price: float
    touched:   bool = False
    confirmed: bool = False
    used:      bool = False
    zone_bars: int  = 0
    extreme:   Optional[float] = None   # furthest excursion since touch
    entry_time: Optional[datetime] = None


@dataclass
class Trade:
    week_start:  date
    level:       str
    direction:   str
    entry_time:  datetime
    entry_px:    float
    stop_px:     float
    target_px:   float
    exit_time:   datetime
    exit_px:     float
    exit_reason: str
    pnl_usd:     float
    pnl_r:       float


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    return start <= t <= end


def build_level_states(pv: WeeklyPivots) -> list[LevelState]:
    prices = {"S3": pv.s3, "S2": pv.s2, "S1": pv.s1, "P": pv.p,
              "R1": pv.r1, "R2": pv.r2, "R3": pv.r3}
    states = []
    for name, (side, target_name) in _LEVEL_CHAIN.items():
        states.append(LevelState(
            name=name, price=prices[name], side=side,
            target_name=target_name, target_price=prices[target_name],
        ))
    return states


def run_week(pv: WeeklyPivots, week_bars: pd.DataFrame, cfg: Config) -> list[Trade]:
    states = build_level_states(pv)
    tick = cfg.tick_size
    zbuf = cfg.zone_buffer_ticks * tick
    cbuf = cfg.entry_confirm_ticks * tick
    sbuf = cfg.stop_buffer_ticks * tick

    trades: list[Trade] = []
    in_trade = False
    active: Optional[LevelState] = None
    trade_dir = ""
    entry_px = stop_px = target_px = 0.0

    day_pnl: dict[date, float] = defaultdict(float)
    day_trades: dict[date, int] = defaultdict(int)
    week_trade_count = 0

    for row in week_bars.itertuples():
        ts_et = row.timestamp.astimezone(ET)
        bar_time = ts_et.time()
        bar_date = ts_et.date()

        # ── Manage open trade ────────────────────────────────────────
        if in_trade and active is not None:
            risk = abs(entry_px - stop_px)
            hit_tp = (row.high >= target_px) if trade_dir == "long" else (row.low <= target_px)
            hit_sl = (row.low <= stop_px)     if trade_dir == "long" else (row.high >= stop_px)
            if hit_tp or hit_sl:
                # conservative: stop takes priority if both touch in the same bar
                if hit_sl:
                    ep, reason = stop_px, "stop"
                else:
                    ep, reason = target_px, "target"
                pnl = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
                      * cfg.dollars_per_point * cfg.contracts
                r = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) / risk if risk else 0.0
                trades.append(Trade(
                    week_start=pv.week_start, level=active.name, direction=trade_dir,
                    entry_time=active.entry_time, entry_px=entry_px, stop_px=stop_px,
                    target_px=target_px, exit_time=row.timestamp, exit_px=ep,
                    exit_reason=reason, pnl_usd=pnl, pnl_r=r,
                ))
                day_pnl[bar_date] += pnl
                day_trades[bar_date] += 1
                active.used = True
                in_trade = False
                active = None

        if not _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et):
            continue

        locked = (day_pnl[bar_date] <= cfg.daily_loss_limit_usd
                  or day_pnl[bar_date] >= cfg.daily_profit_limit_usd
                  or week_trade_count >= cfg.max_trades_per_week)
        if in_trade or locked:
            continue

        # ── Update zone states / look for entries ───────────────────
        for ls in states:
            if ls.used:
                continue

            at_zone = (row.low <= ls.price + zbuf) if ls.side == "long" \
                      else (row.high >= ls.price - zbuf)

            if not ls.touched:
                if at_zone:
                    ls.touched = True
                    ls.zone_bars = 0
                    ls.extreme = row.low if ls.side == "long" else row.high
                continue

            ls.zone_bars += 1
            if ls.zone_bars > cfg.max_zone_bars:
                ls.used = True
                continue

            ls.extreme = (min(ls.extreme, row.low) if ls.side == "long"
                          else max(ls.extreme, row.high))

            confirmed = (row.close >= ls.price + cbuf) if ls.side == "long" \
                        else (row.close <= ls.price - cbuf)
            if not confirmed:
                continue

            direction = ls.side
            ep = row.close + tick * cfg.slippage_ticks if direction == "long" \
                 else row.close - tick * cfg.slippage_ticks
            stp = ls.extreme - sbuf if direction == "long" else ls.extreme + sbuf
            risk = abs(ep - stp)
            if risk <= 0:
                ls.used = True
                continue

            tgt = ls.target_price
            # fall back to R-multiple if the "next level in" doesn't make sense
            if direction == "long" and tgt <= ep:
                tgt = ep + risk * cfg.target_r_multiple
            if direction == "short" and tgt >= ep:
                tgt = ep - risk * cfg.target_r_multiple

            ls.entry_time = row.timestamp
            in_trade = True
            active = ls
            trade_dir = direction
            entry_px, stop_px, target_px = ep, stp, tgt
            week_trade_count += 1
            break

    return trades


def print_summary(trades: list[Trade], symbol: str, start_balance: float = 0.0) -> None:
    if not trades:
        print("\nNo trades generated.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total_pnl = sum(t.pnl_usd for t in trades)
    win_rate = len(wins) / len(trades) * 100
    gross_win = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    pf = gross_win / gross_loss if gross_loss else float("inf")

    equity = peak = max_dd = 0.0
    for t in trades:
        equity += t.pnl_usd
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    print(f"\n{'='*68}")
    print(f"  BACKTEST SUMMARY — {symbol} Weekly Pivot Reversion")
    print(f"{'='*68}")
    if start_balance > 0:
        print(f"  Starting balance: ${start_balance:>10,.2f}")
        print(f"  Ending balance  : ${start_balance + total_pnl:>10,.2f}  "
              f"({total_pnl / start_balance * 100:+.1f}%)")
    print(f"  Total trades    : {len(trades)}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total PnL       : ${total_pnl:>10,.2f}")
    if wins:
        print(f"  Avg win         : ${sum(t.pnl_usd for t in wins)/len(wins):>10,.2f}")
    if losses:
        print(f"  Avg loss        : ${sum(t.pnl_usd for t in losses)/len(losses):>10,.2f}")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Max drawdown    : ${max_dd:>10,.2f}"
          + (f"  ({max_dd / start_balance * 100:.1f}% of starting balance)" if start_balance > 0 else ""))

    print(f"\n  By level:")
    by_level = defaultdict(list)
    for t in trades:
        by_level[t.level].append(t)
    for lvl in ["S3", "S2", "S1", "R1", "R2", "R3"]:
        grp = by_level.get(lvl, [])
        if not grp:
            continue
        w = sum(1 for t in grp if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in grp)
        print(f"    {lvl:<4} {len(grp):>3} trades  {w}W/{len(grp)-w}L  ${pnl:>+10,.2f}")

    print(f"\n  By exit reason:")
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t.pnl_usd)
    for reason, pnls in sorted(by_reason.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<10} {len(pnls):>3} trades  {w}W/{len(pnls)-w}L  ${sum(pnls):>+10,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/NQZ5_1m.csv", help="1m bar CSV to load")
    ap.add_argument("--symbol", default="NQ", help="Instrument for tick/point value lookup")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--start-balance", type=float, default=0.0, dest="start_balance")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    spec = INSTRUMENT_SPECS.get(args.symbol.upper(), INSTRUMENT_SPECS["NQ"])
    cfg = Config(contracts=args.contracts, tick_size=spec["tick_size"],
                 dollars_per_point=spec["dollars_per_point"])

    print(f"Loading {path} ...")
    df = load_bars(path)
    print(f"Loaded {len(df):,} bars")

    weeks = build_weekly_bars(df)
    pivots = compute_pivots(weeks)
    bar_count_by_week = {w.week_start: w.bar_count for w in weeks}

    et = df["timestamp"].dt.tz_convert(ET)
    df = df.assign(session_date=et.apply(_session_date))

    reliable = [pv for pv in pivots
                if pv.reliable and bar_count_by_week.get(pv.week_start, 0) >= FULL_WEEK_BARS * 0.5]
    skipped = len(pivots) - len(reliable)
    print(f"{len(reliable)} week(s) with reliable prior-week AND current-week data "
          f"({skipped} skipped for gappy source data)")

    all_trades: list[Trade] = []
    for pv in reliable:
        week_end = [w for w in weeks if w.week_start == pv.week_start][0].week_end
        mask = (df["session_date"] >= pv.week_start) & (df["session_date"] <= week_end)
        week_bars = df.loc[mask].sort_values("timestamp")
        if week_bars.empty:
            continue
        trades = run_week(pv, week_bars, cfg)
        all_trades.extend(trades)

    print_summary(all_trades, args.symbol.upper(), start_balance=args.start_balance)


if __name__ == "__main__":
    main()
