#!/usr/bin/env python3
"""
Weekly Swing-Extreme Fade — backtest

Instead of fading fixed floor-pivot prices, this tracks the week's running
high/low AS IT UNFOLDS and fades every fresh extreme that stalls out:

  - Whenever price makes a new week-high, that high becomes the "candidate."
  - If price keeps pushing higher, the candidate just rolls up with it.
  - Once price stops extending and closes back below the candidate high by
    `confirm_ticks` within `max_wait_bars`, that's a rejection -> go short.
  - Symmetric logic for week-lows -> long.
  - A candidate that gets traded (or expires) needs a NEW, higher/lower
    extreme to re-arm -> avoids re-firing on the exact same high repeatedly.

This fires far more often than static pivot levels since it reacts to
whatever shape the week actually takes, aiming for a handful of trades
per week rather than one shot per pre-computed level.

Usage:
    python3 backtest_weekly_swing_fade.py
    python3 backtest_weekly_swing_fade.py --file data/GCZ5_1m.csv --symbol GC
    python3 backtest_weekly_swing_fade.py --contracts 2 --start-balance 2000
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from weekly_levels import FULL_WEEK_BARS, build_weekly_bars, _session_date

ET = ZoneInfo("America/New_York")

INSTRUMENT_SPECS = {
    "NQ":  dict(tick_size=0.25, dollars_per_point=20.0),
    "MNQ": dict(tick_size=0.25, dollars_per_point=2.0),
    "ES":  dict(tick_size=0.25, dollars_per_point=50.0),
    "MES": dict(tick_size=0.25, dollars_per_point=5.0),
    "GC":  dict(tick_size=0.10, dollars_per_point=100.0),
    "MGC": dict(tick_size=0.10, dollars_per_point=10.0),
    "YM":  dict(tick_size=1.0, dollars_per_point=5.0),
    "MYM": dict(tick_size=1.0, dollars_per_point=0.50),
}


@dataclass
class Config:
    confirm_ticks:          int   = 4     # rejection close distance from the extreme
    max_wait_bars:          int   = 15    # bars to wait for rejection before candidate expires
    stop_buffer_ticks:      int   = 8
    max_risk_usd:           float = 500.0  # cap risk/contract by tightening the stop, not skipping the trade
    target_r_multiple:      float = 1.5
    contracts:              int   = 1
    tick_size:              float = 0.25
    dollars_per_point:      float = 20.0
    slippage_ticks:         int   = 1
    max_trades_per_week:    int   = 4
    daily_loss_limit_usd:   float = -1000.0
    daily_profit_limit_usd: float = 3000.0
    use_trend_filter:       bool  = False
    ma_period_days:         int   = 20     # daily SMA lookback for the trend filter
    trade_start_et:         dtime = dtime(8, 0)
    trade_end_et:           dtime = dtime(16, 0)


@dataclass
class Trade:
    week_start:  date
    direction:   str
    extreme_px:  float
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


def build_daily_trend_ma(df: pd.DataFrame, period_days: int) -> dict[date, float]:
    """Rolling SMA of daily closes, indexed by the date it applies to (i.e. computed
    from the `period_days` closes strictly BEFORE that date, so there's no lookahead)."""
    daily_close = df.groupby("session_date")["close"].last().sort_index()
    sma = daily_close.rolling(period_days).mean().shift(1)
    return {d: v for d, v in sma.items() if pd.notna(v)}


def run_week(week_bars: pd.DataFrame, week_start: date, cfg: Config,
             trend_ma: dict[date, float] | None = None) -> list[Trade]:
    tick = cfg.tick_size
    cbuf = cfg.confirm_ticks * tick
    sbuf = cfg.stop_buffer_ticks * tick
    max_stop_pts = cfg.max_risk_usd / (cfg.dollars_per_point * cfg.contracts)

    trades: list[Trade] = []
    in_trade = False
    trade_dir = ""
    entry_px = stop_px = target_px = 0.0
    entry_time: Optional[datetime] = None
    faded_extreme = 0.0

    running_high: Optional[float] = None
    running_low:  Optional[float] = None
    cand_high: Optional[float] = None   # armed candidate high awaiting rejection
    cand_high_bars = 0
    cand_high_used = False              # this exact extreme already traded/expired
    cand_low:  Optional[float] = None
    cand_low_bars = 0
    cand_low_used = False

    day_pnl:    dict[date, float] = defaultdict(float)
    week_trade_count = 0

    for row in week_bars.itertuples():
        ts_et = row.timestamp.astimezone(ET)
        bar_time = ts_et.time()
        bar_date = ts_et.date()

        if running_high is None:
            running_high = row.high
            running_low  = row.low

        # ── Manage open trade ────────────────────────────────────────
        if in_trade:
            risk = abs(entry_px - stop_px)
            hit_tp = (row.high >= target_px) if trade_dir == "long" else (row.low <= target_px)
            hit_sl = (row.low <= stop_px)     if trade_dir == "long" else (row.high >= stop_px)
            if hit_tp or hit_sl:
                ep, reason = (stop_px, "stop") if hit_sl else (target_px, "target")
                pnl = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
                      * cfg.dollars_per_point * cfg.contracts
                r = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) / risk if risk else 0.0
                trades.append(Trade(
                    week_start=week_start, direction=trade_dir, extreme_px=faded_extreme,
                    entry_time=entry_time, entry_px=entry_px, stop_px=stop_px,
                    target_px=target_px, exit_time=row.timestamp, exit_px=ep,
                    exit_reason=reason, pnl_usd=pnl, pnl_r=r,
                ))
                day_pnl[bar_date] += pnl
                in_trade = False

        # ── Track running extremes / arm candidates ─────────────────
        if row.high > running_high:
            running_high = row.high
            cand_high = running_high
            cand_high_bars = 0
            cand_high_used = False
        elif cand_high is not None and not cand_high_used:
            cand_high_bars += 1

        if row.low < running_low:
            running_low = row.low
            cand_low = running_low
            cand_low_bars = 0
            cand_low_used = False
        elif cand_low is not None and not cand_low_used:
            cand_low_bars += 1

        if not _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et):
            continue

        locked = (day_pnl[bar_date] <= cfg.daily_loss_limit_usd
                  or day_pnl[bar_date] >= cfg.daily_profit_limit_usd
                  or week_trade_count >= cfg.max_trades_per_week)
        if in_trade or locked:
            continue

        # ── Short: rejection from the high candidate ─────────────────
        trend_blocks_short = (cfg.use_trend_filter and trend_ma is not None
                               and trend_ma.get(bar_date) is not None
                               and row.close > trend_ma[bar_date])
        if (not trend_blocks_short and cand_high is not None and not cand_high_used
                and cand_high_bars > 0 and row.close <= cand_high - cbuf):
            ep  = row.close - tick * cfg.slippage_ticks
            stp = min(cand_high + sbuf, ep + max_stop_pts)
            risk = stp - ep
            if risk > 0:
                in_trade = True
                trade_dir = "short"
                entry_px, stop_px = ep, stp
                target_px = ep - risk * cfg.target_r_multiple
                entry_time = row.timestamp
                faded_extreme = cand_high
                cand_high_used = True
                week_trade_count += 1
                continue
        if cand_high is not None and cand_high_bars > cfg.max_wait_bars:
            cand_high_used = True

        # ── Long: rejection from the low candidate ────────────────────
        if (cand_low is not None and not cand_low_used
                and cand_low_bars > 0 and row.close >= cand_low + cbuf):
            ep  = row.close + tick * cfg.slippage_ticks
            stp = max(cand_low - sbuf, ep - max_stop_pts)
            risk = ep - stp
            if risk > 0:
                in_trade = True
                trade_dir = "long"
                entry_px, stop_px = ep, stp
                target_px = ep + risk * cfg.target_r_multiple
                entry_time = row.timestamp
                faded_extreme = cand_low
                cand_low_used = True
                week_trade_count += 1
                continue
        if cand_low is not None and cand_low_bars > cfg.max_wait_bars:
            cand_low_used = True

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

    by_week = defaultdict(list)
    for t in trades:
        by_week[t.week_start].append(t)

    print(f"\n{'='*68}")
    print(f"  BACKTEST SUMMARY — {symbol} Weekly Swing-Extreme Fade")
    print(f"{'='*68}")
    if start_balance > 0:
        print(f"  Starting balance: ${start_balance:>10,.2f}")
        print(f"  Ending balance  : ${start_balance + total_pnl:>10,.2f}  "
              f"({total_pnl / start_balance * 100:+.1f}%)")
    print(f"  Weeks tested    : {len(by_week)}   Avg trades/week: {len(trades)/max(len(by_week),1):.1f}")
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

    print(f"\n  By direction:")
    for d in ["long", "short"]:
        grp = [t for t in trades if t.direction == d]
        if not grp:
            continue
        w = sum(1 for t in grp if t.pnl_usd > 0)
        print(f"    {d:<6} {len(grp):>3} trades  {w}W/{len(grp)-w}L  ${sum(t.pnl_usd for t in grp):>+10,.2f}")

    print(f"\n  By exit reason:")
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t.exit_reason].append(t.pnl_usd)
    for reason, pnls in sorted(by_reason.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<10} {len(pnls):>3} trades  {w}W/{len(pnls)-w}L  ${sum(pnls):>+10,.2f}")

    print(f"\n  By week:")
    for wk in sorted(by_week):
        grp = by_week[wk]
        w = sum(1 for t in grp if t.pnl_usd > 0)
        print(f"    {wk}  {len(grp)} trades  {w}W/{len(grp)-w}L  ${sum(t.pnl_usd for t in grp):>+10,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/NQZ5_1m.csv", help="1m bar CSV to load")
    ap.add_argument("--symbol", default="NQ", help="Instrument for tick/point value lookup")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--start-balance", type=float, default=0.0, dest="start_balance")
    ap.add_argument("--trend-filter", action="store_true", dest="trend_filter",
                    help="Skip shorts when price is above the daily SMA (long-only in an uptrend)")
    ap.add_argument("--ma-period", type=int, default=20, dest="ma_period",
                    help="Daily SMA lookback in days for the trend filter (default 20)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    spec = INSTRUMENT_SPECS.get(args.symbol.upper(), INSTRUMENT_SPECS["NQ"])
    cfg = Config(contracts=args.contracts, tick_size=spec["tick_size"],
                 dollars_per_point=spec["dollars_per_point"],
                 use_trend_filter=args.trend_filter, ma_period_days=args.ma_period)

    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} bars")

    weeks = build_weekly_bars(df)
    et = df["timestamp"].dt.tz_convert(ET)
    df = df.assign(session_date=et.apply(_session_date))

    reliable_weeks = [w for w in weeks if w.bar_count >= FULL_WEEK_BARS * 0.5]
    skipped = len(weeks) - len(reliable_weeks)
    print(f"{len(reliable_weeks)} week(s) with full data ({skipped} skipped for gappy source data)")

    trend_ma = build_daily_trend_ma(df, cfg.ma_period_days) if cfg.use_trend_filter else None
    if cfg.use_trend_filter:
        print(f"Trend filter ON: shorts skipped when price > {cfg.ma_period_days}-day SMA")

    all_trades: list[Trade] = []
    for w in reliable_weeks:
        mask = (df["session_date"] >= w.week_start) & (df["session_date"] <= w.week_end)
        week_bars = df.loc[mask].sort_values("timestamp")
        if week_bars.empty:
            continue
        all_trades.extend(run_week(week_bars, w.week_start, cfg, trend_ma=trend_ma))

    print_summary(all_trades, args.symbol.upper(), start_balance=args.start_balance)


if __name__ == "__main__":
    main()
