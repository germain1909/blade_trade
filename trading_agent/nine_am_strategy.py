from __future__ import annotations

"""
9AM Breakout Strategy
=====================

Build the 9AM 1-hour candle (9:00-9:59 ET), then during the 10AM hour
(10:00-10:59 ET) trade breakouts of the 9AM high and low.

  - Price closes above 9AM high → long
  - Price closes below 9AM low  → short
  - Stop: opposite side of the 9AM range (or midpoint for tighter stop)
  - Target: R multiple of risk

One trade per direction per day. Checked on 1m bars for precise entry.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")


@dataclass
class NineAmConfig:
    stop_mode:         str   = "opposite"   # "opposite" = other side of range, "mid" = midpoint
    target_rr:         float = 2.0
    contracts:         int   = 1
    tick_size:         float = 0.25
    slippage_ticks:    int   = 1
    dollars_per_point: float = 5.0
    max_trades_per_day: int  = 2

    nine_start_et:     dtime = dtime(9, 0)
    nine_end_et:       dtime = dtime(10, 0)
    ten_start_et:      dtime = dtime(10, 0)
    ten_end_et:        dtime = dtime(11, 0)
    session_end_et:    dtime = dtime(16, 0)


@dataclass
class NineAmTrade:
    date:          date
    entry_time:    datetime
    exit_time:     datetime
    direction:     str
    entry_price:   float
    stop_price:    float
    target_price:  float
    exit_price:    float
    exit_reason:   str
    pnl_usd:       float
    nine_high:     float
    nine_low:      float
    nine_open:     float
    nine_close:    float


def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


def run_nine_am_strategy(
    bars: list[Bar],
    cfg: NineAmConfig,
) -> list[NineAmTrade]:
    """Walk-forward backtest of the 9AM breakout strategy."""
    slip = cfg.slippage_ticks * cfg.tick_size
    dpp  = cfg.dollars_per_point * cfg.contracts

    trades: list[NineAmTrade] = []

    # Group bars by ET date
    daily_bars: dict[date, list[Bar]] = defaultdict(list)
    for bar in bars:
        d = _et(bar.timestamp).date()
        daily_bars[d].append(bar)

    for day in sorted(daily_bars.keys()):
        day_bars = daily_bars[day]

        # Build 9AM 1-hour candle
        nine_bars = [b for b in day_bars
                     if cfg.nine_start_et <= _et(b.timestamp).time() < cfg.nine_end_et]
        if len(nine_bars) < 5:
            continue

        nine_o = nine_bars[0].open
        nine_h = max(b.high for b in nine_bars)
        nine_l = min(b.low for b in nine_bars)
        nine_c = nine_bars[-1].close
        nine_mid = (nine_h + nine_l) / 2.0

        # Get 10AM hour + rest of day bars
        action_bars = [b for b in day_bars
                       if _et(b.timestamp).time() >= cfg.ten_start_et]
        if not action_bars:
            continue

        # Track state
        in_trade    = False
        t_dir:      Optional[str]   = None
        t_entry:    Optional[float] = None
        t_stop:     Optional[float] = None
        t_target:   Optional[float] = None
        t_ts:       Optional[datetime] = None
        day_trades  = 0
        took_long   = False
        took_short  = False

        for bar in action_bars:
            bar_time = _et(bar.timestamp).time()
            in_ten   = cfg.ten_start_et <= bar_time < cfg.ten_end_et

            # ── Check TP/SL ──────────────────────────────────────────
            if in_trade:
                hit_tp = hit_sl = False
                if t_dir == "long":
                    if bar.low <= t_stop:
                        hit_sl = True
                    elif bar.high >= t_target:
                        hit_tp = True
                else:
                    if bar.high >= t_stop:
                        hit_sl = True
                    elif bar.low <= t_target:
                        hit_tp = True

                if hit_tp or hit_sl:
                    exit_px = t_target if hit_tp else t_stop
                    reason  = "target" if hit_tp else "stop"
                    pts = (exit_px - t_entry if t_dir == "long"
                           else t_entry - exit_px)
                    trades.append(NineAmTrade(
                        date=day, entry_time=t_ts, exit_time=bar.timestamp,
                        direction=t_dir, entry_price=t_entry,
                        stop_price=t_stop, target_price=t_target,
                        exit_price=exit_px, exit_reason=reason,
                        pnl_usd=round(pts * dpp, 2),
                        nine_high=nine_h, nine_low=nine_l,
                        nine_open=nine_o, nine_close=nine_c,
                    ))
                    in_trade = False

            # ── Session end — close any open trade ───────────────────
            if in_trade and bar_time >= cfg.session_end_et:
                pts = (bar.close - t_entry if t_dir == "long"
                       else t_entry - bar.close)
                trades.append(NineAmTrade(
                    date=day, entry_time=t_ts, exit_time=bar.timestamp,
                    direction=t_dir, entry_price=t_entry,
                    stop_price=t_stop, target_price=t_target,
                    exit_price=bar.close, exit_reason="session_end",
                    pnl_usd=round(pts * dpp, 2),
                    nine_high=nine_h, nine_low=nine_l,
                    nine_open=nine_o, nine_close=nine_c,
                ))
                in_trade = False

            # ── Entry: only during 10AM hour ─────────────────────────
            if (not in_trade and in_ten
                    and day_trades < cfg.max_trades_per_day):

                # Long breakout: bar closes above 9AM high
                if not took_long and bar.close > nine_h:
                    entry_px = bar.close + slip
                    if cfg.stop_mode == "mid":
                        stop_px = nine_mid
                    else:
                        stop_px = nine_l
                    risk = entry_px - stop_px
                    if risk > 0:
                        target_px = entry_px + risk * cfg.target_rr
                        t_dir    = "long"
                        t_entry  = entry_px
                        t_stop   = stop_px
                        t_target = target_px
                        t_ts     = bar.timestamp
                        in_trade = True
                        took_long = True
                        day_trades += 1

                # Short breakout: bar closes below 9AM low
                elif not took_short and bar.close < nine_l:
                    entry_px = bar.close - slip
                    if cfg.stop_mode == "mid":
                        stop_px = nine_mid
                    else:
                        stop_px = nine_h
                    risk = stop_px - entry_px
                    if risk > 0:
                        target_px = entry_px - risk * cfg.target_rr
                        t_dir    = "short"
                        t_entry  = entry_px
                        t_stop   = stop_px
                        t_target = target_px
                        t_ts     = bar.timestamp
                        in_trade = True
                        took_short = True
                        day_trades += 1

        # Close any open trade at end of data
        if in_trade and action_bars:
            last = action_bars[-1]
            pts = (last.close - t_entry if t_dir == "long"
                   else t_entry - last.close)
            trades.append(NineAmTrade(
                date=day, entry_time=t_ts, exit_time=last.timestamp,
                direction=t_dir, entry_price=t_entry,
                stop_price=t_stop, target_price=t_target,
                exit_price=last.close, exit_reason="data_end",
                pnl_usd=round(pts * dpp, 2),
                nine_high=nine_h, nine_low=nine_l,
                nine_open=nine_o, nine_close=nine_c,
            ))

    return trades


def print_nine_am_summary(trades: list[NineAmTrade], cfg: NineAmConfig) -> None:
    if not trades:
        print("No trades generated.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
    pnl = sum(t.pnl_usd for t in trades)
    wr  = len(wins) / len(trades) * 100

    eq = 0.0; peak = 0.0; max_dd = 0.0
    for t in trades:
        eq   += t.pnl_usd
        peak  = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    dates = sorted({t.date for t in trades})

    print(f"\n{'='*72}")
    print(f"  9AM BREAKOUT STRATEGY  |  stop={cfg.stop_mode}  RR={cfg.target_rr}  "
          f"|  {cfg.contracts} contracts")
    print(f"  {dates[0]}  →  {dates[-1]}  |  {len(dates)} trading days")
    print(f"{'='*72}")

    print(f"\n── OVERALL ──────────────────────────────────────────────────────")
    print(f"  Total trades  : {len(trades)}")
    print(f"  Wins / Losses : {len(wins)} / {len(losses)}")
    print(f"  Win rate      : {wr:.1f}%")
    print(f"  Total PnL     : ${pnl:>10,.2f}")
    print(f"  Profit factor : {pf_s}")
    avg_w = gw / len(wins) if wins else 0.0
    avg_l = gl / len(losses) if losses else 0.0
    print(f"  Avg win       : ${avg_w:>10,.2f}")
    print(f"  Avg loss      : ${avg_l:>10,.2f}")
    print(f"  Max drawdown  : ${max_dd:>10,.2f}")

    longs  = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    print(f"\n── LONG / SHORT ─────────────────────────────────────────────────")
    for label, grp in [("Long", longs), ("Short", shorts)]:
        if not grp:
            print(f"  {label:<8}  (no trades)")
            continue
        w = [t for t in grp if t.pnl_usd > 0]
        l_ = [t for t in grp if t.pnl_usd <= 0]
        g_w = sum(t.pnl_usd for t in w)
        g_l = abs(sum(t.pnl_usd for t in l_))
        g_pf = g_w / g_l if g_l > 0 else float("inf")
        g_pf_s = f"{g_pf:.2f}" if g_pf != float("inf") else "∞"
        g_pnl = sum(t.pnl_usd for t in grp)
        g_wr  = len(w) / len(grp) * 100
        print(f"  {label:<8}  N={len(grp):>4}  {len(w)}W/{len(l_)}L  "
              f"WR={g_wr:>5.1f}%  PnL=${g_pnl:>10,.2f}  PF={g_pf_s}")

    from collections import defaultdict as dd
    print(f"\n── BY EXIT REASON ───────────────────────────────────────────────")
    by_reason: dict[str, list] = dd(list)
    for t in trades:
        by_reason[t.exit_reason].append(t)
    for reason in sorted(by_reason):
        grp = by_reason[reason]
        g_pnl = sum(t.pnl_usd for t in grp)
        w = sum(1 for t in grp if t.pnl_usd > 0)
        print(f"  {reason:<14}  {len(grp):>4} trades  {w}W/{len(grp)-w}L  "
              f"${g_pnl:>+10,.2f}")

    show = trades[-30:]
    if len(trades) > 30:
        print(f"\n── LAST 30 TRADES (of {len(trades)}) ─────────────────────────────────")
    else:
        print(f"\n── TRADE LOG ────────────────────────────────────────────────────")
    for t in show:
        et = t.entry_time.astimezone(_ET).strftime("%H:%M") if t.entry_time.tzinfo else "?"
        xt = t.exit_time.astimezone(_ET).strftime("%H:%M") if t.exit_time.tzinfo else "?"
        icon = "✓" if t.pnl_usd > 0 else "✗"
        rng = t.nine_high - t.nine_low
        print(f"  {t.date}  {t.direction.upper():<5}  "
              f"ep={t.entry_price:.2f}  sl={t.stop_price:.2f}  tp={t.target_price:.2f}  "
              f"exit={t.exit_price:.2f}  ({t.exit_reason})  "
              f"9H/L={t.nine_high:.0f}/{t.nine_low:.0f}({rng:.0f}r)  "
              f"{icon} ${t.pnl_usd:>+8,.2f}  {et}→{xt}")
    print()
