#!/usr/bin/env python3
"""
POI Strategy Backtest

For each trading day:
  1. Generate POI levels from the prior 5 days of bars (same as daily_prep)
  2. Replay that day's session bars through the strategy state machine
  3. Report what happened at each level

Usage:
    python3 backtest_poi.py                  # MES (default)
    python3 backtest_poi.py --symbol MNQ
"""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent.indicators import MinuteBarAccumulator, _EMA
from trading_agent.live_runner import POILiveSession          # for _estimate_daily_atr
from trading_agent.poi_scanner import scan_poi
from trading_agent.poi_strategy import (
    POIStrategyConfig, POIZoneState, _POIState,
    _at_zone, _macd_aligned, _ema_aligned,
    _entry_direction, _clean_close, _next_poi_target, _in_window,
)
from trading_agent.projectx_client import Bar

ET = ZoneInfo("America/New_York")

# ── Instrument specs (tick size / $ per point) ───────────────────────────────

INSTRUMENT_SPECS = {
    "MES": dict(tick_size=0.25, dollars_per_point=5.0),
    "MNQ": dict(tick_size=0.25, dollars_per_point=2.0),
    "ES":  dict(tick_size=0.25, dollars_per_point=50.0),
    "NQ":  dict(tick_size=0.25, dollars_per_point=20.0),
}


def build_cfg(symbol: str, contracts: int = 10) -> POIStrategyConfig:
    spec = INSTRUMENT_SPECS.get(symbol.upper(), INSTRUMENT_SPECS["MES"])
    return POIStrategyConfig(
        zone_buffer_ticks        = 8,
        max_zone_bars            = 20,
        require_macd_slope_align = True,
        require_ema_slope_align  = False,
        entry_clean_close_ticks  = 8,
        stop_atr_pct             = 0.25,
        use_atr_stop             = True,
        target_r_multiple        = 1.0,
        contracts                = contracts,
        slippage_ticks           = 1,
        max_trades_per_day       = 3,
        daily_loss_limit_usd     = -500.0,
        daily_profit_limit_usd   = 1500.0,
        trade_start_et           = dtime(9, 30),
        entry_cutoff_et          = dtime(15, 30),
        trade_end_et             = dtime(16, 0),
        tick_size                = spec["tick_size"],
        dollars_per_point        = spec["dollars_per_point"],
    )


CFG = build_cfg("MES")


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class DayResult:
    session_date: date
    daily_atr:    float
    pois:         list   # list of POI
    touches:      list   # list of TouchResult
    trades:       list   # list of TradeResult
    day_pnl:      float

@dataclass
class TouchResult:
    poi_rank:  int
    poi_price: float
    poi_label: str
    touched:   bool
    touch_time: Optional[datetime]
    peak_move:  float   # max favourable pts from zone after touch
    outcome:    str     # "no_touch" | "touched_no_setup" | "setup_no_entry" | "entered"

@dataclass
class TradeResult:
    poi_rank:    int
    poi_price:   float
    poi_label:   str
    direction:   str
    entry_time:  datetime
    entry_px:    float
    stop_px:     float
    target_px:   float
    exit_time:   Optional[datetime]
    exit_px:     float
    exit_reason: str     # "target" | "stop" | "session_end"
    pnl_usd:     float
    pnl_r:       float


# ── Bar loading ───────────────────────────────────────────────────────────────

def load_bars(paths: list[str]) -> list[Bar]:
    dfs = [pd.read_excel(p) for p in paths]
    df  = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    bars = []
    for _, row in df.iterrows():
        ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bars.append(Bar(
            timestamp = ts,
            open      = float(row["open"]),
            high      = float(row["high"]),
            low       = float(row["low"]),
            close     = float(row["close"]),
            volume    = int(row["volume"]),
        ))
    return bars


def session_date_of(bar: Bar) -> date:
    """Globex session date: bars at/after 5pm ET belong to the next calendar day."""
    ts_et = bar.timestamp.astimezone(ET)
    d = ts_et.date()
    if ts_et.hour >= 17:
        d += timedelta(days=1)
    return d


def split_by_session(bars: list[Bar]) -> dict[date, list[Bar]]:
    buckets: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        buckets[session_date_of(b)].append(b)
    return dict(sorted(buckets.items()))


# ── ATR (mirrors live_runner._estimate_daily_atr) ────────────────────────────

def estimate_atr(bars: list[Bar], period: int = 14) -> float:
    buckets: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        buckets[session_date_of(b)].append(b)

    daily: list[Bar] = []
    for d in sorted(buckets):
        bs = buckets[d]
        if len(bs) < 10:
            continue
        daily.append(Bar(
            timestamp = bs[0].timestamp,
            open      = bs[0].open,
            high      = max(b.high for b in bs),
            low       = min(b.low  for b in bs),
            close     = bs[-1].close,
            volume    = sum(b.volume for b in bs),
        ))

    if len(daily) < 2:
        return 0.0
    trs = [
        max(daily[i].high - daily[i].low,
            abs(daily[i].high - daily[i-1].close),
            abs(daily[i].low  - daily[i-1].close))
        for i in range(1, len(daily))
    ]
    atr = sum(trs[:period]) / period if len(trs) >= period else sum(trs) / max(len(trs), 1)
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 2)


# ── Core session simulation ───────────────────────────────────────────────────

def run_session(session_date: date, session_bars: list[Bar],
                prev_bars: list[Bar],
                use_macd_crossover: bool = False) -> Optional[DayResult]:
    if len(prev_bars) < 200 or not session_bars:
        return None

    daily_atr = estimate_atr(prev_bars)
    if daily_atr == 0:
        return None

    # Scan POIs — pass prev_bars newest-first (as scan_poi expects from API)
    prev_rev = list(reversed(prev_bars))
    current_price = prev_rev[0].close
    scan_time = session_bars[0].timestamp.astimezone(ET)

    try:
        pois, levels = scan_poi(
            bars          = prev_rev,
            scan_time     = scan_time,
            current_price = current_price,
            daily_atr     = daily_atr,
        )
    except Exception as exc:
        print(f"  [{session_date}] scan_poi failed: {exc}")
        return None

    if not pois:
        return None

    # ── Indicators ───────────────────────────────────────────────────────────
    ema1m      = _EMA(CFG.ema_length)
    ema_hist   = deque(maxlen=CFG.ema_slope_lookback_bars + 1)
    acc5m      = MinuteBarAccumulator(5)
    macd_fast  = _EMA(CFG.macd_fast_length)
    macd_slow  = _EMA(CFG.macd_slow_length)
    macd_sig   = _EMA(CFG.macd_signal_length)
    macd_hist  = deque(maxlen=CFG.macd_slope_lookback_bars + 1)

    cur_ema: Optional[float]     = None
    cur_ema_sl: Optional[float]  = None
    cur_macd_sl: Optional[float] = None   # slope mode
    cur_macd_ln: Optional[float] = None   # crossover mode: MACD line
    cur_macd_sg: Optional[float] = None   # crossover mode: signal line

    poi_states = [_POIState(poi=p) for p in pois]

    # ── Trade state ───────────────────────────────────────────────────────────
    in_trade   = False
    trade_dir: Optional[str]   = None
    trade_stop: Optional[float] = None
    trade_tgt:  Optional[float] = None
    active_ps: Optional[_POIState] = None
    entry_px:   float = 0.0

    day_pnl    = 0.0
    day_trades = 0
    locked     = False
    trades:  list[TradeResult]  = []
    touches: dict[int, TouchResult] = {
        p.rank: TouchResult(
            poi_rank=p.rank, poi_price=p.price, poi_label=p.label,
            touched=False, touch_time=None, peak_move=0.0,
            outcome="no_touch",
        )
        for p in pois
    }

    tick  = CFG.tick_size
    slip  = CFG.slippage_ticks * tick
    cbuf  = CFG.entry_clean_close_ticks * tick
    zbuf  = CFG.zone_buffer_ticks * tick

    # ── Replay bars ───────────────────────────────────────────────────────────
    for bar in session_bars:
        ts_et    = bar.timestamp.astimezone(ET)
        bar_time = ts_et.time()

        # ── Check open trade exit on this bar ─────────────────────────────
        if in_trade and active_ps is not None:
            risk_pts = abs(entry_px - trade_stop)

            if trade_dir == "long":
                if bar.high >= trade_tgt:
                    ep = trade_tgt
                    pnl = (ep - entry_px) * CFG.dollars_per_point * CFG.contracts
                    r   = (ep - entry_px) / risk_pts if risk_pts else 0
                    _close_trade(active_ps, bar.timestamp, ep, "target", pnl, r,
                                 entry_px, trade_stop, trade_tgt, trades, touches)
                    day_pnl += pnl; day_trades += 1
                    in_trade = False; active_ps = None
                elif bar.low <= trade_stop:
                    ep = trade_stop
                    pnl = (ep - entry_px) * CFG.dollars_per_point * CFG.contracts
                    r   = (ep - entry_px) / risk_pts if risk_pts else 0
                    _close_trade(active_ps, bar.timestamp, ep, "stop", pnl, r,
                                 entry_px, trade_stop, trade_tgt, trades, touches)
                    day_pnl += pnl; day_trades += 1
                    in_trade = False; active_ps = None

            elif trade_dir == "short":
                if bar.low <= trade_tgt:
                    ep = trade_tgt
                    pnl = (entry_px - ep) * CFG.dollars_per_point * CFG.contracts
                    r   = (entry_px - ep) / risk_pts if risk_pts else 0
                    _close_trade(active_ps, bar.timestamp, ep, "target", pnl, r,
                                 entry_px, trade_stop, trade_tgt, trades, touches)
                    day_pnl += pnl; day_trades += 1
                    in_trade = False; active_ps = None
                elif bar.high >= trade_stop:
                    ep = trade_stop
                    pnl = (entry_px - ep) * CFG.dollars_per_point * CFG.contracts
                    r   = (entry_px - ep) / risk_pts if risk_pts else 0
                    _close_trade(active_ps, bar.timestamp, ep, "stop", pnl, r,
                                 entry_px, trade_stop, trade_tgt, trades, touches)
                    day_pnl += pnl; day_trades += 1
                    in_trade = False; active_ps = None

        # ── Update indicators ─────────────────────────────────────────────
        completed_5m = acc5m.update(bar)
        if completed_5m is not None:
            fv = macd_fast.update(completed_5m.close)
            sv = macd_slow.update(completed_5m.close)
            if fv is not None and sv is not None:
                ml  = fv - sv
                sgv = macd_sig.update(ml)
                if sgv is not None:
                    cur_macd_ln = ml
                    cur_macd_sg = sgv
                    macd_hist.append(ml)
                    if len(macd_hist) >= CFG.macd_slope_lookback_bars + 1:
                        cur_macd_sl = macd_hist[-1] - macd_hist[0]

            # Update zone states on 5m bar close
            if _in_window(bar_time, CFG.trade_start_et, CFG.trade_end_et):
                macd_val = _macd_crossover_val(cur_macd_ln, cur_macd_sg) \
                           if use_macd_crossover else cur_macd_sl
                _update_zone_states(poi_states, completed_5m, bar.timestamp,
                                    macd_val, cur_ema_sl, zbuf, CFG, touches)

        ev = ema1m.update(bar.close)
        if ev is not None:
            cur_ema = ev
            ema_hist.append(ev)
            if len(ema_hist) >= CFG.ema_slope_lookback_bars + 1:
                cur_ema_sl = ema_hist[-1] - ema_hist[0]

        if not _in_window(bar_time, CFG.trade_start_et, CFG.trade_end_et):
            continue

        # ── Daily limits ─────────────────────────────────────────────────
        if (day_pnl <= CFG.daily_loss_limit_usd
                or day_pnl >= CFG.daily_profit_limit_usd
                or day_trades >= CFG.max_trades_per_day):
            locked = True

        can_enter  = _in_window(bar_time, CFG.trade_start_et, CFG.entry_cutoff_et)
        macd_ready = (cur_macd_ln is not None and cur_macd_sg is not None) \
                     if use_macd_crossover else cur_macd_sl is not None
        if in_trade or locked or cur_ema is None or not macd_ready or not can_enter:
            continue

        macd_val = _macd_crossover_val(cur_macd_ln, cur_macd_sg) \
                   if use_macd_crossover else cur_macd_sl

        # ── Check entry ───────────────────────────────────────────────────
        for ps in poi_states:
            if ps.state != POIZoneState.SETUP_FORMING:
                continue

            direction = _entry_direction(ps.poi.bias, bar, ps.poi.price, macd_val)
            if direction is None:
                continue
            if direction == "long"  and not CFG.allow_longs:
                continue
            if direction == "short" and not CFG.allow_shorts:
                continue

            macd_ok = _macd_aligned(ps.poi.bias, macd_val, CFG.macd_slope_threshold)
            ema_ok  = _ema_aligned(ps.poi.bias, cur_ema_sl, CFG.ema_slope_threshold) \
                      if CFG.require_ema_slope_align else True
            if not (macd_ok and ema_ok):
                continue
            if not _clean_close(direction, bar, ps.poi.price, cbuf):
                continue

            # Build bracket — cap stop at 25 points max
            MAX_STOP_PTS = 25.0
            ep       = bar.close + slip if direction == "long" else bar.close - slip
            atr_dist = min(CFG.stop_atr_pct * daily_atr, MAX_STOP_PTS)
            stp      = ep - atr_dist if direction == "long" else ep + atr_dist
            risk_pts = abs(ep - stp)
            nxt      = _next_poi_target(direction, ep, poi_states)
            tgt      = (nxt if nxt is not None
                        else ep + risk_pts * CFG.target_r_multiple if direction == "long"
                        else ep - risk_pts * CFG.target_r_multiple)

            if direction == "long"  and tgt <= ep: continue
            if direction == "short" and tgt >= ep: continue

            ps.state   = POIZoneState.ENTRY_CONFIRMED
            in_trade   = True
            trade_dir  = direction
            trade_stop = stp
            trade_tgt  = tgt
            active_ps  = ps
            entry_px   = ep

            # Store entry info on the _POIState for close reporting
            ps._entry_time = bar.timestamp
            ps._entry_px   = ep
            ps._stop_px    = stp
            ps._target_px  = tgt

            touches[ps.poi.rank].outcome = "entered"
            break

    # ── Close any open trade at session end ───────────────────────────────
    if in_trade and active_ps is not None and session_bars:
        last = session_bars[-1]
        ep   = last.close
        risk_pts = abs(entry_px - trade_stop)
        pnl = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
              * CFG.dollars_per_point * CFG.contracts
        r   = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
              / risk_pts if risk_pts else 0
        _close_trade(active_ps, last.timestamp, ep, "session_end", pnl, r,
                     entry_px, trade_stop, trade_tgt, trades, touches)
        day_pnl += pnl

    return DayResult(
        session_date = session_date,
        daily_atr    = daily_atr,
        pois         = pois,
        touches      = list(touches.values()),
        trades       = trades,
        day_pnl      = day_pnl,
    )


def _macd_crossover_val(macd_ln: Optional[float], macd_sg: Optional[float]) -> Optional[float]:
    """Return a pseudo-slope from MACD crossover position.
    Positive when MACD line is above signal (bullish), negative when below."""
    if macd_ln is None or macd_sg is None:
        return None
    return macd_ln - macd_sg


def _update_zone_states(poi_states, bar5m, bar_ts, cur_macd_sl, cur_ema_sl,
                        zbuf, cfg, touches):
    for ps in poi_states:
        if ps.state in (POIZoneState.USED, POIZoneState.EXPIRED,
                        POIZoneState.ENTRY_CONFIRMED):
            continue

        at_z = _at_zone(bar5m, ps.poi.price, zbuf)
        rank = ps.poi.rank

        if ps.state == POIZoneState.WATCHING:
            if at_z:
                ps.zone_bars    = 1
                ps.touched_time = bar_ts
                ps.state        = POIZoneState.ZONE_TOUCHED
                touches[rank].touched    = True
                touches[rank].touch_time = bar_ts
                if touches[rank].outcome == "no_touch":
                    touches[rank].outcome = "touched_no_setup"

        elif ps.state == POIZoneState.ZONE_TOUCHED:
            if not at_z:
                ps.state = POIZoneState.WATCHING
                continue
            ps.zone_bars += 1
            macd_ok = _macd_aligned(ps.poi.bias, cur_macd_sl, cfg.macd_slope_threshold)
            ema_ok  = _ema_aligned(ps.poi.bias, cur_ema_sl,  cfg.ema_slope_threshold)
            if macd_ok or ema_ok:
                ps.state = POIZoneState.SETUP_FORMING
                if touches[rank].outcome == "touched_no_setup":
                    touches[rank].outcome = "setup_no_entry"
            if ps.zone_bars > cfg.max_zone_bars:
                ps.state = POIZoneState.EXPIRED

        elif ps.state == POIZoneState.SETUP_FORMING:
            if not at_z:
                ps.state = POIZoneState.WATCHING
                continue
            ps.zone_bars += 1
            if ps.zone_bars > cfg.max_zone_bars:
                ps.state = POIZoneState.EXPIRED


def _close_trade(ps, exit_time, exit_px, reason, pnl, pnl_r,
                 entry_px, stop_px, target_px, trades, touches):
    ps.state = POIZoneState.USED
    trades.append(TradeResult(
        poi_rank    = ps.poi.rank,
        poi_price   = ps.poi.price,
        poi_label   = ps.poi.label,
        direction   = getattr(ps, "_entry_time", None) and "?",
        entry_time  = getattr(ps, "_entry_time", exit_time),
        entry_px    = entry_px,
        stop_px     = stop_px,
        target_px   = target_px,
        exit_time   = exit_time,
        exit_px     = exit_px,
        exit_reason = reason,
        pnl_usd     = pnl,
        pnl_r       = pnl_r,
    ))


# ── Report ────────────────────────────────────────────────────────────────────

def print_day(result: DayResult) -> None:
    pnl_str = f"${result.day_pnl:+,.2f}"
    print(f"\n{'═'*68}")
    print(f"  {result.session_date}   ATR={result.daily_atr:.2f}   Day PnL: {pnl_str}")
    print(f"{'─'*68}")

    for poi in result.pois:
        touch = next((t for t in result.touches if t.poi_rank == poi.rank), None)
        trade = next((t for t in result.trades  if t.poi_rank == poi.rank), None)

        outcome_icon = {
            "no_touch":       "  ○",
            "touched_no_setup": "  ~",
            "setup_no_entry": "  ≈",
            "entered":        "  →",
        }.get(touch.outcome if touch else "no_touch", "  ?")

        print(f"{outcome_icon}  POI#{poi.rank} @ {poi.price:>9.2f}  [{poi.label}]")

        if touch and touch.touched:
            tt = touch.touch_time.astimezone(ET).strftime("%H:%M ET") if touch.touch_time else "?"
            print(f"       Touched {tt}")

        if trade:
            result_icon = "✓ WIN " if trade.pnl_usd > 0 else "✗ LOSS"
            entry_t = trade.entry_time.astimezone(ET).strftime("%H:%M ET")
            exit_t  = trade.exit_time.astimezone(ET).strftime("%H:%M ET") if trade.exit_time else "?"
            print(f"       {result_icon}  entry={trade.entry_px:.2f}  stop={trade.stop_px:.2f}"
                  f"  target={trade.target_px:.2f}")
            print(f"       exit={trade.exit_px:.2f} ({trade.exit_reason})  "
                  f"PnL=${trade.pnl_usd:+.2f}  {trade.pnl_r:+.2f}R  "
                  f"{entry_t}→{exit_t}")


def print_summary(results: list[DayResult], symbol: str = "MES",
                   start_balance: float = 0.0) -> None:
    all_trades = [t for r in results for t in r.trades]
    if not all_trades:
        print("\nNo trades generated.")
        return

    wins   = [t for t in all_trades if t.pnl_usd > 0]
    losses = [t for t in all_trades if t.pnl_usd <= 0]
    total_pnl = sum(t.pnl_usd for t in all_trades)
    win_rate  = len(wins) / len(all_trades) * 100

    # Equity curve + drawdown
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    max_dd_start = max_dd_end = None
    peak_time = None
    equity_curve = []
    for t in all_trades:
        equity += t.pnl_usd
        equity_curve.append(equity)
        if equity > peak:
            peak      = equity
            peak_time = t.exit_time
        dd = peak - equity
        if dd > max_dd:
            max_dd       = dd
            max_dd_start = peak_time
            max_dd_end   = t.exit_time

    print(f"\n{'═'*68}")
    print(f"  BACKTEST SUMMARY — {symbol} POI Strategy")
    print(f"{'═'*68}")
    if start_balance > 0:
        end_balance = start_balance + total_pnl
        print(f"  Starting balance: ${start_balance:>10,.2f}")
        print(f"  Ending balance  : ${end_balance:>10,.2f}  ({total_pnl / start_balance * 100:+.1f}%)")
    print(f"  Sessions tested : {len(results)}")
    print(f"  Total trades    : {len(all_trades)}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Win rate        : {win_rate:.1f}%")
    print(f"  Total PnL       : ${total_pnl:>10,.2f}")
    if wins:
        print(f"  Avg win         : ${sum(t.pnl_usd for t in wins)/len(wins):>10,.2f}")
    if losses:
        print(f"  Avg loss        : ${sum(t.pnl_usd for t in losses)/len(losses):>10,.2f}")
    gross_win  = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Profit factor   : {pf:.2f}")
    dd_start = max_dd_start.astimezone(ET).strftime("%b %d") if max_dd_start else "?"
    dd_end   = max_dd_end.astimezone(ET).strftime("%b %d")   if max_dd_end   else "?"
    print(f"  Max drawdown    : ${max_dd:>10,.2f}  ({dd_start} → {dd_end})")
    if start_balance > 0 and max_dd > 0:
        print(f"                    ({max_dd / start_balance * 100:.1f}% of starting balance)")

    # Equity curve per day
    print(f"\n  Equity curve (cumulative):")
    running = 0.0
    for r in results:
        running += r.day_pnl
        bar_len = int(abs(running) / (total_pnl / 30)) if total_pnl else 0
        bar = ("█" * bar_len) if running >= 0 else ("░" * bar_len)
        bal_str = f"  bal=${start_balance + running:>9,.2f}" if start_balance > 0 else ""
        print(f"    {r.session_date}  {bar:<32}  ${running:>+10,.2f}{bal_str}")

    # Exit reason breakdown
    by_reason = defaultdict(list)
    for t in all_trades:
        by_reason[t.exit_reason].append(t.pnl_usd)
    print(f"\n  By exit reason:")
    for reason, pnls in sorted(by_reason.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<14} {len(pnls):>3} trades  {w}W/{len(pnls)-w}L"
              f"  ${sum(pnls):>+10,.2f}")


def save_excel(results: list[DayResult], path: str = "data/poi_backtest.xlsx") -> None:
    rows = []
    for r in results:
        for t in r.trades:
            entry_t = t.entry_time.astimezone(ET) if t.entry_time else None
            exit_t  = t.exit_time.astimezone(ET)  if t.exit_time  else None
            rows.append({
                "date":        r.session_date,
                "atr":         r.daily_atr,
                "poi_rank":    t.poi_rank,
                "poi_price":   t.poi_price,
                "poi_label":   t.poi_label,
                "entry_time":  entry_t.strftime("%H:%M") if entry_t else "",
                "entry_px":    t.entry_px,
                "stop_px":     t.stop_px,
                "target_px":   t.target_px,
                "exit_time":   exit_t.strftime("%H:%M")  if exit_t  else "",
                "exit_px":     t.exit_px,
                "exit_reason": t.exit_reason,
                "pnl_usd":     round(t.pnl_usd, 2),
                "pnl_r":       round(t.pnl_r, 2),
            })
    if rows:
        pd.DataFrame(rows).to_excel(path, index=False)
        print(f"\n  Trades saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global CFG

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MES", help="Symbol prefix, e.g. MES, MNQ")
    ap.add_argument("--contracts", type=int, default=10, help="Contracts per trade")
    ap.add_argument("--start-balance", type=float, default=0.0, dest="start_balance",
                    help="Starting account balance USD (0 = show raw PnL only)")
    args = ap.parse_args()
    symbol = args.symbol.upper()
    CFG = build_cfg(symbol, contracts=args.contracts)

    data_dir = Path("data")
    paths = sorted(data_dir.glob(f"{symbol}_2026-*_1m.xlsx"))
    if not paths:
        print(f"No data files found for {symbol}. Run:  python -m trading_agent download --symbol {symbol} --continuous --excel ...")
        return

    print(f"Loading {len(paths)} file(s): {[p.name for p in paths]}")
    bars = load_bars([str(p) for p in paths])
    print(f"Loaded {len(bars):,} bars")

    sessions = split_by_session(bars)
    session_dates = sorted(sessions.keys())
    print(f"Found {len(session_dates)} sessions: {session_dates[0]} → {session_dates[-1]}")

    all_sessions = []
    for i, sd in enumerate(session_dates):
        if i < 5:
            continue
        prior_dates  = session_dates[max(0, i-5):i]
        prior_bars   = [b for d in prior_dates for b in sessions[d]]
        all_sessions.append((sd, sessions[sd], prior_bars))

    all_results = {}
    for label, use_crossover in [("SLOPE", False), ("CROSSOVER", True)]:
        print(f"\n\n{'█'*68}")
        print(f"  MODE: MACD {label}")
        print(f"{'█'*68}")
        results = []
        for sd, session_bars, prior_bars in all_sessions:
            result = run_session(sd, session_bars, prior_bars,
                                 use_macd_crossover=use_crossover)
            if result:
                results.append(result)
                print_day(result)
        print_summary(results, symbol=symbol, start_balance=args.start_balance)
        all_results[label] = results

    # Side-by-side comparison
    print(f"\n\n{'═'*68}")
    print(f"  COMPARISON")
    print(f"{'═'*68}")
    for label, results in all_results.items():
        trades = [t for r in results for t in r.trades]
        wins   = [t for t in trades if t.pnl_usd > 0]
        total  = sum(t.pnl_usd for t in trades)
        gl     = abs(sum(t.pnl_usd for t in trades if t.pnl_usd <= 0))
        gw     = sum(t.pnl_usd for t in wins)
        pf     = gw / gl if gl else float("inf")
        wr     = len(wins) / len(trades) * 100 if trades else 0
        print(f"  {label:<12}  trades={len(trades):>2}  WR={wr:.1f}%  "
              f"PnL=${total:>+10,.2f}  PF={pf:.2f}")

    save_excel(all_results["CROSSOVER"], path=f"data/poi_backtest_{symbol.lower()}.xlsx")


if __name__ == "__main__":
    main()
