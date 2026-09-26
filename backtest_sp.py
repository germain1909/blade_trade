#!/usr/bin/env python3
"""
SP Futures POI backtest — CSV tick data

Loads tick data from data/SP.csv, builds 1-minute bars, and runs the
POI strategy. Since this is pit-session only data (no overnight), Asia
and London levels are skipped — POIs are built from PDH/PDL/PDC and ATR.

Usage:
    python3 backtest_sp.py                  # first month (Jan 2000)
    python3 backtest_sp.py 2000-01 2000-03  # Jan–Mar 2000
"""

import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from trading_agent.indicators import MinuteBarAccumulator, _EMA
from trading_agent.poi_strategy import (
    POIStrategyConfig, POIZoneState, _POIState,
    _at_zone, _macd_aligned, _entry_direction, _clean_close,
    _next_poi_target, _in_window,
)
from trading_agent.projectx_client import Bar

ET  = ZoneInfo("America/New_York")
CT  = ZoneInfo("America/Chicago")

# ── Config ────────────────────────────────────────────────────────────────────

CFG = POIStrategyConfig(
    zone_buffer_ticks        = 8,
    max_zone_bars            = 20,
    require_macd_slope_align = True,
    require_ema_slope_align  = False,
    entry_clean_close_ticks  = 8,
    stop_atr_pct             = 0.25,
    use_atr_stop             = True,
    target_r_multiple        = 1.0,
    contracts                = 1,
    slippage_ticks           = 1,
    dollars_per_point        = 250.0,   # SP big contract
    max_trades_per_day       = 3,
    daily_loss_limit_usd     = -2000.0,
    daily_profit_limit_usd   = 5000.0,
    trade_start_et           = dtime(9, 30),
    entry_cutoff_et          = dtime(15, 30),
    trade_end_et             = dtime(16, 15),
)

MAX_STOP_PTS = 25.0


# ── Tick → 1m bar aggregation ─────────────────────────────────────────────────

def load_bars_from_csv(path: str, start_ym: str, end_ym: str) -> list[Bar]:
    """
    Read SP.csv and aggregate ticks into 1-minute bars.
    Times in the CSV are CT; we store as UTC internally.

    start_ym / end_ym : "YYYY-MM" inclusive range.
    """
    start_month = datetime.strptime(start_ym, "%Y-%m").date().replace(day=1)
    end_month   = datetime.strptime(end_ym,   "%Y-%m").date().replace(day=28) + timedelta(days=4)
    end_month   = end_month.replace(day=1)   # first day of next month

    # First pass: collect ticks in range
    ticks: list[tuple[datetime, float, int]] = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            date_str, time_str, price_str = parts[0], parts[1], parts[2]
            vol = int(parts[3]) if len(parts) > 3 and parts[3].strip() else 1

            # Parse date to check range
            try:
                d = datetime.strptime(date_str, "%m/%d/%Y").date()
            except ValueError:
                continue
            if d < start_month or d >= end_month:
                if d >= end_month:
                    break
                continue

            # Parse full timestamp in CT → convert to UTC
            try:
                ts_str = f"{date_str} {time_str[:8]}"
                ts_ct  = datetime.strptime(ts_str, "%m/%d/%Y %H:%M:%S").replace(tzinfo=CT)
                ts_utc = ts_ct.astimezone(timezone.utc)
                price  = float(price_str)
            except (ValueError, OverflowError):
                continue

            ticks.append((ts_utc, price, max(vol, 1)))

    # Second pass: aggregate into 1m bars
    def floor_min(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0)

    bars: list[Bar] = []
    cur_min: Optional[datetime] = None
    o = h = l = c = 0.0
    vol_sum = 0

    for ts, price, vol in ticks:
        m = floor_min(ts)
        if cur_min is None:
            cur_min = m
            o = h = l = c = price
            vol_sum = vol
        elif m == cur_min:
            h = max(h, price)
            l = min(l, price)
            c = price
            vol_sum += vol
        else:
            bars.append(Bar(timestamp=cur_min, open=o, high=h, low=l, close=c, volume=vol_sum))
            cur_min = m
            o = h = l = c = price
            vol_sum = vol

    if cur_min is not None:
        bars.append(Bar(timestamp=cur_min, open=o, high=h, low=l, close=c, volume=vol_sum))

    return bars


# ── Session split ─────────────────────────────────────────────────────────────

def session_date_of(bar: Bar) -> date:
    ts_et = bar.timestamp.astimezone(ET)
    return ts_et.date()


def split_by_session(bars: list[Bar]) -> dict[date, list[Bar]]:
    buckets: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        buckets[session_date_of(b)].append(b)
    return dict(sorted(buckets.items()))


# ── ATR ───────────────────────────────────────────────────────────────────────

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


# ── Simple POI generator (no Asia/London — pit session only) ──────────────────

def generate_pois_pit(prev_bars: list[Bar], current_price: float,
                      daily_atr: float, session_date: date):
    """
    Build POI levels from PDH/PDL/PDC/PD-Mid and ATR levels only.
    Returns list of simple level dicts sorted by distance from current price.
    """
    from trading_agent.poi_scanner import POI, LevelRef

    if not prev_bars:
        return []

    buckets: dict[date, list[Bar]] = defaultdict(list)
    for b in prev_bars:
        buckets[session_date_of(b)].append(b)

    sorted_dates = sorted(buckets.keys())
    if not sorted_dates:
        return []

    prev_d   = sorted_dates[-1]
    prev_day = buckets[prev_d]
    pdh      = max(b.high  for b in prev_day)
    pdl      = min(b.low   for b in prev_day)
    pdc      = prev_day[-1].close
    pd_mid   = (pdh + pdl) / 2

    # Day open = first bar of session
    all_session_dates = sorted(set(session_date_of(b) for b in prev_bars))
    today_first = None
    for b in prev_bars:
        if session_date_of(b) == session_date:
            today_first = b.open
            break
    day_open = today_first or current_price

    raw_levels = {
        "PDH":    pdh,
        "PDL":    pdl,
        "PDC":    pdc,
        "PD Mid": pd_mid,
    }

    # ATR levels
    for pct_label, pct in [("-100%", -1.0), ("-75%", -0.75), ("-50%", -0.50),
                            ("-25%", -0.25), ("+25%", 0.25), ("+50%", 0.50),
                            ("+75%", 0.75), ("+100%", 1.0)]:
        raw_levels[f"ATR {pct_label}"] = round(day_open + pct * daily_atr, 2)

    # Find confluence clusters (tolerance = 3 ATR ticks)
    tol = 3 * CFG.tick_size
    used = set()
    clusters = []
    level_items = list(raw_levels.items())

    for i, (name_i, price_i) in enumerate(level_items):
        if i in used:
            continue
        group_names  = [name_i]
        group_prices = [price_i]
        for j, (name_j, price_j) in enumerate(level_items):
            if j <= i or j in used:
                continue
            if abs(price_i - price_j) <= tol:
                group_names.append(name_j)
                group_prices.append(price_j)
                used.add(j)
        used.add(i)
        avg_price = sum(group_prices) / len(group_prices)

        # Bias
        if avg_price > current_price:
            bias = "resistance"
        elif avg_price < pdl:
            bias = "support"
        else:
            bias = "pivot"

        clusters.append({
            "price":      round(avg_price, 2),
            "confluence": len(group_names),
            "names":      group_names,
            "bias":       bias,
        })

    clusters.sort(key=lambda c: (-c["confluence"], abs(c["price"] - current_price)))
    top = clusters[:3]

    pois = []
    for rank, c in enumerate(top, 1):
        pois.append(POI(
            rank             = rank,
            price            = c["price"],
            confluence_score = c["confluence"],
            bias             = c["bias"],
            above_price      = c["price"] > current_price,
            nearest_atr      = "",
            distance_pts     = abs(c["price"] - current_price),
            levels           = [LevelRef(n, raw_levels[n]) for n in c["names"]],
        ))
    return pois


# ── Session simulation (mirrors backtest_poi.run_session) ─────────────────────

@dataclass
class TradeResult:
    session_date: date
    direction:    str
    entry_time:   datetime
    entry_px:     float
    stop_px:      float
    target_px:    float
    exit_time:    Optional[datetime]
    exit_px:      float
    exit_reason:  str
    pnl_usd:      float
    pnl_r:        float


def run_session(session_date: date, session_bars: list[Bar],
                prev_bars: list[Bar]) -> list[TradeResult]:
    if len(prev_bars) < 100 or not session_bars:
        return []

    daily_atr = estimate_atr(prev_bars)
    if daily_atr == 0:
        return []

    current_price = session_bars[0].open
    pois = generate_pois_pit(prev_bars, current_price, daily_atr, session_date)
    if not pois:
        return []

    # ── Indicators ───────────────────────────────────────────────────────────
    ema1m     = _EMA(CFG.ema_length)
    ema_hist  = deque(maxlen=CFG.ema_slope_lookback_bars + 1)
    acc5m     = MinuteBarAccumulator(5)
    macd_fast = _EMA(CFG.macd_fast_length)
    macd_slow = _EMA(CFG.macd_slow_length)
    macd_sig  = _EMA(CFG.macd_signal_length)
    macd_hist = deque(maxlen=CFG.macd_slope_lookback_bars + 1)

    cur_ema_sl:  Optional[float] = None
    cur_macd_sl: Optional[float] = None

    poi_states = [_POIState(poi=p) for p in pois]

    in_trade   = False
    trade_dir: Optional[str]   = None
    trade_stop: Optional[float] = None
    trade_tgt:  Optional[float] = None
    active_ps: Optional[_POIState] = None
    entry_px   = 0.0
    entry_time: Optional[datetime] = None

    day_pnl    = 0.0
    day_trades = 0
    locked     = False
    trades: list[TradeResult] = []

    tick = CFG.tick_size
    slip = CFG.slippage_ticks * tick
    cbuf = CFG.entry_clean_close_ticks * tick
    zbuf = CFG.zone_buffer_ticks * tick

    for bar in session_bars:
        ts_et    = bar.timestamp.astimezone(ET)
        bar_time = ts_et.time()

        # ── Check open trade exit ─────────────────────────────────────────
        if in_trade and active_ps is not None:
            risk_pts = abs(entry_px - trade_stop)
            exited   = False

            if trade_dir == "long":
                if bar.high >= trade_tgt:
                    ep = trade_tgt; reason = "target"
                    pnl = (ep - entry_px) * CFG.dollars_per_point * CFG.contracts
                    r   = (ep - entry_px) / risk_pts if risk_pts else 0
                    exited = True
                elif bar.low <= trade_stop:
                    ep = trade_stop; reason = "stop"
                    pnl = (ep - entry_px) * CFG.dollars_per_point * CFG.contracts
                    r   = (ep - entry_px) / risk_pts if risk_pts else 0
                    exited = True
            else:
                if bar.low <= trade_tgt:
                    ep = trade_tgt; reason = "target"
                    pnl = (entry_px - ep) * CFG.dollars_per_point * CFG.contracts
                    r   = (entry_px - ep) / risk_pts if risk_pts else 0
                    exited = True
                elif bar.high >= trade_stop:
                    ep = trade_stop; reason = "stop"
                    pnl = (entry_px - ep) * CFG.dollars_per_point * CFG.contracts
                    r   = (entry_px - ep) / risk_pts if risk_pts else 0
                    exited = True

            if exited:
                active_ps.state = POIZoneState.USED
                trades.append(TradeResult(
                    session_date=session_date, direction=trade_dir,
                    entry_time=entry_time, entry_px=entry_px,
                    stop_px=trade_stop, target_px=trade_tgt,
                    exit_time=bar.timestamp, exit_px=ep,
                    exit_reason=reason, pnl_usd=pnl, pnl_r=r,
                ))
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
                    macd_hist.append(ml)
                    if len(macd_hist) >= CFG.macd_slope_lookback_bars + 1:
                        cur_macd_sl = macd_hist[-1] - macd_hist[0]

            if _in_window(bar_time, CFG.trade_start_et, CFG.trade_end_et):
                for ps in poi_states:
                    if ps.state in (POIZoneState.USED, POIZoneState.EXPIRED,
                                    POIZoneState.ENTRY_CONFIRMED):
                        continue
                    at_z = _at_zone(completed_5m, ps.poi.price, zbuf)
                    if ps.state == POIZoneState.WATCHING:
                        if at_z:
                            ps.zone_bars = 1
                            macd_ok = _macd_aligned(ps.poi.bias, cur_macd_sl,
                                                    CFG.macd_slope_threshold)
                            ps.state = (POIZoneState.SETUP_FORMING
                                        if macd_ok else POIZoneState.ZONE_TOUCHED)
                    elif ps.state == POIZoneState.ZONE_TOUCHED:
                        if not at_z:
                            ps.state = POIZoneState.WATCHING; continue
                        ps.zone_bars += 1
                        if _macd_aligned(ps.poi.bias, cur_macd_sl, CFG.macd_slope_threshold):
                            ps.state = POIZoneState.SETUP_FORMING
                        if ps.zone_bars > CFG.max_zone_bars:
                            ps.state = POIZoneState.EXPIRED
                    elif ps.state == POIZoneState.SETUP_FORMING:
                        if not at_z:
                            ps.state = POIZoneState.WATCHING; continue
                        ps.zone_bars += 1
                        if ps.zone_bars > CFG.max_zone_bars:
                            ps.state = POIZoneState.EXPIRED

        ev = ema1m.update(bar.close)
        if ev is not None:
            ema_hist.append(ev)
            if len(ema_hist) >= CFG.ema_slope_lookback_bars + 1:
                cur_ema_sl = ema_hist[-1] - ema_hist[0]

        if not _in_window(bar_time, CFG.trade_start_et, CFG.trade_end_et):
            continue

        if (day_pnl <= CFG.daily_loss_limit_usd
                or day_pnl >= CFG.daily_profit_limit_usd
                or day_trades >= CFG.max_trades_per_day):
            locked = True

        can_enter = _in_window(bar_time, CFG.trade_start_et, CFG.entry_cutoff_et)
        if in_trade or locked or cur_macd_sl is None or not can_enter:
            continue

        for ps in poi_states:
            if ps.state != POIZoneState.SETUP_FORMING:
                continue

            direction = _entry_direction(ps.poi.bias, bar, ps.poi.price, cur_macd_sl)
            if direction is None:
                continue

            macd_ok = _macd_aligned(ps.poi.bias, cur_macd_sl, CFG.macd_slope_threshold)
            if not macd_ok:
                continue
            if not _clean_close(direction, bar, ps.poi.price, cbuf):
                continue

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
            entry_time = bar.timestamp
            break

    # ── Close open trade at session end ───────────────────────────────────
    if in_trade and active_ps is not None and session_bars:
        last     = session_bars[-1]
        ep       = last.close
        risk_pts = abs(entry_px - trade_stop)
        pnl = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
              * CFG.dollars_per_point * CFG.contracts
        r   = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
              / risk_pts if risk_pts else 0
        active_ps.state = POIZoneState.USED
        trades.append(TradeResult(
            session_date=session_date, direction=trade_dir,
            entry_time=entry_time, entry_px=entry_px,
            stop_px=trade_stop, target_px=trade_tgt,
            exit_time=last.timestamp, exit_px=ep,
            exit_reason="session_end", pnl_usd=pnl, pnl_r=r,
        ))

    return trades


# ── Report ────────────────────────────────────────────────────────────────────

def print_summary(all_trades: list[TradeResult], label: str = "") -> None:
    if not all_trades:
        print("\nNo trades generated.")
        return

    wins   = [t for t in all_trades if t.pnl_usd > 0]
    losses = [t for t in all_trades if t.pnl_usd <= 0]
    total  = sum(t.pnl_usd for t in all_trades)
    wr     = len(wins) / len(all_trades) * 100

    equity = 0.0; peak = 0.0; max_dd = 0.0
    for t in all_trades:
        equity += t.pnl_usd
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)

    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl else float("inf")

    title = f"  SP FUTURES POI BACKTEST{' — ' + label if label else ''}"
    print(f"\n{'═'*68}")
    print(title)
    print(f"{'═'*68}")
    print(f"  Total trades : {len(all_trades)}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Win rate     : {wr:.1f}%")
    print(f"  Total PnL    : ${total:>10,.2f}")
    if wins:   print(f"  Avg win      : ${gw/len(wins):>10,.2f}")
    if losses: print(f"  Avg loss     : ${-gl/len(losses):>10,.2f}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Max drawdown : ${max_dd:>10,.2f}")

    print(f"\n  By exit reason:")
    from collections import defaultdict
    by_reason = defaultdict(list)
    for t in all_trades:
        by_reason[t.exit_reason].append(t.pnl_usd)
    for reason, pnls in sorted(by_reason.items()):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {reason:<14} {len(pnls):>3} trades  {w}W/{len(pnls)-w}L  "
              f"${sum(pnls):>+10,.2f}")

    print(f"\n  Trade log:")
    for t in all_trades:
        et = t.entry_time.astimezone(ET).strftime("%H:%M") if t.entry_time else "?"
        xt = t.exit_time.astimezone(ET).strftime("%H:%M") if t.exit_time else "?"
        icon = "✓" if t.pnl_usd > 0 else "✗"
        print(f"    {t.session_date}  {t.direction.upper():<5}  "
              f"entry={t.entry_px:.2f}  stop={t.stop_px:.2f}  tgt={t.target_px:.2f}  "
              f"exit={t.exit_px:.2f} ({t.exit_reason})  "
              f"{icon} ${t.pnl_usd:>+8,.2f}  {et}→{xt}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    start_ym = sys.argv[1] if len(sys.argv) > 1 else "2000-01"
    end_ym   = sys.argv[2] if len(sys.argv) > 2 else start_ym

    print(f"Loading SP.csv for {start_ym} → {end_ym} ...")
    bars = load_bars_from_csv("data/SP.csv", start_ym, end_ym)
    print(f"Built {len(bars):,} 1-minute bars")

    sessions = split_by_session(bars)
    dates    = sorted(sessions.keys())
    print(f"Sessions: {len(dates)}  ({dates[0]} → {dates[-1]})")

    all_trades: list[TradeResult] = []
    for i, sd in enumerate(dates):
        if i < 5:
            continue   # need prior sessions for ATR + levels
        prior_dates = dates[max(0, i-10):i]
        prior_bars  = [b for d in prior_dates for b in sessions[d]]
        trades = run_session(sd, sessions[sd], prior_bars)
        if trades:
            for t in trades:
                icon = "✓" if t.pnl_usd > 0 else "✗"
                et = t.entry_time.astimezone(ET).strftime("%H:%M") if t.entry_time else "?"
                print(f"  {sd}  {t.direction.upper():<5}  ep={t.entry_px:.2f}  "
                      f"sl={t.stop_px:.2f}  tp={t.target_px:.2f}  "
                      f"exit={t.exit_px:.2f}({t.exit_reason})  "
                      f"{icon} ${t.pnl_usd:>+,.2f}  @{et}")
        all_trades.extend(trades)

    print_summary(all_trades, label=f"{start_ym} → {end_ym}")


if __name__ == "__main__":
    main()
