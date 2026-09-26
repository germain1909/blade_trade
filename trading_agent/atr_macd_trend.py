from __future__ import annotations

"""
ATR MACD Trend Strategy
=======================
Simple trend-following strategy using 15-minute MACD for direction
and ATR percentage levels as pullback entry zones.

Trend  : 15M MACD (bull / bear / flat)
Entry  : 1M bar touches ATR level, pullback in trend direction
Target : adjacent ATR level in trend direction  (1:1)
Stop   : adjacent ATR level against trend        (1:1)
"""

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .indicators import MinuteBarAccumulator, _EMA, _WilderEMA
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

_ATR_PCTS: list[float] = [
    -1.50, -1.25, -1.00, -0.75, -0.50, -0.25,
     0.00, +0.25, +0.50, +0.75, +1.00, +1.25, +1.50,
]


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TrendState(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    FLAT = "FLAT"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ATRMTConfig:
    # ATR
    atr_period:             int   = 14

    # 15M MACD
    macd_fast_length:       int   = 12
    macd_slow_length:       int   = 26
    macd_signal_length:     int   = 9
    min_macd_separation:    float = 0.5   # below this = FLAT, no trades

    # Entry
    entry_buffer_ticks:     int   = 2     # ticks price must reach within ATR level
    slippage_ticks:         int   = 1
    stop_atr_pct:           float = 0.0   # stop as fraction of daily ATR from entry (0 = adjacent level)

    # Execution
    contracts:              int   = 3
    tick_size:              float = 0.25
    dollars_per_point:      float = 5.0

    # Session (ET).  Default = NY session.
    trade_start_et:         dtime = dtime(8, 30)
    entry_cutoff_et:        dtime = dtime(15, 0)
    trade_end_et:           dtime = dtime(16, 0)

    # Daily risk
    max_trades_per_day:     int   = 3
    daily_loss_limit_usd:   float = -600.0
    daily_profit_limit_usd: float = 1500.0

    # Toggles
    allow_longs:            bool  = True
    allow_shorts:           bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ATRMTTrade:
    date:             date
    entry_time:       datetime
    direction:        str
    atr_level_pct:    float
    atr_level_price:  float
    entry_price:      float
    stop_price:       float
    target_price:     float
    exit_time:        datetime
    exit_price:       float
    exit_reason:      str
    pnl_usd:          float
    pnl_r:            float
    risk_pts:         float
    daily_open:       float
    daily_atr:        float
    session:          str
    day_of_week:      str
    hour_of_entry:    int
    trend_at_entry:   str
    macd_line:        Optional[float]
    macd_signal:      Optional[float]
    macd_separation:  Optional[float]
    macd_slope:       Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


def _et_time(ts: datetime) -> dtime:
    return _et(ts).time()


def _et_date(ts: datetime) -> date:
    return _et(ts).date()


def _cme_day(ts: datetime) -> date:
    et = _et(ts)
    if et.time() >= dtime(18, 0):
        return (et + timedelta(days=1)).date()
    return et.date()


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    if start < end:
        return start <= t < end
    return t >= start or t < end


def _session(ts: datetime) -> str:
    t = _et_time(ts)
    if t >= dtime(18, 0) or t < dtime(3, 0):
        return "asia"
    if dtime(3, 0) <= t < dtime(8, 0):
        return "london"
    if dtime(8, 0) <= t < dtime(16, 0):
        return "ny"
    return "other"


def _dpp(cfg: ATRMTConfig) -> float:
    return cfg.dollars_per_point * cfg.contracts


# ─────────────────────────────────────────────────────────────────────────────
# Daily state builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _DailyState:
    trade_date: date
    daily_open: float
    daily_atr:  float
    levels:     list[float]


def _build_daily_states(bars: list[Bar]) -> list[_DailyState]:
    daily: dict[date, dict] = {}
    for bar in bars:
        d = _cme_day(bar.timestamp)
        if d not in daily:
            daily[d] = {"open": bar.open, "high": bar.high,
                        "low":  bar.low,  "close": bar.close}
        else:
            s = daily[d]
            s["high"]  = max(s["high"],  bar.high)
            s["low"]   = min(s["low"],   bar.low)
            s["close"] = bar.close

    sorted_dates = sorted(daily.keys())
    wilder     = _WilderEMA(14)
    prev_close: Optional[float] = None
    atr_vals:   dict[date, float] = {}

    for d in sorted_dates:
        s  = daily[d]
        hi, lo, cl = s["high"], s["low"], s["close"]
        tr = hi - lo if prev_close is None else max(
            hi - lo, abs(hi - prev_close), abs(lo - prev_close)
        )
        prev_close = cl
        val = wilder.update(tr)
        if val is not None:
            atr_vals[d] = val

    states = []
    for d in sorted_dates:
        if d not in atr_vals:
            continue
        s   = daily[d]
        atr = atr_vals[d]
        levels = [round(s["open"] + atr * p, 4) for p in _ATR_PCTS]
        states.append(_DailyState(trade_date=d, daily_open=s["open"],
                                  daily_atr=atr, levels=levels))
    return states


# ─────────────────────────────────────────────────────────────────────────────
# Core logic functions
# ─────────────────────────────────────────────────────────────────────────────

def getTrend(
    macd_line:    Optional[float],
    macd_signal:  Optional[float],
    macd_slope:   Optional[float],
    min_sep:      float,
) -> TrendState:
    """
    BULL : MACD Line > Signal Line  AND  slope > 0  AND  separation > min_sep
    BEAR : MACD Line < Signal Line  AND  slope < 0  AND  separation > min_sep
    FLAT : everything else
    """
    if any(v is None for v in (macd_line, macd_signal, macd_slope)):
        return TrendState.FLAT
    sep = abs(macd_line - macd_signal)  # type: ignore[operator]
    if sep < min_sep:
        return TrendState.FLAT
    if macd_line > macd_signal and macd_slope > 0:  # type: ignore[operator]
        return TrendState.BULL
    if macd_line < macd_signal and macd_slope < 0:  # type: ignore[operator]
        return TrendState.BEAR
    return TrendState.FLAT


def findTouchedLevel(
    bar:       Bar,
    levels:    list[float],
    direction: str,
    buffer:    float,
) -> Optional[tuple[float, float]]:
    """
    Find the ATR level this 1M bar pulled back to.

    Long  : bar.high >= level (was above) AND bar.low <= level + buffer (pulled back)
            → returns highest such level (most recent support)
    Short : bar.low <= level (was below) AND bar.high >= level - buffer (bounced up)
            → returns lowest such level (most recent resistance)
    """
    if direction == "long":
        candidates = [
            (lvl, pct) for lvl, pct in zip(levels, _ATR_PCTS)
            if bar.high >= lvl and bar.low <= lvl + buffer
        ]
        return max(candidates, key=lambda x: x[0]) if candidates else None
    else:
        candidates = [
            (lvl, pct) for lvl, pct in zip(levels, _ATR_PCTS)
            if bar.low <= lvl and bar.high >= lvl - buffer
        ]
        return min(candidates, key=lambda x: x[0]) if candidates else None


def getTargetAndStop(
    direction:  str,
    level_pct:  float,
    levels:     list[float],
) -> Optional[tuple[float, float]]:
    """
    Target = adjacent ATR level in trend direction.
    Stop   = adjacent ATR level against trend direction.
    Returns (target, stop) or None if no adjacent level exists.
    """
    try:
        idx = _ATR_PCTS.index(level_pct)
    except ValueError:
        return None
    if direction == "long":
        if idx + 1 >= len(levels) or idx - 1 < 0:
            return None
        return levels[idx + 1], levels[idx - 1]
    else:
        if idx - 1 < 0 or idx + 1 >= len(levels):
            return None
        return levels[idx - 1], levels[idx + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Position close helper
# ─────────────────────────────────────────────────────────────────────────────

def _close_position(
    p:       dict,
    exit_px: float,
    exit_ts: datetime,
    reason:  str,
    cfg:     ATRMTConfig,
) -> ATRMTTrade:
    direction = p["dir"]
    pnl_pts   = (exit_px - p["entry"]) if direction == "long" else (p["entry"] - exit_px)
    risk_pts  = abs(p["entry"] - p["stop"])
    pnl_usd   = pnl_pts * _dpp(cfg)
    pnl_r     = pnl_pts / risk_pts if risk_pts > 0 else 0.0
    et_entry  = _et(p["entry_time"])
    return ATRMTTrade(
        date            = _et_date(p["entry_time"]),
        entry_time      = p["entry_time"],
        direction       = direction,
        atr_level_pct   = p["level_pct"],
        atr_level_price = p["level_price"],
        entry_price     = p["entry"],
        stop_price      = p["stop"],
        target_price    = p["target"],
        exit_time       = exit_ts,
        exit_price      = exit_px,
        exit_reason     = reason,
        pnl_usd         = round(pnl_usd, 2),
        pnl_r           = round(pnl_r,   3),
        risk_pts        = round(risk_pts, 4),
        daily_open      = p["daily_open"],
        daily_atr       = p["daily_atr"],
        session         = p["session"],
        day_of_week     = et_entry.strftime("%a"),
        hour_of_entry   = et_entry.hour,
        trend_at_entry  = p["trend"],
        macd_line       = p.get("macd_line"),
        macd_signal     = p.get("macd_signal"),
        macd_separation = p.get("macd_separation"),
        macd_slope      = p.get("macd_slope"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_atrmt_research(
    bars: list[Bar],
    cfg:  ATRMTConfig,
) -> list[ATRMTTrade]:
    """Walk-forward simulation of the ATR MACD Trend strategy."""
    daily_states  = _build_daily_states(bars)
    state_by_date = {s.trade_date: s for s in daily_states}

    # 15M MACD indicators (continuous — never reset)
    acc15m       = MinuteBarAccumulator(15)
    macd_fast    = _EMA(cfg.macd_fast_length)
    macd_slow    = _EMA(cfg.macd_slow_length)
    macd_sig_ema = _EMA(cfg.macd_signal_length)

    cur_macd_line:   Optional[float] = None
    cur_macd_signal: Optional[float] = None
    cur_macd_sep:    Optional[float] = None
    cur_macd_slope:  Optional[float] = None
    prev_macd_line:  Optional[float] = None

    position: Optional[dict] = None

    cur_date:       Optional[date] = None
    day_trade_count = 0
    day_pnl         = 0.0
    day_halted      = False

    trades: list[ATRMTTrade] = []
    tick   = cfg.tick_size
    buf    = cfg.entry_buffer_ticks * tick
    slip   = cfg.slippage_ticks * tick

    for bar in bars:
        bar_date = _cme_day(bar.timestamp)
        bar_time = _et_time(bar.timestamp)

        # ── Day rollover ──────────────────────────────────────────────────────
        if bar_date != cur_date:
            if position is not None:
                trades.append(_close_position(
                    position, bar.open, bar.timestamp, "session_end", cfg
                ))
                position = None
            cur_date        = bar_date
            day_trade_count = 0
            day_pnl         = 0.0
            day_halted      = False

        ds = state_by_date.get(bar_date)
        if ds is None:
            continue

        # ── Update 15M MACD ───────────────────────────────────────────────────
        completed_15m = acc15m.update(bar)
        if completed_15m is not None:
            fv = macd_fast.update(completed_15m.close)
            sv = macd_slow.update(completed_15m.close)
            if fv is not None and sv is not None:
                prev_macd_line = cur_macd_line
                cur_macd_line  = fv - sv
                sig = macd_sig_ema.update(cur_macd_line)
                if sig is not None:
                    cur_macd_signal = sig
                    cur_macd_sep    = abs(cur_macd_line - sig)
                if prev_macd_line is not None:
                    cur_macd_slope = cur_macd_line - prev_macd_line

        # ── Manage open position (stop / target on 1M bars) ───────────────────
        if position is not None and bar.timestamp > position["entry_time"]:
            p = position
            if p["dir"] == "long":
                hit_stop   = bar.low  <= p["stop"]
                hit_target = bar.high >= p["target"]
            else:
                hit_stop   = bar.high >= p["stop"]
                hit_target = bar.low  <= p["target"]

            if hit_stop or hit_target:
                reason  = "target" if hit_target else "stop"
                exit_px = p["target"] if hit_target else p["stop"]
                t = _close_position(p, exit_px, bar.timestamp, reason, cfg)
                trades.append(t)
                day_pnl += t.pnl_usd
                position = None

        # ── Force-flat outside session ────────────────────────────────────────
        if position is not None and not _in_window(bar_time, cfg.trade_start_et, cfg.trade_end_et):
            t = _close_position(position, bar.close, bar.timestamp, "session_end", cfg)
            trades.append(t)
            day_pnl += t.pnl_usd
            position = None

        # ── Daily halt check ──────────────────────────────────────────────────
        if not day_halted and (
            day_pnl <= cfg.daily_loss_limit_usd
            or day_pnl >= cfg.daily_profit_limit_usd
        ):
            day_halted = True

        # ── Entry check ───────────────────────────────────────────────────────
        if (
            position is None
            and not day_halted
            and day_trade_count < cfg.max_trades_per_day
            and _in_window(bar_time, cfg.trade_start_et, cfg.entry_cutoff_et)
        ):
            trend = getTrend(cur_macd_line, cur_macd_signal, cur_macd_slope,
                             cfg.min_macd_separation)

            direction: Optional[str] = None
            if trend == TrendState.BULL and cfg.allow_longs:
                direction = "long"
            elif trend == TrendState.BEAR and cfg.allow_shorts:
                direction = "short"

            if direction is not None:
                touched = findTouchedLevel(bar, ds.levels, direction, buf)
                if touched is not None:
                    level_price, level_pct = touched
                    ts_info = getTargetAndStop(direction, level_pct, ds.levels)
                    if ts_info is not None:
                        target, adj_stop = ts_info
                        e_px = bar.close + slip if direction == "long" else bar.close - slip
                        if cfg.stop_atr_pct > 0:
                            atr_dist = cfg.stop_atr_pct * ds.daily_atr
                            stop = (e_px - atr_dist) if direction == "long" else (e_px + atr_dist)
                        else:
                            stop = adj_stop
                        position = {
                            "dir":          direction,
                            "entry":        e_px,
                            "entry_time":   bar.timestamp,
                            "stop":         stop,
                            "target":       target,
                            "level_pct":    level_pct,
                            "level_price":  level_price,
                            "daily_open":   ds.daily_open,
                            "daily_atr":    ds.daily_atr,
                            "session":      _session(bar.timestamp),
                            "trend":        trend.value,
                            "macd_line":    cur_macd_line,
                            "macd_signal":  cur_macd_signal,
                            "macd_separation": cur_macd_sep,
                            "macd_slope":   cur_macd_slope,
                        }
                        day_trade_count += 1

    if position is not None:
        last = bars[-1]
        trades.append(_close_position(position, last.close, last.timestamp, "session_end", cfg))

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_atrmt_trades_csv(trades: list[ATRMTTrade], path: Path) -> None:
    if not trades:
        return
    fields = list(ATRMTTrade.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in trades:
            row = {f: getattr(t, f) for f in fields}
            row["date"]       = str(row["date"])
            row["entry_time"] = str(row["entry_time"])
            row["exit_time"]  = str(row["exit_time"])
            w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _stats_block(label: str, group: list[ATRMTTrade], col: int = 18) -> str:
    if not group:
        return f"  {label:<{col}}  (no trades)"
    wins   = [t for t in group if t.pnl_usd > 0]
    losses = [t for t in group if t.pnl_usd <= 0]
    gw     = sum(t.pnl_usd for t in wins)
    gl     = abs(sum(t.pnl_usd for t in losses))
    pnl    = sum(t.pnl_usd for t in group)
    wr     = len(wins) / len(group) * 100
    avg_w  = gw / len(wins)   if wins   else 0.0
    avg_l  = gl / len(losses) if losses else 0.0
    avg_r  = statistics.mean(t.pnl_r for t in group)
    pf     = gw / gl           if gl    else float("inf")
    pf_s   = f"{pf:.2f}"       if pf   != float("inf") else "∞"
    return (
        f"  {label:<{col}}  N={len(group):>4}  W={len(wins):>3}  L={len(losses):>3}  "
        f"WR={wr:>5.1f}%  PnL=${pnl:>9,.0f}  AvgR={avg_r:>+5.2f}  PF={pf_s:>5}  "
        f"AvgW=${avg_w:>6,.0f}  AvgL=${avg_l:>6,.0f}"
    )


def print_atrmt_summary(trades: list[ATRMTTrade], cfg: ATRMTConfig) -> None:
    if not trades:
        print("No trades generated.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw     = sum(t.pnl_usd for t in wins)
    gl     = abs(sum(t.pnl_usd for t in losses))
    pnl    = sum(t.pnl_usd for t in trades)
    wr     = len(wins) / len(trades) * 100
    avg_w  = gw / len(wins)   if wins   else 0.0
    avg_l  = gl / len(losses) if losses else 0.0
    avg_r  = statistics.mean(t.pnl_r for t in trades)
    pf     = gw / gl           if gl    else float("inf")
    pf_s   = f"{pf:.2f}"       if pf   != float("inf") else "∞"

    equity = peak = max_dd = 0.0
    daily_pnl: dict[date, float] = defaultdict(float)
    for t in sorted(trades, key=lambda x: x.entry_time):
        equity    += t.pnl_usd
        peak       = max(peak, equity)
        max_dd     = max(max_dd, peak - equity)
        daily_pnl[t.date] += t.pnl_usd

    sharpe = 0.0
    dpnl   = list(daily_pnl.values())
    if len(dpnl) >= 2:
        mu  = statistics.mean(dpnl)
        std = statistics.stdev(dpnl)
        sharpe = mu / std * math.sqrt(252) if std > 0 else 0.0

    dates = sorted({t.date for t in trades})

    print(f"\n{'='*80}")
    print(f"  ATR MACD TREND STRATEGY  |  15M MACD({cfg.macd_fast_length}/{cfg.macd_slow_length}/{cfg.macd_signal_length})"
          f"  minSep={cfg.min_macd_separation}  |  {cfg.contracts} contracts")
    print(f"  {dates[0]}  →  {dates[-1]}  |  {len(dates)} trading days")
    print(f"{'='*80}")

    print(f"\n── OVERALL ───────────────────────────────────────────────────────────────────")
    print(f"  Total trades     : {len(trades)}")
    print(f"  Wins / Losses    : {len(wins)} / {len(losses)}")
    print(f"  Win rate         : {wr:.1f}%")
    print(f"  Total PnL        : ${pnl:>10,.2f}")
    print(f"  Profit factor    : {pf_s}")
    print(f"  Avg win          : ${avg_w:>8,.2f}")
    print(f"  Avg loss         : ${avg_l:>8,.2f}")
    print(f"  Avg R            : {avg_r:>+7.3f}")
    print(f"  Max drawdown     : ${max_dd:>8,.2f}")
    print(f"  Sharpe (ann.)    : {sharpe:.2f}")

    print(f"\n── LONG / SHORT ──────────────────────────────────────────────────────────────")
    print(_stats_block("Long",  [t for t in trades if t.direction == "long"]))
    print(_stats_block("Short", [t for t in trades if t.direction == "short"]))

    print(f"\n── BY TREND ──────────────────────────────────────────────────────────────────")
    print(_stats_block("Bull trend (long)",  [t for t in trades if t.trend_at_entry == "BULL"]))
    print(_stats_block("Bear trend (short)", [t for t in trades if t.trend_at_entry == "BEAR"]))

    print(f"\n── BY SESSION ────────────────────────────────────────────────────────────────")
    for ses in ("asia", "london", "ny", "other"):
        grp = [t for t in trades if t.session == ses]
        if grp:
            print(_stats_block(ses, grp))

    print(f"\n── BY DAY OF WEEK ────────────────────────────────────────────────────────────")
    for dow in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        grp = [t for t in trades if t.day_of_week == dow]
        if grp:
            print(_stats_block(dow, grp))

    print(f"\n── BY ATR LEVEL ──────────────────────────────────────────────────────────────")
    lvl_groups: dict[float, list[ATRMTTrade]] = defaultdict(list)
    for t in trades:
        lvl_groups[t.atr_level_pct].append(t)
    for pct in sorted(lvl_groups.keys()):
        lbl = f"{'+' if pct >= 0 else ''}{int(pct * 100)}%"
        print(_stats_block(f"Level {lbl}", lvl_groups[pct]))

    print(f"\n── BY EXIT REASON ────────────────────────────────────────────────────────────")
    er_groups: dict[str, list[ATRMTTrade]] = defaultdict(list)
    for t in trades:
        er_groups[t.exit_reason].append(t)
    for reason in sorted(er_groups.keys()):
        grp = er_groups[reason]
        print(f"  {reason:<14}  N={len(grp):>4}  PnL=${sum(t.pnl_usd for t in grp):>10,.0f}")

    # MACD at entry
    mt = [t for t in trades if t.macd_separation is not None and t.macd_slope is not None]
    if mt:
        print(f"\n── MACD AT ENTRY (15M, {cfg.macd_fast_length}/{cfg.macd_slow_length}/{cfg.macd_signal_length}) ──────────────────────────────────────────")
        seps   = sorted(t.macd_separation for t in mt)
        n      = len(seps)
        q1, q2, q3 = seps[n//4], seps[n//2], seps[3*n//4]
        print(f"  Separation quartiles: Q1={q1:.3f}  Q2={q2:.3f}  Q3={q3:.3f}")
        for lbl, grp in [
            (f"Sep Q1 (<={q1:.3f})",  [t for t in mt if t.macd_separation <= q1]),
            (f"Sep Q2 (<={q2:.3f})",  [t for t in mt if q1 < t.macd_separation <= q2]),
            (f"Sep Q3 (<={q3:.3f})",  [t for t in mt if q2 < t.macd_separation <= q3]),
            (f"Sep Q4 (>{q3:.3f})",   [t for t in mt if t.macd_separation > q3]),
        ]:
            print(_stats_block(lbl, grp))

        slopes = sorted(abs(t.macd_slope) for t in mt)
        s1, s2, s3 = slopes[n//4], slopes[n//2], slopes[3*n//4]
        print(f"\n  |Slope| quartiles:    Q1={s1:.3f}  Q2={s2:.3f}  Q3={s3:.3f}")
        for lbl, grp in [
            (f"|Slope| Q1 (<={s1:.3f})", [t for t in mt if abs(t.macd_slope) <= s1]),
            (f"|Slope| Q2 (<={s2:.3f})", [t for t in mt if s1 < abs(t.macd_slope) <= s2]),
            (f"|Slope| Q3 (<={s3:.3f})", [t for t in mt if s2 < abs(t.macd_slope) <= s3]),
            (f"|Slope| Q4 (>{s3:.3f})",  [t for t in mt if abs(t.macd_slope) > s3]),
        ]:
            print(_stats_block(lbl, grp))

    print()
