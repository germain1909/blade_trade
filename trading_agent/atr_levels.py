from __future__ import annotations

"""
ATR Percentage Level backtesting framework.

Levels are derived from the Daily Open and the 14-period Wilder Daily ATR:
  Level = Daily_Open + (Daily_ATR × pct)

Percentages: -150%, -125%, -100%, -75%, -50%, -25%, 0%, +25%, +50%, +75%,
             +100%, +125%, +150%

Trend filter: 15-minute 7 EMA — long bias when previous closed 15m bar's
close > 7 EMA; short bias when below.

Entry: immediate when price touches an ATR level (simulated market order at
the level price, not on candle close).

Exits:
  - Target: next ATR level in the direction of trade
  - Stop:   ATR level immediately behind entry
  - Break-even: move stop to entry when unrealized profit >= breakeven_usd
  - Session end (flat at close of last 1m bar in trading window)

Conservative: if both stop and target hit same 15m candle, stop wins.
"""

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .indicators import MinuteBarAccumulator, _EMA, _WilderEMA, Bar5m
from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

# Percentages for ATR levels (expressed as fractions: -1.5 … +1.5)
_ATR_PCTS: list[float] = [-1.50, -1.25, -1.00, -0.75, -0.50, -0.25,
                            0.00, +0.25, +0.50, +0.75, +1.00, +1.25, +1.50]

# Trading window (ET)
_TRADE_START  = dtime(8, 30)
_ENTRY_CUTOFF = dtime(15, 0)   # no new entries at or after this time
_TRADE_END    = dtime(16, 0)   # force-close any open position at this time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ATRLevelConfig:
    atr_period:       int   = 14
    ema_period:       int   = 7
    bar_minutes:      int   = 15         # chart timeframe for EMA + signal
    contracts:        int   = 3
    tick_size:        float = 0.25
    slippage_ticks:   int   = 1
    dollars_per_point: float = 5.0

    breakeven_usd:    float = 100.0      # move stop to entry at this unrealized profit
    max_trades_per_day: int = 10
    daily_loss_limit_usd: float = -1000.0
    daily_profit_limit_usd: float = 2000.0

    # Chop filter
    blacklist_on_stop: bool = True       # ban a level for the rest of the day after a full stop


# ---------------------------------------------------------------------------
# Per-trade record
# ---------------------------------------------------------------------------

@dataclass
class ATRTrade:
    date:             date
    entry_time:       datetime
    direction:        str       # "long" | "short"
    entry_atr_pct:    float     # pct label of the touched level (e.g. +0.25)
    entry_price:      float
    stop_atr_pct:     float
    stop_price:       float
    target_atr_pct:   float
    target_price:     float
    exit_time:        datetime
    exit_price:       float
    exit_reason:      str       # "target" | "stop" | "breakeven_stop" | "session_end"
    pnl_usd:          float
    contracts:        int
    mfe_usd:          float     # max favorable excursion in USD
    mae_usd:          float     # max adverse excursion in USD
    daily_open:       float
    daily_atr:        float
    ema7_at_entry:    float     # 7 EMA value on prev completed 15m bar
    day_of_week:      str       # "Mon" … "Fri"
    hour_of_entry:    int       # ET hour


# ---------------------------------------------------------------------------
# Internal daily state
# ---------------------------------------------------------------------------

@dataclass
class _DailyState:
    trade_date:  date
    daily_open:  float
    daily_atr:   float
    levels:      list[float]   # 13 price levels in ascending order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET)


def _et_time(ts: datetime) -> dtime:
    return _et(ts).time()


def _et_date(ts: datetime) -> date:
    return _et(ts).date()


def _cme_day(ts: datetime) -> date:
    """CME session day: bars at or after 18:00 ET belong to the next calendar day."""
    from datetime import timedelta
    et = _et(ts)
    if et.time() >= dtime(18, 0):
        return (et + timedelta(days=1)).date()
    return et.date()


def _level_prices(daily_open: float, daily_atr: float) -> list[float]:
    """Return 13 ATR level prices in ascending order."""
    return [round(daily_open + daily_atr * p, 4) for p in _ATR_PCTS]


def _dpp(cfg: ATRLevelConfig) -> float:
    return cfg.dollars_per_point * cfg.contracts


def _pts_to_usd(pts: float, cfg: ATRLevelConfig) -> float:
    return pts * _dpp(cfg)


def _slip(cfg: ATRLevelConfig) -> float:
    return cfg.slippage_ticks * cfg.tick_size


# ---------------------------------------------------------------------------
# Pass 1 — build daily bars and compute Wilder ATR
# ---------------------------------------------------------------------------

def _build_daily_states(bars: list[Bar]) -> list[_DailyState]:
    """
    Aggregate 1m bars into CME session OHLC (matches TradingView daily bars).
    Session for day D: D-1 18:00 ET → D 17:00 ET.
    Compute Wilder's 14-period ATR.  Return list of _DailyState sorted by date.
    """
    daily: dict[date, dict] = {}
    for bar in bars:
        d = _cme_day(bar.timestamp)
        if d not in daily:
            daily[d] = {"open": bar.open, "high": bar.high,
                        "low": bar.low,  "close": bar.close}
        else:
            s = daily[d]
            s["high"]  = max(s["high"],  bar.high)
            s["low"]   = min(s["low"],   bar.low)
            s["close"] = bar.close

    sorted_dates = sorted(daily.keys())
    if not sorted_dates:
        return []

    # Wilder ATR
    period   = 14
    tr_list: list[float] = []
    prev_close: Optional[float] = None
    atr_vals: dict[date, float] = {}
    wilder    = _WilderEMA(period)

    for d in sorted_dates:
        s   = daily[d]
        hi, lo, cl = s["high"], s["low"], s["close"]
        if prev_close is None:
            tr = hi - lo
        else:
            tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        prev_close = cl
        atr = wilder.update(tr)
        if atr is not None:
            atr_vals[d] = atr

    # Build _DailyState for each day that has an ATR value (skip warmup)
    states: list[_DailyState] = []
    for d in sorted_dates:
        if d not in atr_vals:
            continue
        s   = daily[d]
        atr = atr_vals[d]
        levels = _level_prices(s["open"], atr)
        states.append(_DailyState(
            trade_date  = d,
            daily_open  = s["open"],
            daily_atr   = atr,
            levels      = levels,
        ))
    return states


# ---------------------------------------------------------------------------
# Pass 2 — simulate trades on 15-minute bars with 7 EMA trend filter
# ---------------------------------------------------------------------------

def run_atr_research(
    bars: list[Bar],
    cfg:  ATRLevelConfig,
) -> list[ATRTrade]:
    """
    Full walk-forward simulation of the ATR Percentage Level strategy.
    Returns a list of ATRTrade records.
    """
    daily_states  = _build_daily_states(bars)
    if not daily_states:
        return []

    # Index daily states by date for O(1) lookup
    state_by_date: dict[date, _DailyState] = {s.trade_date: s for s in daily_states}

    trades: list[ATRTrade] = []

    # Per-day counters reset at new day
    cur_date:          Optional[date]   = None
    day_trade_count:   int              = 0
    day_pnl:           float            = 0.0
    day_halted:        bool             = False
    banned_levels:     set              = set()   # levels stopped out today

    # Open position state
    position: Optional[dict] = None    # see keys below
    mfe_pts   = 0.0
    mae_pts   = 0.0

    # 15m accumulator and 7 EMA — never reset between days (continuous series)
    accum15  = MinuteBarAccumulator(cfg.bar_minutes)
    ema7     = _EMA(cfg.ema_period)
    prev_15m_close:  Optional[float] = None   # last completed 15m bar close
    prev_ema7_val:   Optional[float] = None   # 7 EMA at last completed 15m bar
    ema7_at_entry:   float           = 0.0
    ema7_history:    list            = []      # last 3 EMA values, oldest first

    # 4H accumulator with 21 and 50 EMAs for higher-timeframe regime filter
    accum4h      = MinuteBarAccumulator(240)
    ema21_4h     = _EMA(21)
    ema50_4h     = _EMA(50)
    h4_close:    Optional[float] = None   # last completed 4H bar close
    h4_ema21:    Optional[float] = None
    h4_ema50:    Optional[float] = None

    def _reset_day(d: date) -> None:
        nonlocal cur_date, day_trade_count, day_pnl, day_halted, banned_levels
        cur_date        = d
        day_trade_count = 0
        day_pnl         = 0.0
        day_halted      = False
        banned_levels   = set()

    def _close_position(p: dict, exit_px: float, exit_ts: datetime, reason: str) -> ATRTrade:
        nonlocal day_pnl, day_trade_count, position, mfe_pts, mae_pts
        direction = p["dir"]
        if direction == "long":
            pnl_pts = exit_px - p["entry"]
        else:
            pnl_pts = p["entry"] - exit_px
        pnl_usd = _pts_to_usd(pnl_pts, cfg)
        mfe_usd = _pts_to_usd(mfe_pts, cfg)
        mae_usd = _pts_to_usd(mae_pts, cfg)
        day_pnl += pnl_usd

        et_entry = _et(p["entry_time"])
        t = ATRTrade(
            date           = _et_date(p["entry_time"]),
            entry_time     = p["entry_time"],
            direction      = direction,
            entry_atr_pct  = p["entry_pct"],
            entry_price    = p["entry"],
            stop_atr_pct   = p["stop_pct"],
            stop_price     = p["stop"],
            target_atr_pct = p["target_pct"],
            target_price   = p["target"],
            exit_time      = exit_ts,
            exit_price     = exit_px,
            exit_reason    = reason,
            pnl_usd        = round(pnl_usd, 2),
            contracts      = cfg.contracts,
            mfe_usd        = round(mfe_usd, 2),
            mae_usd        = round(mae_usd, 2),
            daily_open     = p["daily_open"],
            daily_atr      = p["daily_atr"],
            ema7_at_entry  = p["ema7_at_entry"],
            day_of_week    = et_entry.strftime("%a"),
            hour_of_entry  = et_entry.hour,
        )
        position = None
        mfe_pts  = 0.0
        mae_pts  = 0.0
        return t

    slip = _slip(cfg)
    dpp  = _dpp(cfg)

    prev_bar_high: Optional[float] = None
    prev_bar_low:  Optional[float] = None

    for bar in bars:
        bar_date = _cme_day(bar.timestamp)
        bar_time = _et_time(bar.timestamp)

        # Day rollover
        if bar_date != cur_date:
            # Force close any open position at session end (carry-over shouldn't happen
            # since we close at TRADE_END, but guard anyway)
            if position is not None:
                trades.append(_close_position(position, bar.open, bar.timestamp, "session_end"))
            _reset_day(bar_date)

        # Feed the 15m accumulator — emits a completed Bar5m when next bucket starts
        completed_15m: Optional[Bar5m] = accum15.update(bar)
        if completed_15m is not None:
            ema_val = ema7.update(completed_15m.close)
            if ema_val is not None:
                prev_ema7_val  = ema_val
                prev_15m_close = completed_15m.close
                ema7_history.append(ema_val)
                if len(ema7_history) > 3:
                    ema7_history.pop(0)

        completed_4h: Optional[Bar5m] = accum4h.update(bar)
        if completed_4h is not None:
            v21 = ema21_4h.update(completed_4h.close)
            v50 = ema50_4h.update(completed_4h.close)
            h4_close = completed_4h.close
            if v21 is not None:
                h4_ema21 = v21
            if v50 is not None:
                h4_ema50 = v50

        # Only act during the trading window
        if not (_TRADE_START <= bar_time < _TRADE_END):
            if position is not None and bar_time >= _TRADE_END:
                trades.append(_close_position(position, bar.close, bar.timestamp, "session_end"))
            continue

        # ── Daily state needed for levels ────────────────────────────────
        ds = state_by_date.get(bar_date)
        if ds is None:
            continue   # warmup days (no ATR yet)

        levels = ds.levels  # 13 floats, ascending

        # ── Update MFE / MAE on open position ────────────────────────────
        if position is not None:
            p = position
            if p["dir"] == "long":
                mfe_pts = max(mfe_pts, bar.high - p["entry"])
                mae_pts = max(mae_pts, p["entry"] - bar.low)
            else:
                mfe_pts = max(mfe_pts, p["entry"] - bar.low)
                mae_pts = max(mae_pts, bar.high - p["entry"])

        # ── Break-even management ─────────────────────────────────────────
        if position is not None:
            p = position
            if not p["be_applied"]:
                unreal_usd = _pts_to_usd(
                    (bar.close - p["entry"]) if p["dir"] == "long" else (p["entry"] - bar.close),
                    cfg
                )
                if unreal_usd >= cfg.breakeven_usd:
                    p["stop"]       = p["entry"]
                    p["stop_pct"]   = p["entry_pct"]  # symbolic — entry level
                    p["be_applied"] = True

        # ── Exit check on open position ───────────────────────────────────
        if position is not None:
            p       = position
            hit_stop   = False
            hit_target = False
            if p["dir"] == "long":
                hit_stop   = bar.low  <= p["stop"]
                hit_target = bar.high >= p["target"]
            else:
                hit_stop   = bar.high >= p["stop"]
                hit_target = bar.low  <= p["target"]

            if hit_stop or hit_target:
                if hit_stop and hit_target:
                    # Conservative: stop wins
                    reason   = "breakeven_stop" if p["be_applied"] else "stop"
                    exit_px  = p["stop"]
                elif hit_stop:
                    reason   = "breakeven_stop" if p["be_applied"] else "stop"
                    exit_px  = p["stop"]
                else:
                    reason   = "target"
                    exit_px  = p["target"]
                # Ban the entry level for the rest of the day on a full stop
                if reason == "stop" and cfg.blacklist_on_stop:
                    banned_levels.add(p["entry_pct"])
                trades.append(_close_position(p, exit_px, bar.timestamp, reason))

        # ── Entry logic ───────────────────────────────────────────────────
        if (position is None
                and not day_halted
                and bar_time < _ENTRY_CUTOFF
                and day_trade_count < cfg.max_trades_per_day
                and day_pnl > cfg.daily_loss_limit_usd
                and day_pnl < cfg.daily_profit_limit_usd
                and prev_ema7_val is not None
                and prev_15m_close is not None):

            ema_rising  = len(ema7_history) >= 3 and ema7_history[-1] > ema7_history[0]
            ema_falling = len(ema7_history) >= 3 and ema7_history[-1] < ema7_history[0]

            # 4H regime: bull = longs only, bear = shorts only, neutral = both
            if h4_close is not None and h4_ema21 is not None and h4_ema50 is not None:
                h4_bull = h4_close > h4_ema21 and h4_close > h4_ema50
                h4_bear = h4_close < h4_ema21 and h4_close < h4_ema50
            else:
                h4_bull = h4_bear = False   # no 4H data yet — neutral, allow both

            allow_long  = h4_bull or (not h4_bull and not h4_bear)
            allow_short = h4_bear or (not h4_bull and not h4_bear)

            long_bias  = prev_15m_close > prev_ema7_val and ema_rising  and allow_long
            short_bias = prev_15m_close < prev_ema7_val and ema_falling and allow_short

            # Check each level for a touch this bar
            for idx, lvl_price in enumerate(levels):
                if position is not None:
                    break  # only one entry per bar

                if (_ATR_PCTS[idx] in banned_levels):
                    continue

                if (long_bias
                        and prev_bar_high is not None
                        and prev_bar_high < lvl_price
                        and bar.high >= lvl_price):
                    # Price approached level from below — long entry
                    entry_px = lvl_price + slip
                    if idx + 1 >= len(levels):
                        continue   # no target level above
                    target_px  = levels[idx + 1]
                    if idx - 1 < 0:
                        continue   # no stop level below
                    stop_px    = levels[idx - 1]
                    if entry_px <= stop_px:
                        continue   # degenerate

                    position = {
                        "dir":        "long",
                        "entry":      entry_px,
                        "entry_time": bar.timestamp,
                        "entry_pct":  _ATR_PCTS[idx],
                        "stop":       stop_px,
                        "stop_pct":   _ATR_PCTS[idx - 1],
                        "target":     target_px,
                        "target_pct": _ATR_PCTS[idx + 1],
                        "be_applied": False,
                        "daily_open": ds.daily_open,
                        "daily_atr":  ds.daily_atr,
                        "ema7_at_entry": prev_ema7_val,
                    }
                    day_trade_count += 1

                elif (short_bias
                        and prev_bar_low is not None
                        and prev_bar_low > lvl_price
                        and bar.low <= lvl_price):
                    # Price approached level from above — short entry
                    entry_px = lvl_price - slip
                    if idx - 1 < 0:
                        continue   # no target level below
                    target_px  = levels[idx - 1]
                    if idx + 1 >= len(levels):
                        continue   # no stop level above
                    stop_px    = levels[idx + 1]
                    if entry_px >= stop_px:
                        continue

                    position = {
                        "dir":        "short",
                        "entry":      entry_px,
                        "entry_time": bar.timestamp,
                        "entry_pct":  _ATR_PCTS[idx],
                        "stop":       stop_px,
                        "stop_pct":   _ATR_PCTS[idx + 1],
                        "target":     target_px,
                        "target_pct": _ATR_PCTS[idx - 1],
                        "be_applied": False,
                        "daily_open": ds.daily_open,
                        "daily_atr":  ds.daily_atr,
                        "ema7_at_entry": prev_ema7_val,
                    }
                    day_trade_count += 1

        prev_bar_high = bar.high
        prev_bar_low  = bar.low

    # Force-close any remaining position at end of data
    if position is not None:
        last_bar = bars[-1]
        trades.append(_close_position(position, last_bar.close, last_bar.timestamp, "session_end"))

    return trades


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_atr_trades_csv(trades: list[ATRTrade], path: Path) -> None:
    if not trades:
        return
    fields = list(ATRTrade.__dataclass_fields__)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in trades:
            row = {f: getattr(t, f) for f in fields}
            row["date"]       = str(row["date"])
            row["entry_time"] = str(row["entry_time"])
            row["exit_time"]  = str(row["exit_time"])
            w.writerow(row)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _sharpe(pnl_list: list[float]) -> float:
    if len(pnl_list) < 2:
        return 0.0
    mu  = statistics.mean(pnl_list)
    std = statistics.stdev(pnl_list)
    return (mu / std * math.sqrt(252)) if std > 0 else 0.0


def _pct_lbl(pct: float) -> str:
    return f"{'+' if pct >= 0 else ''}{int(pct * 100)}%"


def _stats_block(label: str, trades: list[ATRTrade]) -> str:
    if not trades:
        return f"  {label}: (no trades)"
    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
    pnl  = sum(t.pnl_usd for t in trades)
    wr   = len(wins) / len(trades) * 100
    avg_w = gw / len(wins)   if wins   else 0.0
    avg_l = gl / len(losses) if losses else 0.0
    exp   = wr / 100 * avg_w - (1 - wr / 100) * avg_l
    return (
        f"  {label:<20}  N={len(trades):>4}  W={len(wins):>3}  L={len(losses):>3}  "
        f"WR={wr:>5.1f}%  PnL=${pnl:>10,.0f}  PF={pf_s:>5}  "
        f"AvgW=${avg_w:>6,.0f}  AvgL=${avg_l:>6,.0f}  Exp=${exp:>6,.0f}"
    )


def print_atr_summary(trades: list[ATRTrade], cfg: ATRLevelConfig) -> None:
    if not trades:
        print("No trades generated.")
        return

    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
    pnl  = sum(t.pnl_usd for t in trades)
    wr   = len(wins) / len(trades) * 100
    avg_w = gw / len(wins)   if wins   else 0.0
    avg_l = gl / len(losses) if losses else 0.0
    exp   = wr / 100 * avg_w - (1 - wr / 100) * avg_l

    # Drawdown
    equity, peak, max_dd = 0.0, 0.0, 0.0
    daily_pnl: dict[date, float] = defaultdict(float)
    for t in sorted(trades, key=lambda x: x.entry_time):
        equity    += t.pnl_usd
        peak       = max(peak, equity)
        max_dd     = max(max_dd, peak - equity)
        daily_pnl[t.date] += t.pnl_usd

    sharpe = _sharpe(list(daily_pnl.values()))

    dates = sorted({t.date for t in trades})

    print(f"\n{'='*76}")
    print(f"  ATR PERCENTAGE LEVEL STRATEGY  |  {cfg.bar_minutes}m bars  |  "
          f"{cfg.ema_period} EMA  |  {cfg.contracts} contracts")
    print(f"  {dates[0]}  →  {dates[-1]}  |  "
          f"{len(dates)} trading days")
    print(f"{'='*76}")

    print(f"\n── OVERALL ───────────────────────────────────────────────────────────────")
    print(f"  Total trades      : {len(trades)}")
    print(f"  Wins / Losses     : {len(wins)} / {len(losses)}")
    print(f"  Win rate          : {wr:.1f}%")
    print(f"  Total PnL         : ${pnl:>10,.2f}")
    print(f"  Profit factor     : {pf_s}")
    print(f"  Avg win           : ${avg_w:>8,.2f}")
    print(f"  Avg loss          : ${avg_l:>8,.2f}")
    print(f"  Expectancy        : ${exp:>8,.2f} per trade")
    print(f"  Max drawdown      : ${max_dd:>8,.2f}")
    print(f"  Sharpe (ann.)     : {sharpe:.2f}")

    # Long / short breakdown
    longs  = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]
    print(f"\n── LONG / SHORT ──────────────────────────────────────────────────────────")
    print(_stats_block("Long",  longs))
    print(_stats_block("Short", shorts))

    # Above / below daily open
    above_open = [t for t in trades if t.entry_price >= t.daily_open]
    below_open = [t for t in trades if t.entry_price <  t.daily_open]
    print(f"\n── ABOVE / BELOW DAILY OPEN ──────────────────────────────────────────────")
    print(_stats_block("Above open", above_open))
    print(_stats_block("Below open", below_open))

    # By ATR level touched
    print(f"\n── BY ATR LEVEL TOUCHED (entry_atr_pct) ─────────────────────────────────")
    level_groups: dict[float, list[ATRTrade]] = defaultdict(list)
    for t in trades:
        level_groups[t.entry_atr_pct].append(t)
    for pct in sorted(level_groups.keys()):
        grp = level_groups[pct]
        lbl = _pct_lbl(pct)
        print(_stats_block(f"Level {lbl}", grp))

    # By day of week
    print(f"\n── BY DAY OF WEEK ────────────────────────────────────────────────────────")
    dow_groups: dict[str, list[ATRTrade]] = defaultdict(list)
    for t in trades:
        dow_groups[t.day_of_week].append(t)
    for dow in ("Mon", "Tue", "Wed", "Thu", "Fri"):
        grp = dow_groups.get(dow, [])
        if grp:
            print(_stats_block(dow, grp))

    # By hour of entry (ET)
    print(f"\n── BY HOUR OF ENTRY (ET) ─────────────────────────────────────────────────")
    hour_groups: dict[int, list[ATRTrade]] = defaultdict(list)
    for t in trades:
        hour_groups[t.hour_of_entry].append(t)
    for hr in sorted(hour_groups.keys()):
        grp = hour_groups[hr]
        print(_stats_block(f"{hr:02d}:00–{hr:02d}:59", grp))

    # Exit reason breakdown
    print(f"\n── BY EXIT REASON ────────────────────────────────────────────────────────")
    exit_groups: dict[str, list[ATRTrade]] = defaultdict(list)
    for t in trades:
        exit_groups[t.exit_reason].append(t)
    for reason in sorted(exit_groups.keys()):
        grp = exit_groups[reason]
        pnl_grp = sum(t.pnl_usd for t in grp)
        print(f"  {reason:<20}  N={len(grp):>4}  PnL=${pnl_grp:>10,.0f}")

    # Research: avg move after each level touch
    print(f"\n── ATR LEVEL RESEARCH (avg MFE / MAE per level) ─────────────────────────")
    hdr = f"  {'Level':<10}  {'N':>4}  {'AvgMFE':>8}  {'AvgMAE':>8}  {'WR%':>6}  {'Exp$':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pct in sorted(level_groups.keys()):
        grp      = level_groups[pct]
        avg_mfe  = sum(t.mfe_usd for t in grp) / len(grp)
        avg_mae  = sum(t.mae_usd for t in grp) / len(grp)
        w        = [t for t in grp if t.pnl_usd > 0]
        l        = [t for t in grp if t.pnl_usd <= 0]
        wr_g     = len(w) / len(grp) * 100
        gw_g     = sum(t.pnl_usd for t in w)
        gl_g     = abs(sum(t.pnl_usd for t in l))
        avg_wg   = gw_g / len(w)   if w else 0.0
        avg_lg   = gl_g / len(l)   if l else 0.0
        exp_g    = wr_g / 100 * avg_wg - (1 - wr_g / 100) * avg_lg
        print(
            f"  {_pct_lbl(pct):<10}  {len(grp):>4}  "
            f"${avg_mfe:>7,.0f}  ${avg_mae:>7,.0f}  {wr_g:>5.1f}%  ${exp_g:>7,.0f}"
        )

    print()
