from __future__ import annotations

"""
Pure Opening Range Breakout (ORB) research framework.
No indicators — raw price structure only.

Entry: close crosses OR High (long) or OR Low (short).
One pass collects session-day data; a second pass per (stop_method × target_r)
simulates trades so every combination is evaluated consistently.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

_SESSIONS: list[tuple[str, dtime, dtime]] = [
    ("london",  dtime(4,  0), dtime(8,  0)),
    ("ny_open", dtime(8,  0), dtime(12, 0)),
    ("ny_pm",   dtime(12, 0), dtime(16, 0)),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ORBConfig:
    or_minutes: int = 30

    # Sessions to include (subset of "london", "ny_open", "ny_pm")
    sessions: tuple[str, ...] = ("london", "ny_open", "ny_pm")

    max_trades_per_session: int   = 4
    session_loss_limit_r:   float = -2.0   # halt session once cumulative R loss hits this

    # Stop methods to evaluate
    stop_methods: tuple[str, ...] = ("opposite_or", "fixed_ticks", "or_pct")
    fixed_stop_ticks: int   = 24
    tick_size:        float = 0.25
    or_pct_stop:      float = 0.5    # stop distance = or_size × this

    # Target R multiples to evaluate
    targets_r: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

    contracts:         int   = 1
    dollars_per_point: float = 5.0


# ---------------------------------------------------------------------------
# Per-trade record
# ---------------------------------------------------------------------------

@dataclass
class ORBTrade:
    date:         date
    session:      str
    direction:    str        # "long" | "short"
    entry_time:   datetime
    entry_price:  float
    stop_price:   float
    target_price: float
    stop_method:  str
    target_r:     float
    risk_pts:     float
    exit_price:   float
    exit_time:    datetime
    exit_reason:  str        # "target" | "stop" | "session_end"
    pnl_pts:      float
    pnl_r:        float
    mfe_pts:      float      # max favorable excursion from entry
    mae_pts:      float      # max adverse excursion from entry
    or_high:      float
    or_low:       float
    or_size:      float
    pnl_usd:      float


# ---------------------------------------------------------------------------
# Per-session research stats (independent of stop/target choice)
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    session:              str
    total_days:           int
    days_broke_high:      int
    days_broke_low:       int
    days_broke_both:      int
    days_broke_one_only:  int
    avg_move_after_high_break: float   # pts above OR High (max excursion)
    avg_move_after_low_break:  float   # pts below OR Low  (max excursion)
    avg_or_size:          float
    # Continuation rates (% of break-days reaching N × OR size above/below level)
    cont_high_0_5r: float
    cont_high_1r:   float
    cont_high_2r:   float
    cont_high_3r:   float
    cont_low_0_5r:  float
    cont_low_1r:    float
    cont_low_2r:    float
    cont_low_3r:    float
    # Failure: closed back inside the range after break
    fail_high_rate: float
    fail_low_rate:  float
    # Reversal: after high break, did price reach OR Low (and vice-versa)?
    rev_high_to_low:  float
    rev_low_to_high:  float


# ---------------------------------------------------------------------------
# Internal session-day container
# ---------------------------------------------------------------------------

class _SessionDay:
    __slots__ = (
        "trade_date", "session", "or_high", "or_low", "or_size",
        "last_or_close", "post_or_bars",
    )

    def __init__(
        self, trade_date: date, session: str,
        or_high: float, or_low: float,
        last_or_close: float, post_or_bars: list[Bar],
    ) -> None:
        self.trade_date    = trade_date
        self.session       = session
        self.or_high       = or_high
        self.or_low        = or_low
        self.or_size       = or_high - or_low
        self.last_or_close = last_or_close
        self.post_or_bars  = post_or_bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et_time(ts: datetime) -> dtime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET).time()


def _et_date(ts: datetime) -> date:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET).date()


def _stop_price(
    entry: float, direction: str,
    sd: _SessionDay, cfg: ORBConfig, method: str,
) -> float:
    if method == "opposite_or":
        return sd.or_low if direction == "long" else sd.or_high
    if method == "fixed_ticks":
        dist = cfg.fixed_stop_ticks * cfg.tick_size
        return entry - dist if direction == "long" else entry + dist
    # or_pct
    dist = sd.or_size * cfg.or_pct_stop
    return entry - dist if direction == "long" else entry + dist


# ---------------------------------------------------------------------------
# Pass 1 — build session days
# ---------------------------------------------------------------------------

def _parse_session_days(bars: list[Bar], cfg: ORBConfig) -> list[_SessionDay]:
    states: dict[tuple, dict] = {}

    for bar in bars:
        t = _et_time(bar.timestamp)
        d = _et_date(bar.timestamp)

        for sess_name, sess_start, sess_end in _SESSIONS:
            if sess_name not in cfg.sessions:
                continue
            if not (sess_start <= t < sess_end):
                continue

            or_end_total = sess_start.hour * 60 + sess_start.minute + cfg.or_minutes
            or_end = dtime(or_end_total // 60, or_end_total % 60)
            key = (d, sess_name)

            if key not in states:
                states[key] = {
                    "or_high": float("-inf"),
                    "or_low":  float("inf"),
                    "last_or_close": None,
                    "or_done": False,
                    "post_or_bars": [],
                    "or_end": or_end,
                }

            s = states[key]
            if not s["or_done"]:
                if t < s["or_end"]:
                    s["or_high"]       = max(s["or_high"], bar.high)
                    s["or_low"]        = min(s["or_low"],  bar.low)
                    s["last_or_close"] = bar.close
                else:
                    s["or_done"] = True
                    s["post_or_bars"].append(bar)
            else:
                s["post_or_bars"].append(bar)

    days = []
    for (d, sess_name), s in sorted(states.items()):
        if not s["or_done"] or s["or_high"] == float("-inf"):
            continue
        or_size = s["or_high"] - s["or_low"]
        if or_size < cfg.tick_size:
            continue
        if s["last_or_close"] is None:
            continue
        days.append(_SessionDay(
            trade_date    = d,
            session       = sess_name,
            or_high       = s["or_high"],
            or_low        = s["or_low"],
            last_or_close = s["last_or_close"],
            post_or_bars  = s["post_or_bars"],
        ))
    return days


# ---------------------------------------------------------------------------
# Pass 2 — research metrics (no stop/target dependency)
# ---------------------------------------------------------------------------

def _compute_session_stats(days: list[_SessionDay]) -> dict[str, SessionStats]:
    grouped: dict[str, list[_SessionDay]] = defaultdict(list)
    for d in days:
        grouped[d.session].append(d)

    result: dict[str, SessionStats] = {}

    for sess_name, sess_days in grouped.items():
        total          = len(sess_days)
        broke_high_ct  = broke_low_ct = broke_both_ct = 0
        moves_high: list[float] = []
        moves_low:  list[float] = []
        cont_h = {0.5: 0, 1.0: 0, 2.0: 0, 3.0: 0}
        cont_l = {0.5: 0, 1.0: 0, 2.0: 0, 3.0: 0}
        fail_h = fail_l = 0
        rev_h_to_l = rev_l_to_h = 0
        or_sizes: list[float] = [sd.or_size for sd in sess_days]

        for sd in sess_days:
            oh, ol, os_ = sd.or_high, sd.or_low, sd.or_size
            broke_h = broke_l = False
            idx_h = idx_l = None
            prev_c = sd.last_or_close

            for i, bar in enumerate(sd.post_or_bars):
                c = bar.close
                if not broke_h and c > oh and prev_c <= oh:
                    broke_h = True
                    idx_h   = i
                if not broke_l and c < ol and prev_c >= ol:
                    broke_l = True
                    idx_l   = i
                prev_c = c

            if broke_h:
                broke_high_ct += 1
                after = sd.post_or_bars[idx_h:]
                max_up = max(b.high for b in after) - oh
                moves_high.append(max_up)
                for lvl in (0.5, 1.0, 2.0, 3.0):
                    if max_up >= os_ * lvl:
                        cont_h[lvl] += 1
                if any(b.close <= oh for b in after[1:]):
                    fail_h += 1
                if any(b.low <= ol for b in after):
                    rev_h_to_l += 1

            if broke_l:
                broke_low_ct += 1
                after = sd.post_or_bars[idx_l:]
                max_dn = ol - min(b.low for b in after)
                moves_low.append(max_dn)
                for lvl in (0.5, 1.0, 2.0, 3.0):
                    if max_dn >= os_ * lvl:
                        cont_l[lvl] += 1
                if any(b.close >= ol for b in after[1:]):
                    fail_l += 1
                if any(b.high >= oh for b in after):
                    rev_l_to_h += 1

            if broke_h and broke_l:
                broke_both_ct += 1

        def pct(n: int, d_: int) -> float:
            return round(n / d_ * 100, 1) if d_ > 0 else 0.0

        days_broke_one_only = broke_high_ct + broke_low_ct - 2 * broke_both_ct

        result[sess_name] = SessionStats(
            session             = sess_name,
            total_days          = total,
            days_broke_high     = broke_high_ct,
            days_broke_low      = broke_low_ct,
            days_broke_both     = broke_both_ct,
            days_broke_one_only = days_broke_one_only,
            avg_move_after_high_break = round(sum(moves_high) / len(moves_high), 2) if moves_high else 0.0,
            avg_move_after_low_break  = round(sum(moves_low)  / len(moves_low),  2) if moves_low  else 0.0,
            avg_or_size         = round(sum(or_sizes) / len(or_sizes), 2) if or_sizes else 0.0,
            cont_high_0_5r = pct(cont_h[0.5], broke_high_ct),
            cont_high_1r   = pct(cont_h[1.0], broke_high_ct),
            cont_high_2r   = pct(cont_h[2.0], broke_high_ct),
            cont_high_3r   = pct(cont_h[3.0], broke_high_ct),
            cont_low_0_5r  = pct(cont_l[0.5], broke_low_ct),
            cont_low_1r    = pct(cont_l[1.0], broke_low_ct),
            cont_low_2r    = pct(cont_l[2.0], broke_low_ct),
            cont_low_3r    = pct(cont_l[3.0], broke_low_ct),
            fail_high_rate = pct(fail_h, broke_high_ct),
            fail_low_rate  = pct(fail_l, broke_low_ct),
            rev_high_to_low = pct(rev_h_to_l, broke_high_ct),
            rev_low_to_high = pct(rev_l_to_h, broke_low_ct),
        )

    return result


# ---------------------------------------------------------------------------
# Pass 3 — trade simulation for one (stop_method × target_r) combo
# ---------------------------------------------------------------------------

def _simulate_combo(
    days:        list[_SessionDay],
    cfg:         ORBConfig,
    stop_method: str,
    target_r:    float,
) -> list[ORBTrade]:
    trades: list[ORBTrade] = []

    for sd in days:
        n             = len(sd.post_or_bars)
        trades_taken  = 0
        session_r_pnl = 0.0
        position: Optional[dict] = None
        mfe = mae  = 0.0
        prev_close   = sd.last_or_close

        for i, bar in enumerate(sd.post_or_bars):
            is_last = (i == n - 1)

            # ── MFE / MAE update ─────────────────────────────────────────────
            if position is not None:
                if position["dir"] == "long":
                    mfe = max(mfe, bar.high - position["entry"])
                    mae = max(mae, position["entry"] - bar.low)
                else:
                    mfe = max(mfe, position["entry"] - bar.low)
                    mae = max(mae, bar.high - position["entry"])

            # ── Exit check ───────────────────────────────────────────────────
            if position is not None:
                exit_px = exit_reason = None

                if position["dir"] == "long":
                    if bar.low <= position["stop"]:
                        exit_px, exit_reason = position["stop"], "stop"
                    elif bar.high >= position["target"]:
                        exit_px, exit_reason = position["target"], "target"
                else:
                    if bar.high >= position["stop"]:
                        exit_px, exit_reason = position["stop"], "stop"
                    elif bar.low <= position["target"]:
                        exit_px, exit_reason = position["target"], "target"

                if exit_px is None and is_last:
                    exit_px, exit_reason = bar.close, "session_end"

                if exit_px is not None:
                    p       = position
                    pnl_pts = (exit_px - p["entry"]) if p["dir"] == "long" else (p["entry"] - exit_px)
                    pnl_r   = pnl_pts / p["risk"]
                    session_r_pnl += pnl_r
                    trades.append(ORBTrade(
                        date         = sd.trade_date,
                        session      = sd.session,
                        direction    = p["dir"],
                        entry_time   = p["entry_time"],
                        entry_price  = p["entry"],
                        stop_price   = p["stop"],
                        target_price = p["target"],
                        stop_method  = stop_method,
                        target_r     = target_r,
                        risk_pts     = p["risk"],
                        exit_price   = exit_px,
                        exit_time    = bar.timestamp,
                        exit_reason  = exit_reason,
                        pnl_pts      = round(pnl_pts, 4),
                        pnl_r        = round(pnl_r,   4),
                        mfe_pts      = round(mfe,     4),
                        mae_pts      = round(mae,     4),
                        or_high      = sd.or_high,
                        or_low       = sd.or_low,
                        or_size      = sd.or_size,
                        pnl_usd      = round(pnl_pts * cfg.contracts * cfg.dollars_per_point, 2),
                    ))
                    position = None
                    mfe = mae = 0.0

            # ── Entry check (never on the last bar) ──────────────────────────
            if (not is_last
                    and position is None
                    and trades_taken < cfg.max_trades_per_session
                    and session_r_pnl > cfg.session_loss_limit_r
                    and prev_close is not None):

                new_dir: Optional[str] = None
                if bar.close > sd.or_high and prev_close <= sd.or_high:
                    new_dir = "long"
                elif bar.close < sd.or_low and prev_close >= sd.or_low:
                    new_dir = "short"

                if new_dir is not None:
                    entry = bar.close
                    stop  = _stop_price(entry, new_dir, sd, cfg, stop_method)
                    risk  = abs(entry - stop)
                    if risk > 0:
                        tgt = (entry + risk * target_r if new_dir == "long"
                               else entry - risk * target_r)
                        position = {
                            "dir":        new_dir,
                            "entry":      entry,
                            "entry_time": bar.timestamp,
                            "stop":       stop,
                            "target":     tgt,
                            "risk":       risk,
                        }
                        trades_taken += 1

            prev_close = bar.close

    return trades


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_orb_research(
    bars: list[Bar],
    cfg:  ORBConfig,
) -> tuple[dict[tuple[str, float], list[ORBTrade]], dict[str, SessionStats], int]:
    """
    Returns:
        trades_by_combo  — {(stop_method, target_r): [ORBTrade, ...]}
        session_stats    — {session_name: SessionStats}
        total_days       — number of complete session-days found
    """
    days = _parse_session_days(bars, cfg)
    session_stats = _compute_session_stats(days)

    trades_by_combo: dict[tuple[str, float], list[ORBTrade]] = {}
    for sm in cfg.stop_methods:
        for tr in cfg.targets_r:
            trades_by_combo[(sm, tr)] = _simulate_combo(days, cfg, sm, tr)

    return trades_by_combo, session_stats, len(days)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_trades_csv(
    trades: list[ORBTrade],
    path:   Path,
) -> None:
    if not trades:
        return
    fields = [f for f in ORBTrade.__dataclass_fields__]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in trades:
            w.writerow({f: getattr(t, f) for f in fields})


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _combo_summary(trades: list[ORBTrade], contracts: int, dpp: float) -> str:
    if not trades:
        return "  (no trades)"
    wins   = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    pnl = sum(t.pnl_usd for t in trades)
    wr  = len(wins) / len(trades) * 100
    avg_w = gw / len(wins)   if wins   else 0.0
    avg_l = gl / len(losses) if losses else 0.0
    exp   = (wr / 100 * avg_w - (1 - wr / 100) * avg_l)
    pf_s  = f"{pf:.2f}" if pf != float("inf") else "∞"
    return (
        f"  N={len(trades):>3}  W={len(wins):>3}  L={len(losses):>3}  "
        f"WR={wr:>5.1f}%  PnL=${pnl:>9,.0f}  "
        f"AvgW=${avg_w:>6,.0f}  AvgL=${avg_l:>6,.0f}  "
        f"PF={pf_s:>5}  Exp=${exp:>6,.0f}"
    )


def _session_breakdown(trades: list[ORBTrade]) -> str:
    groups: dict[str, list[ORBTrade]] = defaultdict(list)
    for t in trades:
        groups[t.session].append(t)

    col = 8
    hdr = (f"  {'Session':<{col}}  {'N':>3}  {'WR%':>5}  "
           f"{'PnL':>9}  {'PF':>5}  {'MFE avg':>7}  {'MAE avg':>7}")
    sep = "  " + "-" * (len(hdr) - 2)
    rows = [hdr, sep]
    for sess in ("london", "ny_open", "ny_pm"):
        grp = groups.get(sess, [])
        if not grp:
            continue
        wins  = [t for t in grp if t.pnl_usd > 0]
        gl    = abs(sum(t.pnl_usd for t in grp if t.pnl_usd <= 0))
        gw    = sum(t.pnl_usd for t in wins)
        pf    = gw / gl if gl > 0 else float("inf")
        pf_s  = f"{pf:.2f}" if pf != float("inf") else "∞"
        wr    = len(wins) / len(grp) * 100
        pnl   = sum(t.pnl_usd for t in grp)
        avg_mfe = sum(t.mfe_pts for t in grp) / len(grp)
        avg_mae = sum(t.mae_pts for t in grp) / len(grp)
        rows.append(
            f"  {sess:<{col}}  {len(grp):>3}  {wr:>5.1f}  "
            f"${pnl:>8,.0f}  {pf_s:>5}  {avg_mfe:>7.2f}  {avg_mae:>7.2f}"
        )
    return "\n".join(rows)


def print_research_summary(
    trades_by_combo: dict[tuple[str, float], list[ORBTrade]],
    session_stats:   dict[str, SessionStats],
    total_days:      int,
    cfg:             ORBConfig,
) -> None:
    print(f"\n{'='*72}")
    print(f"  ORB RESEARCH  |  {total_days} session-days")
    print(f"  OR duration: {cfg.or_minutes}m   Sessions: London / NY Open / NY PM")
    print(f"{'='*72}")

    # ── Research metrics per session ──────────────────────────────────────
    print("\n── OPENING RANGE STATISTICS ──────────────────────────────────────────")
    for sess_name in ("london", "ny_open", "ny_pm"):
        st = session_stats.get(sess_name)
        if st is None:
            continue
        print(f"\n  {sess_name.upper()} ({st.total_days} days)  |  avg OR size: {st.avg_or_size:.2f} pts")
        print(f"    Broke high : {st.days_broke_high:>3} days  |  Broke low : {st.days_broke_low:>3} days  "
              f"|  Both sides: {st.days_broke_both:>3}  |  One side only: {st.days_broke_one_only:>3}")
        print(f"    Avg move after high break : {st.avg_move_after_high_break:>6.2f} pts  "
              f"({st.avg_move_after_high_break / st.avg_or_size * 100:.0f}% of avg OR)")
        print(f"    Avg move after low  break : {st.avg_move_after_low_break:>6.2f} pts  "
              f"({st.avg_move_after_low_break / st.avg_or_size * 100:.0f}% of avg OR)")
        print(f"\n    Continuation after HIGH break (% of break days):")
        print(f"      0.5R={st.cont_high_0_5r:>5.1f}%  1R={st.cont_high_1r:>5.1f}%  "
              f"2R={st.cont_high_2r:>5.1f}%  3R={st.cont_high_3r:>5.1f}%")
        print(f"    Continuation after LOW  break (% of break days):")
        print(f"      0.5R={st.cont_low_0_5r:>5.1f}%  1R={st.cont_low_1r:>5.1f}%  "
              f"2R={st.cont_low_2r:>5.1f}%  3R={st.cont_low_3r:>5.1f}%")
        print(f"\n    Failure  (closed back inside after break): "
              f"HIGH={st.fail_high_rate:.1f}%  LOW={st.fail_low_rate:.1f}%")
        print(f"    Reversal (reached opposite OR side after break): "
              f"high→low={st.rev_high_to_low:.1f}%  low→high={st.rev_low_to_high:.1f}%")

    # ── Trade simulation matrix ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  TRADE SIMULATION — all (stop method × target R) combinations")
    print(f"{'='*72}")

    for sm in cfg.stop_methods:
        print(f"\n  Stop method: {sm}")
        for tr in cfg.targets_r:
            combo_trades = trades_by_combo.get((sm, tr), [])
            longs  = [t for t in combo_trades if t.direction == "long"]
            shorts = [t for t in combo_trades if t.direction == "short"]
            print(f"\n  ── Target {tr}R ──────────────────────────────────────────────────")
            print(f"  Overall  :{_combo_summary(combo_trades, cfg.contracts, cfg.dollars_per_point)}")
            print(f"  Longs    :{_combo_summary(longs,        cfg.contracts, cfg.dollars_per_point)}")
            print(f"  Shorts   :{_combo_summary(shorts,       cfg.contracts, cfg.dollars_per_point)}")
            if combo_trades:
                print("  By session:")
                print(_session_breakdown(combo_trades))

    print()
