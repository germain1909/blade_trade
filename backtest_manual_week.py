#!/usr/bin/env python3
"""
Manual Weekly-Level Backtest — one week at a time

You supply the weekly S/R levels by hand (no auto-computed pivots). For
each level, price during the week can do one of three things:

  - CROSS      price closes through the level by `cross_confirm_ticks`.
  - APPROACH   price enters the watch zone near the level but hasn't
               resolved yet -> not a trade by itself, just arms the level.
  - FAIL       price gets close (or even touches) the level, then closes
               back away by `fail_confirm_ticks` without ever crossing.

A cross or fail enters immediately, but ONLY if the 15m MACD line has
rounded over the matching way (oversold + curling up for longs, overbought
+ curling down for shorts) at any point while that level was being
watched (see `mtf_indicators.macd_line_shape`) — set `require_macd_turn`
to False to disable this and take every cross/fail unconditionally.

Every trade gets an auto-tagged snapshot of 15m MACD, 60m RSI, 4hr MACD,
and 4hr RSI at entry, so you can review the journal afterward and see
which indicator patterns show up around your wins vs. losses.

Usage:
    python3 backtest_manual_week.py --file data/NQZ5_1m.csv \\
        --week-start 2025-11-17 --week-end 2025-11-21 \\
        --level "R1:25776.33" --level "S1:24572.33" --level "PWH:25830.00"
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from mtf_indicators import compute_mtf_features, tags_for_row
from backtest_weekly_swing_fade import INSTRUMENT_SPECS
from market_snapshot import MarketSnapshot, build_snapshot
from decision_strategy import Decision, Outcome, RuleBasedStrategy, Strategy

ET = ZoneInfo("America/New_York")


@dataclass
class Config:
    approach_ticks:       int   = 20    # outer watch-zone distance from the level
    fail_confirm_ticks:   int   = 6     # close-back distance beyond approach zone to confirm a failed reach
    cross_confirm_ticks:  int   = 4     # close-through distance beyond the level to confirm a breakout
    max_watch_bars:        int   = 120   # bars a level can stay "approaching" before it resets
    require_macd_turn:    bool  = True  # 15m MACD must round over the entry's direction while watching
    cooldown_hours:       float = 4.0   # a level can't re-arm for this long after a trade closes there
    stop_buffer_ticks:    int   = 8
    target_r_multiple:    float = 1.5
    contracts:            int   = 1
    tick_size:            float = 0.25
    dollars_per_point:    float = 20.0
    slippage_ticks:       int   = 1
    max_risk_usd:         float = 500.0
    trade_start_et:       dtime = dtime(0, 0)
    trade_end_et:         dtime = dtime(23, 59)
    symbol_label:         str   = "UNKNOWN"  # for MarketSnapshot.symbol / experiment records


@dataclass
class Level:
    name:  str
    price: float


@dataclass
class Snapshot:
    timestamp:   datetime
    close:       float
    trade_desc:  str
    level_lines: list[str]
    tags:        list[str]


@dataclass
class Trade:
    level:       str
    level_price: float
    setup:       str      # "cross" | "fail"
    direction:   str
    entry_time:  datetime
    entry_px:    float
    stop_px:     float
    target_px:   float
    exit_time:   Optional[datetime]
    exit_px:     float
    exit_reason: str
    pnl_usd:     float
    pnl_r:       float
    tags:        list[str] = field(default_factory=list)
    mfe:         float = 0.0   # max favorable excursion in points, tracked from entry to exit
    mae:         float = 0.0   # max adverse excursion in points


class LevelWatcher:
    """Per-level state machine: NOT_NEAR -> APPROACHING -> cross/fail enters immediately,
    gated on the 15m MACD line having rounded over the matching direction at some point
    during the approach (see `require_macd_turn` / `mtf_indicators.macd_line_shape`)."""
    def __init__(self, level: Level):
        self.level = level
        self.state = "NOT_NEAR"
        self.side: Optional[str] = None       # 'above' | 'below' — which side price approached from
        self.bars_watching = 0
        self.extreme: Optional[float] = None  # closest approach (min distance point) for the stop
        self.cooldown_until: Optional[datetime] = None

        # Accumulated anytime during APPROACHING — "have we seen the MACD line round
        # over our way at some point while this level was being watched?"
        self.seen_macd_bottom = False
        self.seen_macd_top = False


def _level_line(name: str, w: "LevelWatcher", price: float, tick: float) -> str:
    dist_pts = price - w.level.price
    if w.state == "NOT_NEAR":
        return f"{name} ({w.level.price:.2f}): flat — {abs(dist_pts):.2f} pts away"
    fail_dir, cross_dir = ("LONG", "SHORT") if w.side == "above" else ("SHORT", "LONG")
    macd_bits = []
    if w.seen_macd_bottom:
        macd_bits.append("MACD bottomed (long-ready)")
    if w.seen_macd_top:
        macd_bits.append("MACD topped (short-ready)")
    macd_str = f" — {', '.join(macd_bits)}" if macd_bits else " — no MACD turn seen yet"
    return (f"{name} ({w.level.price:.2f}): approaching from {w.side} "
            f"({abs(dist_pts):.2f} pts away) — IF it fails here we'd go {fail_dir}, "
            f"IF it breaks through we'd go {cross_dir}{macd_str}")


def _simulate_forward(
    bars_list: list, start_idx: int, direction: str,
    entry_px: float, stop_px: float, target_px: float,
) -> tuple[datetime, float, str, float, float]:
    """Walk bars_list[start_idx+1:] to find where this trade would have
    exited had it been taken, plus max favorable/adverse excursion (points)
    along the way. Used both to score SKIPped candidates ("what would have
    happened anyway") and could equally re-derive a TAKEn trade's exit —
    the exit rule here (stop wins on same-bar collision) matches the live
    position-tracking loop in run_week() exactly.

    Falls back to the last available bar's close ("session_end") if neither
    level is hit before the data runs out.
    """
    mfe = 0.0
    mae = 0.0
    for row in bars_list[start_idx + 1:]:
        if direction == "long":
            mfe = max(mfe, row.high - entry_px)
            mae = max(mae, entry_px - row.low)
            hit_tp = row.high >= target_px
            hit_sl = row.low <= stop_px
        else:
            mfe = max(mfe, entry_px - row.low)
            mae = max(mae, row.high - entry_px)
            hit_tp = row.low <= target_px
            hit_sl = row.high >= stop_px
        if hit_tp or hit_sl:
            exit_px, reason = (stop_px, "stop") if hit_sl else (target_px, "target")
            return row.timestamp, exit_px, reason, mfe, mae

    last = bars_list[-1]
    return last.timestamp, last.close, "session_end", mfe, mae


def run_week(
    week_bars: pd.DataFrame, features: pd.DataFrame,
    levels: list[Level], cfg: Config,
    strategy: Optional[Strategy] = None,
) -> tuple[list[Trade], list[Snapshot], list[tuple[MarketSnapshot, Decision, Outcome]]]:
    """
    strategy: judges each cross/fail candidate as TAKE or SKIP. Defaults to
    RuleBasedStrategy(require_macd_turn=cfg.require_macd_turn) — the original
    boolean MACD gate — so callers that don't pass a strategy get exactly the
    same behavior as before this parameter existed. Pass an LLMStrategy to
    run the same candidate stream through a local LLM instead (see
    run_experiment.py) — every SKIPped candidate still gets a forward-simulated
    hypothetical outcome so both experiments are comparable.
    """
    if strategy is None:
        strategy = RuleBasedStrategy(require_macd_turn=cfg.require_macd_turn)

    tick = cfg.tick_size
    approach_dist = cfg.approach_ticks * tick
    fail_dist     = cfg.fail_confirm_ticks * tick
    cross_dist    = cfg.cross_confirm_ticks * tick
    sbuf          = cfg.stop_buffer_ticks * tick
    max_stop_pts  = cfg.max_risk_usd / (cfg.dollars_per_point * cfg.contracts)

    levels_map = {lv.name: lv.price for lv in levels}
    watchers = {lv.name: LevelWatcher(lv) for lv in levels}
    trades: list[Trade] = []
    snapshots: list[Snapshot] = []
    decisions: list[tuple[MarketSnapshot, Decision, Outcome]] = []

    bars_list = list(week_bars.itertuples())

    in_trade = False
    trade_dir = ""
    entry_px = stop_px = target_px = 0.0
    entry_time: Optional[datetime] = None
    active_level: Optional[Level] = None
    active_setup = ""
    active_tags: list[str] = []
    cur_mfe = 0.0
    cur_mae = 0.0

    for idx, row in enumerate(bars_list):
        ts_et = row.timestamp.astimezone(ET)
        bar_time = ts_et.time()

        # ── Manage open trade ────────────────────────────────────────
        if in_trade:
            if trade_dir == "long":
                cur_mfe = max(cur_mfe, row.high - entry_px)
                cur_mae = max(cur_mae, entry_px - row.low)
            else:
                cur_mfe = max(cur_mfe, entry_px - row.low)
                cur_mae = max(cur_mae, row.high - entry_px)

            risk = abs(entry_px - stop_px)
            hit_tp = (row.high >= target_px) if trade_dir == "long" else (row.low <= target_px)
            hit_sl = (row.low <= stop_px)     if trade_dir == "long" else (row.high >= stop_px)
            if hit_tp or hit_sl:
                ep, reason = (stop_px, "stop") if hit_sl else (target_px, "target")
                pnl = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) \
                      * cfg.dollars_per_point * cfg.contracts
                r = ((ep - entry_px) if trade_dir == "long" else (entry_px - ep)) / risk if risk else 0.0
                trades.append(Trade(
                    level=active_level.name, level_price=active_level.price, setup=active_setup,
                    direction=trade_dir, entry_time=entry_time, entry_px=entry_px,
                    stop_px=stop_px, target_px=target_px, exit_time=row.timestamp,
                    exit_px=ep, exit_reason=reason, pnl_usd=pnl, pnl_r=r, tags=active_tags,
                    mfe=cur_mfe, mae=cur_mae,
                ))
                in_trade = False
                watchers[active_level.name].cooldown_until = \
                    row.timestamp + pd.Timedelta(hours=cfg.cooldown_hours)

        if not (cfg.trade_start_et <= bar_time <= cfg.trade_end_et):
            continue

        # ── Update each level's watcher ──────────────────────────────
        for name, w in watchers.items():
            dist = row.close - w.level.price
            adist = abs(dist)

            if w.state == "NOT_NEAR":
                if w.cooldown_until is not None and row.timestamp < w.cooldown_until:
                    continue
                if adist <= approach_dist:
                    w.state = "APPROACHING"
                    w.side = "above" if dist > 0 else "below"
                    w.bars_watching = 0
                    w.extreme = row.close
                    w.seen_macd_bottom = False
                    w.seen_macd_top = False
                continue

            # APPROACHING
            w.bars_watching += 1
            w.extreme = min(w.extreme, row.low) if w.side == "above" else max(w.extreme, row.high)

            feat_row = features.loc[row.timestamp] if row.timestamp in features.index else None
            if feat_row is not None:
                # mirror the same "recent" window the entry check uses, so the plan
                # narration always agrees with what would actually trigger a trade
                w.seen_macd_bottom = bool(feat_row.get("15m_macd_bottom_recent"))
                w.seen_macd_top    = bool(feat_row.get("15m_macd_top_recent"))

            if adist > approach_dist * 2:
                # price left the neighborhood without resolving -> reset, can re-arm later
                w.state = "NOT_NEAR"
                continue

            if w.bars_watching > cfg.max_watch_bars:
                w.state = "NOT_NEAR"
                continue

            if in_trade:
                continue

            crossed = (row.close <= w.level.price - cross_dist) if w.side == "above" \
                      else (row.close >= w.level.price + cross_dist)
            failed  = (row.close >= w.level.price + fail_dist) if w.side == "above" \
                      else (row.close <= w.level.price - fail_dist)
            if not (crossed or failed):
                continue

            setup = "cross" if crossed else "fail"
            direction = ("short" if w.side == "above" else "long") if crossed \
                        else ("long" if w.side == "above" else "short")

            # Bracket math is computed up front regardless of TAKE/SKIP — a SKIPped
            # candidate still needs an entry/stop/target to score "what would have
            # happened anyway" against.
            if direction == "long":
                ep = row.close + tick * cfg.slippage_ticks
                stp = (w.level.price - sbuf) if setup == "cross" else (w.extreme - sbuf)
                stp = min(stp, ep - tick)
                stp = max(stp, ep - max_stop_pts)
            else:
                ep = row.close - tick * cfg.slippage_ticks
                stp = (w.level.price + sbuf) if setup == "cross" else (w.extreme + sbuf)
                stp = max(stp, ep + tick)
                stp = min(stp, ep + max_stop_pts)

            risk = abs(ep - stp)
            if risk <= 0:
                w.state = "NOT_NEAR"
                continue

            tgt = ep + risk * cfg.target_r_multiple if direction == "long" \
                  else ep - risk * cfg.target_r_multiple
            cand_tags = tags_for_row(features.loc[row.timestamp]) if row.timestamp in features.index else []

            snapshot = build_snapshot(
                timestamp=row.timestamp, symbol=cfg.symbol_label, price=row.close,
                bars_df=week_bars, feat_row=feat_row, levels=levels_map,
                level_name=w.level.name, setup_type=setup, setup_direction=direction,
                tags=tuple(cand_tags),
            )
            decision = strategy.decide(snapshot)

            # Forward-simulate the outcome regardless of TAKE/SKIP — for TAKE this
            # mirrors exactly what the live "Manage open trade" loop above will
            # independently compute for the real Trade record (same bracket, same
            # stop-wins-on-collision rule), just used here for the decisions log;
            # for SKIP it's the only record of what would have happened.
            exit_time, exit_px, exit_reason, mfe, mae = _simulate_forward(
                bars_list, idx, direction, ep, stp, tgt,
            )
            pnl = ((exit_px - ep) if direction == "long" else (ep - exit_px)) \
                  * cfg.dollars_per_point * cfg.contracts
            pnl_r = ((exit_px - ep) if direction == "long" else (ep - exit_px)) / risk if risk else 0.0
            outcome = Outcome(
                entry_px=ep, stop_px=stp, target_px=tgt,
                exit_time=exit_time, exit_px=exit_px, exit_reason=exit_reason,
                pnl_usd=round(pnl, 2), pnl_r=round(pnl_r, 2),
                mfe=round(mfe, 4), mae=round(mae, 4),
                outcome_type="actual" if decision.action == "TAKE" else "hypothetical",
            )
            decisions.append((snapshot, decision, outcome))

            if decision.action == "TAKE":
                in_trade = True
                trade_dir = direction
                entry_px, stop_px, target_px = ep, stp, tgt
                entry_time = row.timestamp
                active_level = w.level
                active_setup = setup
                active_tags = cand_tags
                cur_mfe = 0.0
                cur_mae = 0.0
                w.state = "NOT_NEAR"
            # else: SKIP — leave the watcher APPROACHING so it re-checks next bar,
            # matching the original inline-continue behavior of the boolean MACD
            # gate exactly (a SKIP doesn't retire the level).

        # ── 15-minute plan snapshot ──────────────────────────────────
        if row.timestamp.minute % 15 == 0:
            if in_trade:
                trade_desc = (f"IN TRADE: {trade_dir} from {entry_px:.2f} "
                              f"(level {active_level.name}, {active_setup}) — "
                              f"stop {stop_px:.2f}, target {target_px:.2f}")
            else:
                trade_desc = "no open position"
            tags = tags_for_row(features.loc[row.timestamp]) if row.timestamp in features.index else []
            level_lines = [_level_line(name, w, row.close, tick) for name, w in watchers.items()]
            snapshots.append(Snapshot(
                timestamp=row.timestamp, close=row.close, trade_desc=trade_desc,
                level_lines=level_lines, tags=tags,
            ))

    return trades, snapshots, decisions


def print_plan(snapshots: list[Snapshot], symbol: str) -> None:
    print(f"\n{'='*100}")
    print(f"  {symbol} — 15-Minute Trading Plan  ({len(snapshots)} snapshots)")
    print(f"{'='*100}")
    for s in snapshots:
        et = s.timestamp.astimezone(ET).strftime("%a %Y-%m-%d %H:%M ET")
        print(f"\n[{et}]  close={s.close:.2f}  {s.trade_desc}")
        for line in s.level_lines:
            print(f"    {line}")
        print(f"    tags: {', '.join(s.tags) if s.tags else '(none)'}")


def print_journal(trades: list[Trade], symbol: str) -> None:
    if not trades:
        print("\nNo trades generated.")
        return

    print(f"\n{'='*100}")
    print(f"  {symbol} — Manual Weekly-Level Trade Journal  ({len(trades)} trades)")
    print(f"{'='*100}")
    for t in trades:
        win = "WIN " if t.pnl_usd > 0 else "LOSS"
        et_entry = t.entry_time.astimezone(ET).strftime("%a %H:%M ET")
        et_exit  = t.exit_time.astimezone(ET).strftime("%H:%M ET") if t.exit_time else "?"
        print(f"\n  [{win}] {t.level} ({t.level_price:.2f})  {t.setup:<5} {t.direction:<5}  "
              f"{et_entry} -> {et_exit}")
        print(f"    entry={t.entry_px:.2f}  stop={t.stop_px:.2f}  target={t.target_px:.2f}  "
              f"exit={t.exit_px:.2f} ({t.exit_reason})  pnl=${t.pnl_usd:+.2f}  {t.pnl_r:+.2f}R")
        print(f"    tags: {', '.join(t.tags) if t.tags else '(none)'}")

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total = sum(t.pnl_usd for t in trades)
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl else float("inf")

    print(f"\n{'-'*100}")
    print(f"  Trades: {len(trades)}  ({len(wins)}W / {len(losses)}L)   "
          f"Win rate: {len(wins)/len(trades)*100:.1f}%   PnL: ${total:+,.2f}   PF: {pf:.2f}")

    print(f"\n  By setup type:")
    for setup in ["cross", "fail"]:
        grp = [t for t in trades if t.setup == setup]
        if not grp:
            continue
        w = sum(1 for t in grp if t.pnl_usd > 0)
        print(f"    {setup:<6} {len(grp):>3} trades  {w}W/{len(grp)-w}L  ${sum(t.pnl_usd for t in grp):>+10,.2f}")

    tag_pnl: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        for tag in t.tags:
            tag_pnl[tag].append(t.pnl_usd)
    if tag_pnl:
        print(f"\n  By tag present at entry:")
        for tag, pnls in sorted(tag_pnl.items(), key=lambda kv: -sum(kv[1])):
            w = sum(1 for p in pnls if p > 0)
            print(f"    {tag:<24} {len(pnls):>3} trades  {w}W/{len(pnls)-w}L  ${sum(pnls):>+10,.2f}")


def export_csv(trades: list[Trade], path: Path) -> None:
    rows = []
    for t in trades:
        rows.append(dict(
            level=t.level, level_price=t.level_price, setup=t.setup, direction=t.direction,
            entry_time=t.entry_time, entry_px=t.entry_px, stop_px=t.stop_px, target_px=t.target_px,
            exit_time=t.exit_time, exit_px=t.exit_px, exit_reason=t.exit_reason,
            pnl_usd=round(t.pnl_usd, 2), pnl_r=round(t.pnl_r, 2), tags="|".join(t.tags),
        ))
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nJournal exported -> {path}")


def parse_level(spec: str) -> Level:
    name, price = spec.split(":")
    return Level(name=name.strip(), price=float(price))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="1m bar CSV/xlsx to load (needs history before the week for MTF indicator warmup)")
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--week-start", required=True, help="YYYY-MM-DD (Monday, ET session date)")
    ap.add_argument("--week-end", required=True, help="YYYY-MM-DD (Friday, ET session date)")
    ap.add_argument("--level", action="append", required=True, dest="levels",
                    help='Level as "NAME:PRICE", repeatable, e.g. --level "R1:25776.33"')
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--cooldown-hours", type=float, default=4.0, dest="cooldown_hours",
                    help="Hours a level must sit idle after a trade closes before it can re-arm (default 4)")
    ap.add_argument("--csv", action="store_true", help="Export the trade journal to CSV")
    ap.add_argument("--plan", action="store_true",
                    help="Print the 15-minute trading plan narration alongside the trade journal")
    ap.add_argument("--no-macd-turn", action="store_true", dest="no_macd_turn",
                    help="Disable the 15m MACD cross requirement on entries (default: required)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    spec = INSTRUMENT_SPECS.get(args.symbol.upper(), INSTRUMENT_SPECS["NQ"])
    cfg = Config(contracts=args.contracts, tick_size=spec["tick_size"],
                 dollars_per_point=spec["dollars_per_point"], cooldown_hours=args.cooldown_hours,
                 require_macd_turn=not args.no_macd_turn, symbol_label=args.symbol.upper())
    levels = [parse_level(s) for s in args.levels]

    print(f"Loading {path} ...")
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} bars  ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")

    week_start = pd.Timestamp(args.week_start, tz=ET).normalize()
    week_end   = (pd.Timestamp(args.week_end, tz=ET) + pd.Timedelta(days=1)).normalize()

    print("Computing 15m MACD / 60m RSI / 4hr MACD+RSI ...")
    features = compute_mtf_features(df)

    et_ts = df["timestamp"].dt.tz_convert(ET)
    mask = (et_ts >= week_start) & (et_ts < week_end)
    week_bars = df.loc[mask].sort_values("timestamp")
    if week_bars.empty:
        print("No bars found in that week range — check --week-start/--week-end against the file's coverage.")
        return
    print(f"Week bars: {len(week_bars):,}  ({week_bars['timestamp'].iloc[0]} -> {week_bars['timestamp'].iloc[-1]})")
    print(f"Levels: {', '.join(f'{l.name}={l.price:.2f}' for l in levels)}")

    trades, snapshots, decisions = run_week(week_bars, features, levels, cfg)
    if args.plan:
        print_plan(snapshots, args.symbol.upper())
    print_journal(trades, args.symbol.upper())

    if args.csv:
        export_csv(trades, Path("data") / f"{args.symbol.lower()}_manual_week_{args.week_start}.csv")


if __name__ == "__main__":
    main()
