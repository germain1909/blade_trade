#!/usr/bin/env python3
"""
Weekly floor-pivot levels from 1-minute bar data.

For each trading week, builds a weekly OHLC bar (Globex week: Sunday 6pm ET
open -> Friday 5pm ET close) then derives classic floor-trader pivots from
the PRIOR week's High/Low/Close. Those pivots are the S/R levels traders
watch for the week ahead:

    P  = (H + L + C) / 3
    R1 = 2P - L        S1 = 2P - H
    R2 = P + (H - L)   S2 = P - (H - L)
    R3 = H + 2(P - L)  S3 = L - 2(H - P)

Usage:
    python3 weekly_levels.py                          # data/NQZ5_1m.csv
    python3 weekly_levels.py --file data/GCZ5_1m.csv
    python3 weekly_levels.py --csv                     # also export CSV
"""

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


@dataclass
class WeekBar:
    week_start: date   # Monday of the trading week
    week_end:   date   # Friday of the trading week
    open:  float
    high:  float
    low:   float
    close: float
    volume: int
    bar_count: int


@dataclass
class WeeklyPivots:
    week_start: date            # week these levels apply to
    prior_week_start: date      # source week for H/L/C
    prior_high:  float
    prior_low:   float
    prior_close: float
    p:  float
    r1: float; r2: float; r3: float
    s1: float; s2: float; s3: float
    prior_bar_count: int = 0
    reliable: bool = True


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _session_date(ts_et: pd.Timestamp) -> date:
    """Globex session date — bars at/after 5pm ET belong to the next calendar day."""
    d = ts_et.date()
    return d + timedelta(days=1) if ts_et.hour >= 17 else d


FULL_WEEK_BARS = 6900   # 5 sessions x 23h x 60min, roughly


def build_weekly_bars(df: pd.DataFrame) -> list[WeekBar]:
    et = df["timestamp"].dt.tz_convert(ET)
    session_date = et.apply(_session_date)
    iso = pd.DatetimeIndex(pd.to_datetime(session_date)).isocalendar()
    week_key = list(zip(iso["year"], iso["week"]))

    work = df.copy()
    work["session_date"] = session_date
    work["week_key"] = week_key

    weeks: list[WeekBar] = []
    for _, grp in work.groupby("week_key", sort=False):
        grp = grp.sort_values("timestamp")
        dates = sorted(grp["session_date"].unique())
        weeks.append(WeekBar(
            week_start = dates[0],
            week_end   = dates[-1],
            open   = float(grp["open"].iloc[0]),
            high   = float(grp["high"].max()),
            low    = float(grp["low"].min()),
            close  = float(grp["close"].iloc[-1]),
            volume = int(grp["volume"].sum()),
            bar_count = len(grp),
        ))

    return sorted(weeks, key=lambda w: w.week_start)


def compute_pivots(weeks: list[WeekBar]) -> list[WeeklyPivots]:
    out = []
    for prev, cur in zip(weeks, weeks[1:]):
        h, l, c = prev.high, prev.low, prev.close
        p = (h + l + c) / 3
        r1 = 2 * p - l;         s1 = 2 * p - h
        r2 = p + (h - l);       s2 = p - (h - l)
        r3 = h + 2 * (p - l);   s3 = l - 2 * (h - p)
        out.append(WeeklyPivots(
            week_start=cur.week_start, prior_week_start=prev.week_start,
            prior_high=h, prior_low=l, prior_close=c,
            p=p, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3,
            prior_bar_count=prev.bar_count,
            reliable=prev.bar_count >= FULL_WEEK_BARS * 0.5,
        ))
    return out


def print_table(pivots: list[WeeklyPivots], label: str) -> None:
    n_unreliable = sum(1 for pv in pivots if not pv.reliable)
    print(f"\n{label} — Weekly Floor Pivots  ({len(pivots)} weeks)")
    if n_unreliable:
        print(f"  ⚠ {n_unreliable} week(s) marked with * are built from a prior week with "
              f"< {FULL_WEEK_BARS // 2:,} bars (gappy source data) — treat those levels as unreliable.")
    print("=" * 100)
    header = (f"{'Week of':<13} {'PrevH':>10} {'PrevL':>10} {'PrevC':>10}  |"
              f" {'S3':>9} {'S2':>9} {'S1':>9} {'P':>9} {'R1':>9} {'R2':>9} {'R3':>9}")
    print(header)
    print("-" * len(header))
    for pv in pivots:
        flag = "*" if not pv.reliable else " "
        print(f"{pv.week_start.isoformat():<12}{flag}{pv.prior_high:>10.2f} {pv.prior_low:>10.2f} "
              f"{pv.prior_close:>10.2f}  |"
              f" {pv.s3:>9.2f} {pv.s2:>9.2f} {pv.s1:>9.2f} {pv.p:>9.2f} "
              f"{pv.r1:>9.2f} {pv.r2:>9.2f} {pv.r3:>9.2f}")


def print_current(pivots: list[WeeklyPivots]) -> None:
    if not pivots:
        return
    pv = pivots[-1]
    print(f"\n{'─'*60}")
    print(f"  CURRENT LEVELS — week of {pv.week_start.isoformat()}")
    print(f"  (derived from prior week {pv.prior_week_start.isoformat()}: "
          f"H={pv.prior_high:.2f} L={pv.prior_low:.2f} C={pv.prior_close:.2f}, "
          f"{pv.prior_bar_count:,} bars{'' if pv.reliable else '  ⚠ SPARSE — unreliable'})")
    print(f"{'─'*60}")
    for name, val in [("R3", pv.r3), ("R2", pv.r2), ("R1", pv.r1),
                       ("P ", pv.p), ("S1", pv.s1), ("S2", pv.s2), ("S3", pv.s3)]:
        print(f"    {name}  {val:>10.2f}")


def export_csv(pivots: list[WeeklyPivots], path: Path) -> None:
    rows = [dict(week_start=pv.week_start, prior_week_start=pv.prior_week_start,
                 prior_high=pv.prior_high, prior_low=pv.prior_low, prior_close=pv.prior_close,
                 prior_bar_count=pv.prior_bar_count, reliable=pv.reliable,
                 s3=pv.s3, s2=pv.s2, s1=pv.s1, p=pv.p, r1=pv.r1, r2=pv.r2, r3=pv.r3)
            for pv in pivots]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nExported {len(rows)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/NQZ5_1m.csv", help="1m bar CSV to load")
    ap.add_argument("--csv", action="store_true", help="Export pivot table to CSV")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    label = path.stem.replace("_1m", "")
    print(f"Loading {path} ...")
    df = load_bars(path)
    print(f"Loaded {len(df):,} bars  ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")

    weeks = build_weekly_bars(df)
    print(f"Built {len(weeks)} weekly bars")

    pivots = compute_pivots(weeks)
    print_table(pivots, label)
    print_current(pivots)

    if args.csv:
        export_csv(pivots, Path("data") / f"{label}_weekly_pivots.csv")


if __name__ == "__main__":
    main()
