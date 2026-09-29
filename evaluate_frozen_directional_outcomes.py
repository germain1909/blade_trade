#!/usr/bin/env python3
"""
Outcome evaluation for the 12 FROZEN Phase 2 pilot predictions — read-only
with respect to everything already generated. Does not call the LLM, does
not touch `analyses` or `diagnostic_forced_direction`, does not modify any
snapshot. Reads episode_id/timestamp/price/original-decision/forced-
direction straight out of the existing DB rows and computes directional
MFE/MAE/net-change purely from the historical 1-minute bars.

This is NOT a strategy backtest — there is no entry/stop/target/sizing.
It measures whether the already-frozen directional read (from the
diagnostic forced-direction call, made before any of this was computed)
correlates with subsequent directional price movement at all.

Output: data/phase2_frozen_outcomes.json — a separate, clearly-labeled
research-only file. Never merged back into OpportunitySnapshot/MarketSnapshot.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = "data/experiments.db"
BARS_PATH = "data/NQZ5_1m.csv"
OUT_PATH = "data/phase2_frozen_outcomes.json"

HORIZONS = (30, 60, 120)
FLAT_THRESHOLD_PTS = 2.0  # |net_change| at or below this counts as "approximately flat"


def _directional_outcome(bars_list: list, start_idx: int, episode_price: float,
                          minutes: int, direction: str) -> dict:
    start_ts = bars_list[start_idx].timestamp
    cutoff = start_ts + pd.Timedelta(minutes=minutes)
    favorable = 0.0
    adverse = 0.0
    last_close = episode_price
    bars_seen = 0
    for row in bars_list[start_idx + 1:]:
        if row.timestamp > cutoff:
            break
        if direction == "LONG":
            favorable = max(favorable, row.high - episode_price)
            adverse = max(adverse, episode_price - row.low)
        else:
            favorable = max(favorable, episode_price - row.low)
            adverse = max(adverse, row.high - episode_price)
        last_close = row.close
        bars_seen += 1
    net_change = (last_close - episode_price) if direction == "LONG" else (episode_price - last_close)

    if adverse == 0 and favorable == 0:
        ratio, ratio_note = None, "no movement (MFE=MAE=0)"
    elif adverse == 0:
        ratio, ratio_note = None, "undefined (MAE=0)"
    else:
        ratio, ratio_note = round(favorable / adverse, 2), None

    return {
        "mfe": round(favorable, 2), "mae": round(adverse, 2), "net_change": round(net_change, 2),
        "mfe_mae_ratio": ratio, "mfe_mae_ratio_note": ratio_note, "bars_available": bars_seen,
    }


def _time_to_move(bars_list: list, start_idx: int, episode_price: float, direction: str,
                   max_minutes: int = 120) -> dict:
    start_ts = bars_list[start_idx].timestamp
    cutoff = start_ts + pd.Timedelta(minutes=max_minutes)
    result = {"time_to_+25": None, "time_to_+50": None, "time_to_-25": None, "time_to_-50": None}
    for row in bars_list[start_idx + 1:]:
        if row.timestamp > cutoff:
            break
        elapsed = (row.timestamp - start_ts).total_seconds() / 60.0
        if direction == "LONG":
            fav = row.high - episode_price
            adv = episode_price - row.low
        else:
            fav = episode_price - row.low
            adv = row.high - episode_price
        if result["time_to_+25"] is None and fav >= 25:
            result["time_to_+25"] = round(elapsed, 1)
        if result["time_to_+50"] is None and fav >= 50:
            result["time_to_+50"] = round(elapsed, 1)
        if result["time_to_-25"] is None and adv >= 25:
            result["time_to_-25"] = round(elapsed, 1)
        if result["time_to_-50"] is None and adv >= 50:
            result["time_to_-50"] = round(elapsed, 1)
    return result


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    analyses = {r["episode_id"]: r for r in conn.execute("SELECT * FROM analyses ORDER BY timestamp")}
    diagnostics = {r["episode_id"]: r for r in conn.execute(
        "SELECT * FROM diagnostic_forced_direction ORDER BY timestamp")}
    conn.close()

    print(f"Loading {BARS_PATH} for outcome computation only (read-only, no rescan)...")
    df = pd.read_csv(BARS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    bars_list = list(df.itertuples())
    ts_to_idx = {row.timestamp: i for i, row in enumerate(bars_list)}

    episodes = []
    for episode_id, a in analyses.items():
        diag = diagnostics.get(episode_id)
        if diag is None:
            continue
        ts = pd.Timestamp(a["timestamp"])
        if ts not in ts_to_idx:
            raise AssertionError(f"episode {episode_id} timestamp {ts} not found in {BARS_PATH} — "
                                  f"frozen episode data does not match the bar file being evaluated")
        idx = ts_to_idx[ts]
        snap = json.loads(a["snapshot_json"])
        episode_price = snap["price"]
        direction = diag["forced_direction"]

        outcomes = {f"{h}m": _directional_outcome(bars_list, idx, episode_price, h, direction) for h in HORIZONS}
        time_to_move = _time_to_move(bars_list, idx, episode_price, direction, max_minutes=120)

        episodes.append({
            "episode_id": episode_id,
            "timestamp": a["timestamp"],
            "sessions": snap["sessions"],
            "scanner_categories": snap["categories"],
            "episode_price": episode_price,
            "original_decision": a["decision"],
            "original_confidence": a["confidence"],
            "frozen_forced_direction": direction,
            "directional_confidence": diag["directional_confidence"],
            "outcomes": outcomes,
            "time_to_move_120m": time_to_move,
        })

    out = {
        "meta": {
            "note": "RESEARCH ONLY. Predictions (original decision + forced direction) were frozen and "
                    "persisted BEFORE this file was generated. Not a strategy backtest — no entry/stop/"
                    "target/sizing exists. Never merge this into OpportunitySnapshot/MarketSnapshot.",
            "flat_threshold_pts": FLAT_THRESHOLD_PTS,
            "horizons_minutes": list(HORIZONS),
            "episode_count": len(episodes),
            "bars_source": BARS_PATH,
        },
        "episodes": episodes,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
