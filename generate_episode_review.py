#!/usr/bin/env python3
"""
Generates a full point-in-time review of every is_new_episode=True scanner
event, for human review before Phase 2. Two strictly separated outputs per
episode:

  1. POINT-IN-TIME CONTEXT — everything that would be available to a
     QuantHarness-derived MarketSnapshot at that exact timestamp: price,
     session(s), trigger categories/reasoning, every tracked level's state
     and distance, confluence zones, VWAP + distance, raw 15m MACD/60m RSI/
     4h MACD/4h RSI readings, and volatility (ATR + percentile).

  2. FORWARD EXCURSION (RESEARCH ONLY) — what happened in the 30/60/120
     minutes AFTER the episode, in both directions (no trade direction is
     assigned yet in Phase 1, so both up-move and down-move are reported).
     This is computed and stored ONLY for our own judgment of whether the
     scanner is firing at structurally reasonable moments. It is clearly
     tagged as such in the output and MUST NOT be read by anything that
     builds a MarketSnapshot or feeds QuantHarness/an LLM.

Does not change any scanner rule or threshold based on what this finds
(session_tracker.SCANNER_CONFIG_VERSION stays "v1" regardless).

Usage:
    python3 generate_episode_review.py --file data/NQZ5_1m.csv --symbol NQ \\
        --start 2025-09-22 --end 2025-12-19 --out data/episode_review.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from mtf_indicators import compute_mtf_features
from opportunity_scanner import compute_scanner_features, scan
from session_config import SessionConfig
from session_tracker import SCANNER_CONFIG_VERSION, ScannerConfig, SessionTracker
from weekly_levels import build_weekly_bars, compute_pivots

ET = ZoneInfo("America/New_York")


def _forward_excursion(bars_list: list, start_idx: int, price: float,
                        minutes: int) -> dict:
    """RESEARCH ONLY. Max up-move and down-move (points) from `price` within
    `minutes` after bars_list[start_idx]. Direction-agnostic — Phase 1's
    scanner doesn't assign LONG/SHORT, so both are reported."""
    start_ts = bars_list[start_idx].timestamp
    cutoff = start_ts + pd.Timedelta(minutes=minutes)
    up = 0.0
    down = 0.0
    bars_seen = 0
    for row in bars_list[start_idx + 1:]:
        if row.timestamp > cutoff:
            break
        up = max(up, row.high - price)
        down = max(down, price - row.low)
        bars_seen += 1
    return {"up_pts": round(up, 2), "down_pts": round(down, 2), "bars_available": bars_seen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/episode_review.json")
    args = ap.parse_args()

    path = Path(args.file)
    print(f"Loading {path} ...")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Computing 15m MACD / 60m RSI / 4hr MACD+RSI ...")
    features = compute_mtf_features(df)
    print("Computing ATR / volatility percentile / daily ATR ...")
    scanner_features = compute_scanner_features(df)

    print("Computing weekly floor pivots ...")
    weeks = build_weekly_bars(df)
    weekly_pivots = compute_pivots(weeks)

    session_cfg = SessionConfig()
    scanner_cfg = ScannerConfig()
    assert scanner_cfg.version == SCANNER_CONFIG_VERSION == "v1", "scanner config version drifted from v1"

    tracker = SessionTracker(config=session_cfg, scanner_config=scanner_cfg,
                              weekly_pivots=weekly_pivots, tick_size=0.25)

    start = pd.Timestamp(args.start, tz=ET).normalize()
    end = (pd.Timestamp(args.end, tz=ET) + pd.Timedelta(days=1)).normalize()
    et_ts = df["timestamp"].dt.tz_convert(ET)

    bars_list = list(df.itertuples())

    seen_episodes: set[str] = set()
    episodes: list[dict] = []

    print("Replaying and collecting new-episode events ...")
    for i, row in enumerate(bars_list):
        tracker.update(row)
        ts_et = row.timestamp.astimezone(ET)
        if not (start <= row.timestamp < end):
            continue
        if row.timestamp.minute % 15 != 0:
            continue

        opp = scan(
            tracker=tracker, features=features, scanner_features=scanner_features,
            price=row.close, timestamp=row.timestamp, symbol=args.symbol.upper(),
            seen_episodes=seen_episodes, cfg=scanner_cfg,
        )
        if not opp.is_new_episode:
            continue

        feat_row = features.loc[row.timestamp] if row.timestamp in features.index else None
        sf_row = scanner_features.loc[row.timestamp] if row.timestamp in scanner_features.index else None

        point_in_time = {
            "timestamp": row.timestamp.isoformat(),
            "timestamp_et": ts_et.strftime("%Y-%m-%d %a %H:%M ET"),
            "symbol": opp.symbol,
            "sessions": sorted(opp.sessions),
            "price": round(row.close, 2),
            "episode_id": opp.episode_id,
            "categories": list(opp.categories),
            "reasoning": list(opp.reasoning),
            "levels": [asdict(l) for l in opp.level_states],
            "confluence_zones": [asdict(z) for z in opp.confluence_zones],
            "vwap": round(tracker.vwap, 2) if tracker.vwap is not None else None,
            "vwap_distance_pts": round(row.close - tracker.vwap, 2) if tracker.vwap is not None else None,
            "indicators": {
                "macd_15m_hist": float(feat_row.get("15m_hist")) if feat_row is not None and pd.notna(feat_row.get("15m_hist")) else None,
                "rsi_60m": float(feat_row.get("60m_rsi")) if feat_row is not None and pd.notna(feat_row.get("60m_rsi")) else None,
                "macd_4h_hist": float(feat_row.get("4h_hist")) if feat_row is not None and pd.notna(feat_row.get("4h_hist")) else None,
                "rsi_4h": float(feat_row.get("4h_rsi")) if feat_row is not None and pd.notna(feat_row.get("4h_rsi")) else None,
                "macd_15m_bottom_recent": bool(feat_row.get("15m_macd_bottom_recent")) if feat_row is not None else None,
                "macd_15m_top_recent": bool(feat_row.get("15m_macd_top_recent")) if feat_row is not None else None,
            },
            "volatility": {
                "atr_60bar": float(sf_row.get("atr")) if sf_row is not None and pd.notna(sf_row.get("atr")) else None,
                "atr_percentile": float(sf_row.get("atr_percentile")) if sf_row is not None and pd.notna(sf_row.get("atr_percentile")) else None,
                "daily_atr": float(sf_row.get("daily_atr")) if sf_row is not None and pd.notna(sf_row.get("daily_atr")) else None,
            },
        }

        forward_research_only = {
            "_WARNING": "RESEARCH ONLY — future/outcome data. Must never enter MarketSnapshot or QuantHarness input.",
            "30m": _forward_excursion(bars_list, i, row.close, 30),
            "60m": _forward_excursion(bars_list, i, row.close, 60),
            "120m": _forward_excursion(bars_list, i, row.close, 120),
        }

        episodes.append({
            "point_in_time": point_in_time,
            "forward_excursion_research_only": forward_research_only,
        })

    print(f"Collected {len(episodes)} new-episode events.")

    out = {
        "meta": {
            "symbol": args.symbol.upper(),
            "date_range": [args.start, args.end],
            "scanner_config_version": SCANNER_CONFIG_VERSION,
            "session_config": {k: str(v) for k, v in session_cfg.__dict__.items()},
            "scanner_config": scanner_cfg.__dict__,
            "episode_count": len(episodes),
        },
        "episodes": episodes,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
