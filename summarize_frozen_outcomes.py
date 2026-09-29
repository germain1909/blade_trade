#!/usr/bin/env python3
"""
Descriptive summary of data/phase2_frozen_outcomes.json — aggregate stats,
LONG vs SHORT breakdown, confidence breakdown, and the pre-declared
strong/aligned vs conflicting/range subgroup comparison. Purely descriptive
over a 12-episode sample; makes no statistical claims and computes no
win-rate/profit/Sharpe/expectancy — there is no executable trade defined.
"""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

DATA_PATH = "data/phase2_frozen_outcomes.json"

# Pre-declared BEFORE outcome reveal, from the blind audit (data/phase2_pilot_audit.json).
# Preserved verbatim — not redefined after seeing outcomes.
STRONG_ALIGNED_IDS = {
    "90fe7b18ee276a07": "Episode 3",
    "87023210fce06e9f": "Episode 9",
    "2dcbd02e79f0a3d6": "Episode 11",
    "b26bc793fe5bd62b": "Episode 12",
}
CONFLICTING_RANGE_IDS = {
    "5018233a09cca258": "Episode 2",
    "121a07c22a108096": "Episode 4",
    "163adb558a9f719b": "Episode 5",
    "ba8fb7b876dd63cf": "Episode 6",
    "2260e4bc0b44b43d": "Episode 7",
    "685c4a881498dea0": "Episode 8",
}


def _fmt(v):
    return f"{v:+.2f}" if isinstance(v, (int, float)) else "None"


def _agg(values: list[float]) -> dict:
    return {"median": round(stats.median(values), 2), "mean": round(sum(values) / len(values), 2)}


def print_episode_table(episodes: list[dict]) -> None:
    print("=" * 145)
    print("12-EPISODE FROZEN OUTCOME TABLE")
    print("=" * 145)
    header = (f"{'#':<3}{'timestamp':<22}{'sess':<8}{'orig':<9}{'orig_c':<7}{'fdir':<6}{'fconf':<6}"
              f"{'30m MFE/MAE/net':<20}{'60m MFE/MAE/net':<20}{'120m MFE/MAE/net':<20}")
    print(header)
    print("-" * 145)
    for i, e in enumerate(episodes, 1):
        o30, o60, o120 = e["outcomes"]["30m"], e["outcomes"]["60m"], e["outcomes"]["120m"]
        col30 = f"{o30['mfe']:.1f}/{o30['mae']:.1f}/{_fmt(o30['net_change'])}"
        col60 = f"{o60['mfe']:.1f}/{o60['mae']:.1f}/{_fmt(o60['net_change'])}"
        col120 = f"{o120['mfe']:.1f}/{o120['mae']:.1f}/{_fmt(o120['net_change'])}"
        print(
            f"{i:<3}{e['timestamp'][:16]:<22}{','.join(e['sessions'])[:7]:<8}"
            f"{e['original_decision']:<9}{e['original_confidence']:<7.2f}"
            f"{e['frozen_forced_direction']:<6}{e['directional_confidence']:<6.2f}"
            f"{col30:<20}{col60:<20}{col120:<20}"
        )
    print()
    print("MFE/MAE ratios (None = undefined per MAE=0 or no movement, not an invented large number):")
    for i, e in enumerate(episodes, 1):
        ratios = {h: e["outcomes"][h]["mfe_mae_ratio"] for h in ("30m", "60m", "120m")}
        notes = {h: e["outcomes"][h]["mfe_mae_ratio_note"] for h in ("30m", "60m", "120m") if e["outcomes"][h]["mfe_mae_ratio_note"]}
        print(f"  #{i:<2} {e['episode_id'][:8]}  ratios={ratios}  notes={notes if notes else '-'}")
    print()
    print("Time-to-move within 120m (minutes elapsed; None = threshold never reached):")
    for i, e in enumerate(episodes, 1):
        t = e["time_to_move_120m"]
        print(f"  #{i:<2} {e['episode_id'][:8]}  dir={e['frozen_forced_direction']:<5}  "
              f"+25={t['time_to_+25']}  +50={t['time_to_+50']}  -25={t['time_to_-25']}  -50={t['time_to_-50']}")
    print()


def _subgroup_stats(episodes: list[dict], horizon: str) -> dict:
    mfes = [e["outcomes"][horizon]["mfe"] for e in episodes]
    maes = [e["outcomes"][horizon]["mae"] for e in episodes]
    nets = [e["outcomes"][horizon]["net_change"] for e in episodes]
    return {
        "count": len(episodes),
        "median_mfe": round(stats.median(mfes), 2), "median_mae": round(stats.median(maes), 2),
        "median_net": round(stats.median(nets), 2),
        "positive_net_count": sum(1 for n in nets if n > FLAT_THRESHOLD),
    }


FLAT_THRESHOLD = 2.0


def print_aggregate(episodes: list[dict]) -> None:
    print("=" * 100)
    print("AGGREGATE RESULTS — ALL 12 (descriptive only)")
    print("=" * 100)
    for h in ("30m", "60m", "120m"):
        mfes = [e["outcomes"][h]["mfe"] for e in episodes]
        maes = [e["outcomes"][h]["mae"] for e in episodes]
        nets = [e["outcomes"][h]["net_change"] for e in episodes]
        mfe_agg, mae_agg, net_agg = _agg(mfes), _agg(maes), _agg(nets)
        pos = sum(1 for n in nets if n > FLAT_THRESHOLD)
        neg = sum(1 for n in nets if n < -FLAT_THRESHOLD)
        flat = len(nets) - pos - neg
        mfe_gt = sum(1 for e in episodes if e["outcomes"][h]["mfe"] > e["outcomes"][h]["mae"])
        mae_gt = sum(1 for e in episodes if e["outcomes"][h]["mae"] > e["outcomes"][h]["mfe"])
        eq = len(episodes) - mfe_gt - mae_gt
        print(f"\n{h}:")
        print(f"  MFE  median={mfe_agg['median']:<7} mean={mfe_agg['mean']}")
        print(f"  MAE  median={mae_agg['median']:<7} mean={mae_agg['mean']}")
        print(f"  net  median={net_agg['median']:<7} mean={net_agg['mean']}")
        print(f"  positive directional net: {pos}/12   negative: {neg}/12   "
              f"approx flat (|net|<={FLAT_THRESHOLD}pts): {flat}/12")
        print(f"  MFE>MAE: {mfe_gt}/12   MAE>MFE: {mae_gt}/12   equal: {eq}/12")


def print_long_vs_short(episodes: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("LONG vs SHORT BREAKDOWN (descriptive only)")
    print("=" * 100)
    longs = [e for e in episodes if e["frozen_forced_direction"] == "LONG"]
    shorts = [e for e in episodes if e["frozen_forced_direction"] == "SHORT"]
    for label, group in (("LONG", longs), ("SHORT", shorts)):
        print(f"\n{label} (n={len(group)}):")
        for h in ("30m", "60m", "120m"):
            s = _subgroup_stats(group, h)
            print(f"  {h}: median MFE={s['median_mfe']} median MAE={s['median_mae']} "
                  f"median net={s['median_net']} positive-net count={s['positive_net_count']}/{s['count']}")


def print_confidence_breakdown(episodes: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("CONFIDENCE BREAKDOWN (descriptive only — NOT calibrated probability)")
    print("=" * 100)
    by_conf: dict[float, list[dict]] = {}
    for e in episodes:
        by_conf.setdefault(e["directional_confidence"], []).append(e)
    for conf in sorted(by_conf):
        group = by_conf[conf]
        print(f"\nconfidence={conf} (n={len(group)}):")
        for h in ("30m", "60m", "120m"):
            s = _subgroup_stats(group, h)
            print(f"  {h}: median MFE={s['median_mfe']} median MAE={s['median_mae']} "
                  f"median net={s['median_net']} positive-net count={s['positive_net_count']}/{s['count']}")


def print_subgroup_comparison(episodes: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("PRE-DECLARED STRONG/ALIGNED vs CONFLICTING/RANGE (descriptive only, tiny n)")
    print("=" * 100)
    by_id = {e["episode_id"]: e for e in episodes}
    strong = [by_id[i] for i in STRONG_ALIGNED_IDS if i in by_id]
    conflicting = [by_id[i] for i in CONFLICTING_RANGE_IDS if i in by_id]

    for label, group, names in (("STRONG/ALIGNED", strong, STRONG_ALIGNED_IDS),
                                 ("CONFLICTING/RANGE", conflicting, CONFLICTING_RANGE_IDS)):
        print(f"\n{label} (n={len(group)}): {[names[e['episode_id']] for e in group]}")
        print(f"  directions: {[e['frozen_forced_direction'] for e in group]}")
        for h in ("30m", "60m", "120m"):
            s = _subgroup_stats(group, h)
            print(f"  {h}: median MFE={s['median_mfe']} median MAE={s['median_mae']} "
                  f"median net={s['median_net']} positive-net count={s['positive_net_count']}/{s['count']}")


def main() -> None:
    data = json.loads(Path(DATA_PATH).read_text())
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} frozen episodes from {DATA_PATH}")
    print(f"NOTE: {data['meta']['note']}\n")

    print_episode_table(episodes)
    print_aggregate(episodes)
    print_long_vs_short(episodes)
    print_confidence_breakdown(episodes)
    print_subgroup_comparison(episodes)


if __name__ == "__main__":
    main()
