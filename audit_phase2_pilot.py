#!/usr/bin/env python3
"""
INPUT/OUTPUT audit of the 12 Phase 2 pilot episodes — diagnosis only, no
forward/outcome data read anywhere in this file. Joins the persisted
`analyses` and `diagnostic_forced_direction` rows with a manually-authored
evidence/conflict/category read of each episode (the mechanical fields are
pulled from the DB; bullish/bearish synthesis, conflicts, and diagnostic
category tags are analytic judgments written by reviewing each episode's
full output, not computed).

Writes data/phase2_pilot_audit.json. Does not change the scanner, does not
tune any prompt, does not re-run anything against forward outcomes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = "data/experiments.db"
OUT_PATH = "data/phase2_pilot_audit.json"

# Manually-authored per-episode judgment. Keyed by episode_id. Written after
# reading each episode's full snapshot/trend/vision/decision/diagnostic
# output — no forward/outcome data was consulted in forming any of this.
_AUDIT = {
    "32990adceec66962": {
        "bullish_evidence": [
            "Rising trend channel, price near support side (position 0.12)",
            "60m RSI 67.8 and 4h RSI 72.6 both firmly bullish",
            "LondonHigh/LondonLow both approaching within ~2pts (confluence)",
        ],
        "bearish_evidence": [
            "60m/4h RSI both quite elevated (67.8/72.6) — overbought mean-reversion risk",
            "15m MACD histogram slightly negative (-0.49), no confirming momentum",
        ],
        "conflicts": [
            "Rising-channel structure (bullish) vs elevated RSI readings (overbought, mean-reversion risk)",
            "WeeklyR1/R2 cited as 'lost' 227-451pts away — too far to be relevant, adds noise not signal",
        ],
        "categories": ["INSUFFICIENT_CONFIRMATION", "VISION_DETERMINISTIC_DISAGREEMENT", "MISSING_INFORMATION"],
        "category_notes": "Vision claimed 'distance to support is relatively large' when the exact "
                           "deterministic value was 1.75pts (among the smallest in the pilot) — a direct "
                           "disagreement with Python's own number, not just a different framing.",
    },
    "5018233a09cca258": {
        "bullish_evidence": [
            "15m MACD histogram expanding 2+ bars (the scanner's own trigger)",
            "Price near mid-channel of a tight, compressed range",
        ],
        "bearish_evidence": [
            "Trend channel flat, compression=True — no directional structure",
            "ATR at 0th percentile — the quietest reading in the entire pilot",
        ],
        "conflicts": [
            "Scanner fired momentum_expansion, but vision independently reported "
            "'no clear momentum expansion' — direct disagreement over the same signal.",
        ],
        "categories": ["RANGE_OR_CHOP", "VISION_DETERMINISTIC_DISAGREEMENT", "MODEL_OUTPUT_QUALITY"],
        "category_notes": "Diagnostic bull_case cites 'MACD histogram expanding' as bullish while bear_case "
                           "cites 'no clear momentum expansion' for the same underlying fact — internally "
                           "inconsistent, not just a close call.",
    },
    "90fe7b18ee276a07": {
        "bullish_evidence": [
            "15m MACD confirmed bottom reversal (bottom_recent=True)",
            "Price sitting almost exactly at trend-channel support (position 0.05)",
            "Rising channel structure",
        ],
        "bearish_evidence": [
            "60m RSI neutral (45.2) — no confirming higher-timeframe momentum",
            "Low ATR percentile (0.14) — expansion not yet confirmed by volatility",
        ],
        "conflicts": [
            "None substantial — this is one of the cleaner single-direction reads in the pilot.",
        ],
        "categories": ["INSUFFICIENT_CONFIRMATION", "MODEL_OUTPUT_QUALITY"],
        "category_notes": "This is arguably the most textbook-bullish location in the whole pilot "
                           "(confirmed reversal exactly at support) yet still NO_TRADE at only 0.60 "
                           "confidence. Diagnostic bear_case lists 'distance to support is small, implying "
                           "low risk of reversal' as a BEARISH factor — that's backwards logic (low reversal "
                           "risk at support is bullish, not bearish) — a real reasoning error, not disagreement.",
    },
    "121a07c22a108096": {
        "bullish_evidence": [
            "15m MACD confirmed bottom reversal + expanding histogram",
            "60m RSI mild-bullish (48.6), VWAP below price",
        ],
        "bearish_evidence": [
            "4h MACD histogram negative (-5.29) — opposes the 15m read",
        ],
        "conflicts": [
            "15m bullish reversal/expansion directly conflicts with a negative 4h MACD histogram — a real "
            "cross-timeframe disagreement, correctly identified by both the original and diagnostic calls.",
            "Vision claimed price is 'near upper boundary' of the channel, but the deterministic "
            "price_position_in_channel was 0.36 (closer to a third of the way up) — a real disagreement.",
        ],
        "categories": ["CONFLICTING_DIRECTIONAL_EVIDENCE", "VISION_DETERMINISTIC_DISAGREEMENT", "MODEL_OUTPUT_QUALITY"],
        "category_notes": "supporting_factors and opposing_factors were BOTH left empty in the original "
                           "decision despite a substantive analysis_summary that explicitly discusses the "
                           "4h/15m conflict — the structured fields didn't capture what the prose already knew.",
    },
    "163adb558a9f719b": {
        "bullish_evidence": [
            "15m MACD confirmed reversal + expanding histogram (strongest momentum signal so far)",
            "AsiaLow approaching with supporting context",
        ],
        "bearish_evidence": [
            "Price already 86% up the trend channel (position 0.86) — poor location for a fresh long",
            "VWAP 28.6pts below price (+0.11x ATR) — already extended above VWAP",
            "4h MACD histogram still negative",
        ],
        "conflicts": [
            "Strong 15m bullish momentum vs. price already extended near the top of its channel and above "
            "VWAP — a genuine location/risk-reward tension, not just a coin-flip on direction.",
        ],
        "categories": ["POOR_LOCATION", "CONFLICTING_DIRECTIONAL_EVIDENCE", "MODEL_OUTPUT_QUALITY"],
        "category_notes": "Vision/diagnostic both describe 'price 0.86 into channel' inconsistently — once as "
                           "a 'risk' (in vision_analysis) and once as a bullish 'continuation' decisive factor "
                           "(in the diagnostic) for the same number.",
    },
    "ba8fb7b876dd63cf": {
        "bullish_evidence": [
            "15m MACD confirmed reversal + strongest expansion magnitude in this session cluster (10.70)",
            "LondonHigh broken_holding above — a resistance that broke and is holding, structurally a "
            "bullish continuation signal",
            "VWAP 45pts below price",
        ],
        "bearish_evidence": [
            "4h MACD histogram still negative",
        ],
        "conflicts": [
            "4h MACD negative vs. strong 15m reversal/expansion and a bullish broken_holding structure.",
        ],
        "categories": ["MODEL_OUTPUT_QUALITY", "VISION_DETERMINISTIC_DISAGREEMENT", "CONFLICTING_DIRECTIONAL_EVIDENCE"],
        "category_notes": "The diagnostic call's bear_case describes 'LondonHigh is broken_holding, "
                           "suggesting recent bearish momentum' — this is backwards: a High level that broke "
                           "and is holding above is a structurally bullish continuation signal, not bearish. "
                           "Also, vision called the (pilot's widest, 61.23pt) channel 'narrow' and volatility "
                           "'low' when it was actually the widest channel and mid-range ATR percentile (0.51) "
                           "in the entire pilot — a clear disagreement with Python's own numbers. This is the "
                           "richest confluence episode (4 categories fired) and still NO_TRADE.",
    },
    "2260e4bc0b44b43d": {
        "bullish_evidence": [
            "15m MACD histogram expanding (strongest reading in the pilot, 13.86)",
            "ATR at 64th percentile — the volatility_expansion trigger",
            "Rising channel, positive slope",
        ],
        "bearish_evidence": [
            "Price already 72% up its channel and 0.25x ATR above VWAP — moderately extended",
            "4h MACD histogram still negative",
        ],
        "conflicts": [
            "Persistent 4h-negative-vs-15m-bullish split seen across most of this session's episodes.",
        ],
        "categories": ["EXCESSIVE_EXTENSION", "CONFLICTING_DIRECTIONAL_EVIDENCE", "MODEL_OUTPUT_QUALITY"],
        "category_notes": "vision_analysis.pattern came back as the literal string 'null' instead of JSON "
                           "null. The final summary also downplays the episode's own strongest 15m momentum "
                           "reading in the whole pilot ('no clear momentum expansion beyond 15m MACD "
                           "histogram') — an oddly dismissive read of its own best signal.",
    },
    "685c4a881498dea0": {
        "bullish_evidence": [
            "4h MACD histogram positive (+5.59) and 4h RSI 74.3 — first bullish 4h reading in the pilot",
            "60m RSI 70.3, elevated",
        ],
        "bearish_evidence": [
            "15m MACD histogram expanding NEGATIVELY (-8.36) — the trigger, but bearish direction",
            "AsiaHigh broken_holding below price (bearish structural read)",
        ],
        "conflicts": [
            "15m momentum expanding bearishly while 4h context is bullish/overbought — the mirror image of "
            "the 15m-bullish/4h-bearish conflict seen earlier in the day.",
        ],
        "categories": ["MODEL_OUTPUT_QUALITY", "CONFLICTING_DIRECTIONAL_EVIDENCE", "RANGE_OR_CHOP"],
        "category_notes": "The most internally incoherent diagnostic result in the pilot: bull_case lists "
                           "'4h MACD histogram positive' and 'price near resistance' (both bullish-reading "
                           "items) yet forced_direction=SHORT was chosen anyway, with bear_case items that "
                           "don't clearly argue the opposite either. This looks like genuine model confusion "
                           "on this specific episode, not a considered judgment either way.",
    },
    "87023210fce06e9f": {
        "bullish_evidence": [
            "60m RSI 18.95 — deeply oversold, a classic (if risky) bounce setup",
        ],
        "bearish_evidence": [
            "15m MACD expanding negatively, 4h MACD histogram strongly negative (-40.62)",
            "4h RSI 36.4 — bearish",
            "ATR at 95th percentile — high volatility",
            "VWAP -140pts / -0.57x daily ATR (the vwap_extension trigger)",
        ],
        "conflicts": [
            "Deeply oversold 60m RSI (bounce risk for a fresh short) vs. strongly negative 4h momentum "
            "(continuation case) — the classic oversold-in-a-downtrend tension.",
        ],
        "categories": ["EXCESSIVE_EXTENSION", "POOR_LOCATION", "MISSING_INFORMATION"],
        "category_notes": "Price sits near the BOTTOM of the falling channel (position 0.20) — a "
                           "less favorable spot to newly initiate a short (more favorable to cover). "
                           "AsiaLow cited as broken_holding 289pts away is stale/irrelevant noise, same "
                           "pattern as episode 1's far-away weekly levels. The final summary explicitly names "
                           "'the high ATR and 95% percentile volatility add uncertainty' as its own reason — "
                           "one of the more specific, defensible NO_TRADE rationales in the pilot.",
    },
    "8faa25d3e99f435a": {
        "bullish_evidence": [
            "60m RSI 57.4, mild bullish tilt",
        ],
        "bearish_evidence": [
            "15m MACD top_recent=True — the ONLY topping-reversal signal in the pilot (all others were "
            "bottom_recent)",
            "PDH approaching (-5.00pts) AND AsiaHigh approaching (-7.50pts) — two overhead resistance "
            "levels stacked within ~8pts of each other and of price",
            "4h MACD histogram negative",
        ],
        "conflicts": [
            "Mild bullish 60m RSI vs. a topping reversal signal arriving right at a resistance cluster — "
            "the topping signal and the level cluster reinforce each other more than they conflict.",
        ],
        "categories": ["NEARBY_OPPOSING_LEVEL", "INSUFFICIENT_CONFIRMATION"],
        "category_notes": "The two overhead levels (PDH, AsiaHigh) sit close enough together to be a real "
                           "confluence, but the scanner's confluence_zones didn't flag them as one — worth "
                           "checking why. Otherwise this is one of the more internally consistent, coherent "
                           "reads in the pilot: reversal-at-resistance is a real bearish structure, just not "
                           "strongly confirmed by volatility (4th percentile, very quiet).",
    },
    "2dcbd02e79f0a3d6": {
        "bullish_evidence": [
            "Price 8.5pts from resistance in a tight, consolidating range (weak — mainly 'not yet there')",
        ],
        "bearish_evidence": [
            "15m MACD expanding negatively, 4h MACD histogram -70.6 (most extreme reading in the pilot)",
            "60m RSI 26.2 and 4h RSI 31.2 — both bearish across two timeframes",
            "VWAP -223pts / -0.53x daily ATR (vwap_extension trigger)",
            "Price near the TOP of the falling channel (position 0.84) — a favorable location to "
            "short a retracement in a falling trend",
            "Falling channel with negative support/resistance slopes",
        ],
        "conflicts": [
            "None substantial — this is the cleanest, most multi-timeframe-aligned bearish read in the "
            "entire pilot, combined with a favorable (retracement-into-resistance) location.",
        ],
        "categories": ["OTHER"],
        "category_notes": "This is the single strongest case in the pilot for 'the model sees a real "
                           "directional lean but the framing/threshold causes it to abstain': three aligned "
                           "timeframes, an extreme 4h reading, and a textbook short-the-bounce location, yet "
                           "still NO_TRADE at 0.75 confidence with a fairly generic stated reason ('lack of "
                           "clear breakout/rejection structure'). The diagnostic call was also the single most "
                           "confident and coherent forced-direction response in the whole pilot (SHORT 0.75, "
                           "explicitly citing 'downtrend confirmed by both support and resistance slopes').",
    },
    "b26bc793fe5bd62b": {
        "bullish_evidence": [
            "Price 19.25pts from resistance in a moderate-volatility range (weak, 'not yet there')",
        ],
        "bearish_evidence": [
            "15m MACD expanding negatively, 4h MACD histogram -70.6 (same extreme 4h bar as the prior "
            "episode, 1h45m earlier)",
            "60m RSI 20.1 and 4h RSI 31.2 — both bearish",
            "VWAP -361.6pts / -0.86x daily ATR — the LARGEST VWAP extension in the entire pilot",
            "Price near the top of the falling channel (position 0.81) — similar favorable "
            "short-the-bounce location as the prior episode",
            "WeeklyS2 approaching with supporting context",
        ],
        "conflicts": [
            "None substantial on direction — the one real, specific counter-argument is a large "
            "(84pt) gap down to the nearest support, i.e. real room for an adverse bounce if wrong.",
        ],
        "categories": ["EXCESSIVE_EXTENSION"],
        "category_notes": "Similar clean multi-timeframe bearish alignment to the prior episode, but this "
                           "one has a specific, non-generic stated NO_TRADE reason (large gap to support = "
                           "meaningful rebound risk before the thesis would be proven wrong) — more "
                           "defensible than the prior episode's more generic reasoning, even though the "
                           "underlying evidence looks similarly strong.",
    },
}


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    analyses = {r["episode_id"]: r for r in conn.execute("SELECT * FROM analyses ORDER BY timestamp")}
    diagnostics = {r["episode_id"]: r for r in conn.execute(
        "SELECT * FROM diagnostic_forced_direction ORDER BY timestamp")}
    conn.close()

    episodes = []
    for episode_id, a in analyses.items():
        snap = json.loads(a["snapshot_json"])
        trend = json.loads(a["trend_analysis_json"]) if a["trend_analysis_json"] else None
        vision = json.loads(a["vision_analysis_json"])
        diag = diagnostics.get(episode_id)
        manual = _AUDIT.get(episode_id, {})

        supporting = json.loads(a["supporting_factors_json"])
        opposing = json.loads(a["opposing_factors_json"])

        episodes.append({
            "episode_id": episode_id,
            "timestamp": a["timestamp"],
            "sessions": snap["sessions"],
            "scanner_categories": snap["categories"],
            "important_levels": [
                {"name": l["name"], "state": l["state"], "distance_pts": l["distance_pts"]}
                for l in snap["levels"] if l["state"] != "far"
            ],
            "momentum": {
                "15m_hist": snap["macd_15m_hist"], "bottom_recent": snap["macd_15m_bottom_recent"],
                "top_recent": snap["macd_15m_top_recent"], "60m_rsi": snap["rsi_60m"],
                "4h_hist": snap["macd_4h_hist"], "4h_rsi": snap["rsi_4h"],
            },
            "vwap_context": {
                "vwap": snap["vwap"], "distance_pts": snap["vwap_distance_pts"],
                "distance_daily_atr": snap["vwap_distance_daily_atr"], "extension_flag": snap["vwap_extension_flag"],
            },
            "volatility_context": {
                "atr_60bar": snap["atr_60bar"], "atr_percentile": snap["atr_percentile"], "daily_atr": snap["daily_atr"],
            },
            "trend_analysis_conclusion": (
                f"{trend['channel_direction']}, compression={trend['compression']}, "
                f"width={trend['channel_width_pts']:.2f}pts ({trend['channel_width_atr_mult']:.2f}x daily ATR), "
                f"price_position={trend['price_position_in_channel']:.2f}"
            ) if trend else "insufficient bars for a fit",
            "vision_analysis_conclusion": (
                f"{vision['trend_structure']}, pattern={vision['pattern']!r}, compression={vision['compression']}"
            ),
            "bullish_evidence": manual.get("bullish_evidence", []),
            "bearish_evidence": manual.get("bearish_evidence", []),
            "conflicts_identified": manual.get("conflicts", []),
            "final_decision": a["decision"],
            "confidence": a["confidence"],
            "reason_for_no_trade": a["reason"] or a["analysis_summary"],
            "supporting_factors_populated": bool(supporting),
            "opposing_factors_populated": bool(opposing),
            "diagnostic_categories": manual.get("categories", []),
            "category_notes": manual.get("category_notes", ""),
            "diagnostic_forced_direction": diag["forced_direction"] if diag else None,
            "diagnostic_confidence": diag["directional_confidence"] if diag else None,
            "diagnostic_bull_case": json.loads(diag["bull_case_json"]) if diag else None,
            "diagnostic_bear_case": json.loads(diag["bear_case_json"]) if diag else None,
            "diagnostic_decisive_factors": json.loads(diag["decisive_factors_json"]) if diag else None,
        })

    out = {
        "meta": {
            "note": "Diagnosis only. No forward/outcome data used anywhere in this audit.",
            "episode_count": len(episodes),
        },
        "episodes": episodes,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
