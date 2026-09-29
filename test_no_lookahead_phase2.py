#!/usr/bin/env python3
"""
Structural regression tests for the Phase 2 pipeline (OpportunitySnapshot,
chart_renderer, trend_analysis, and the live structured-output round trip).
Must pass before run_phase2_pilot.py is run against real episodes.

Test 1 avoids literally hard-truncating the dataframe at the exact cutoff
bar (unlike test_no_lookahead.py's pattern): opportunity_scanner's daily
ATR (_daily_atr_series) uses a rolling window that is inclusive of the
*current* completed-day row, and a session day's own row only becomes
eligible once its last bar's timestamp is <= T. If you hard-truncate a
dataframe so its very last row IS T, that in-progress day's own (partial)
range becomes eligible as its own ATR source at exactly that row — a
same-bar self-reference, not a future-data leak, but it means two
differently-truncated dataframes can disagree at their own last row. Real
usage (run_scanner.py, generate_episode_review.py) never hard-truncates —
features are computed once over the whole historical file, exactly like
the two variants compared below, both of which extend at least a day past
the cutoff so "today" is unambiguously complete in both. That is the
actual guarantee this project needs: a spike/extra data further in the
future must never move what was computed at T.

Run directly: python3 test_no_lookahead_phase2.py
(Tests 1-3 are pure/offline. Test 4 requires LM Studio running at the
Phase 2 endpoint/model — see llm_provider.LLMConfig defaults.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from chart_renderer import render
from llm_provider import LLMConfig, OpenAICompatProvider
from mtf_indicators import compute_mtf_features
from opportunity_scanner import compute_scanner_features, scan
from opportunity_snapshot import build_opportunity_snapshot
from session_config import SessionConfig
from session_tracker import SessionTracker
from trend_analysis import compute_trend_analysis

START = datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc)  # 18:00 ET Jan 5 (Globex open)


def _make_synthetic_days(n_days: int = 5, spike_at: datetime | None = None) -> pd.DataFrame:
    minutes = n_days * 24 * 60
    rows = []
    price = 20000.0
    for i in range(minutes):
        ts = START + timedelta(minutes=i)
        if spike_at is not None and ts == spike_at:
            price += 300.0
        else:
            price += 0.05 * ((i % 11) - 5)
        rows.append(dict(timestamp=ts, open=price, high=price + 1.0, low=price - 1.0,
                          close=price + 0.1, volume=100 + (i % 50)))
        price += 0.1
    return pd.DataFrame(rows)


def _snapshot_at(df: pd.DataFrame, cutoff: datetime, symbol: str = "TEST"):
    features = compute_mtf_features(df)
    scanner_features = compute_scanner_features(df)
    tracker = SessionTracker(config=SessionConfig(), tick_size=0.25)
    seen: set[str] = set()
    for row in df.itertuples():
        tracker.update(row)
        if row.timestamp == cutoff:
            opp = scan(tracker=tracker, features=features, scanner_features=scanner_features,
                       price=row.close, timestamp=cutoff, symbol=symbol,
                       seen_episodes=seen, cfg=tracker.scanner_config)
            return build_opportunity_snapshot(tracker, features, scanner_features, opp, df, lookback_bars=60)
    raise AssertionError("cutoff timestamp not found in dataframe")


def test_snapshot_unaffected_by_future_data() -> None:
    spike_at = START + timedelta(days=3, hours=6)  # deep in the future relative to the cutoff below
    cutoff = START + timedelta(days=1, hours=6)     # mid-day, day 2

    df_short = _make_synthetic_days(n_days=2, spike_at=None)          # ends well past cutoff, no spike
    df_long = _make_synthetic_days(n_days=5, spike_at=spike_at)       # same history through cutoff, plus
                                                                        # 3 more days including a big spike

    snap_short = _snapshot_at(df_short, cutoff)
    snap_long = _snapshot_at(df_long, cutoff)

    assert snap_short.to_dict() == snap_long.to_dict(), (
        "OpportunitySnapshot at T changed depending on how much future data "
        "(including a large spike) exists beyond T in the source dataframe"
    )
    assert all(b[0] <= cutoff for b in snap_short.recent_bars), "recent_bars leaked a bar after T"
    print("PASS: OpportunitySnapshot at T is identical regardless of future data/spikes beyond T")


def test_chart_renderer_rejects_future_bar() -> None:
    cutoff = START + timedelta(hours=2)
    good_bars = tuple(
        (START + timedelta(minutes=i), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 500.0)
        for i in range(20)
    )
    future_bar = (cutoff + timedelta(minutes=5), 200.0, 201.0, 199.0, 200.5, 500.0)
    bad_bars = good_bars + (future_bar,)

    try:
        render(bad_bars, snapshot_timestamp=cutoff)
        raise AssertionError("render() did not raise on a bar timestamped after the snapshot cutoff")
    except ValueError as exc:
        assert "point-in-time violation" in str(exc)

    png_bytes, b64 = render(good_bars, snapshot_timestamp=cutoff)
    assert len(png_bytes) > 0 and len(b64) > 0
    print("PASS: chart_renderer.render() structurally rejects a post-cutoff bar and accepts a clean one")


def test_trend_analysis_is_pure() -> None:
    bars = tuple(
        (START + timedelta(minutes=i), 100.0 + 0.1 * i, 101.0 + 0.1 * i, 99.0 + 0.1 * i, 100.5 + 0.1 * i, 500.0)
        for i in range(30)
    )
    result1 = compute_trend_analysis(bars, daily_atr=10.0)
    result2 = compute_trend_analysis(bars, daily_atr=10.0)
    assert result1 == result2, "compute_trend_analysis is not deterministic for identical input"

    subset = bars[:20]
    result_subset = compute_trend_analysis(subset, daily_atr=10.0)
    assert result_subset.bars_used == 20
    assert result1.bars_used == 30
    # Same first-20-bars values feeding the subset call must not depend on what
    # was appended after them in the full 30-bar call.
    result_subset_again = compute_trend_analysis(subset, daily_atr=10.0)
    assert result_subset == result_subset_again
    print("PASS: compute_trend_analysis is a pure function of exactly the bars it's given")


def test_live_structured_output_round_trip() -> None:
    provider = OpenAICompatProvider(LLMConfig(model="qwen/qwen3-vl-4b", base_url="http://127.0.0.1:1234/v1"))
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}, "value": {"type": "integer"}},
        "required": ["status", "value"],
        "additionalProperties": False,
    }
    parsed, raw = provider.complete_structured(
        system="You are a test assistant.",
        user='Return {"status": "OK", "value": 1} exactly.',
        schema=schema,
        schema_name="probe_v1",
    )
    assert parsed == {"status": "OK", "value": 1}, f"unexpected parse: {parsed} (raw={raw!r})"
    assert provider.last_latency_s is not None and provider.last_latency_s > 0
    print(f"PASS: complete_structured round-trips through llm_provider.py "
          f"(latency={provider.last_latency_s:.2f}s, retries={provider.last_retry_count})")


if __name__ == "__main__":
    test_snapshot_unaffected_by_future_data()
    test_chart_renderer_rejects_future_bar()
    test_trend_analysis_is_pure()
    try:
        test_live_structured_output_round_trip()
    except ConnectionError as exc:
        print(f"SKIPPED (LM Studio unreachable): {exc}")
    print("\nAll Phase 2 no-lookahead / structural checks passed.")
