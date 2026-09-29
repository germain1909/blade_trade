#!/usr/bin/env python3
"""
Structural regression test: proves MarketSnapshot cannot leak data from
after its timestamp, rather than trusting it by review.

Builds a synthetic 1-minute bar series with a deliberate price "spike"
placed well after a chosen cutoff T, then:

  1. Confirms every bar in `snapshot.recent_bars` has timestamp <= T, even
     when the dataframe passed in extends hours past T.
  2. Confirms a snapshot built at T from the FULL dataframe (spike included,
     further in the future) is field-for-field identical to one built at T
     from a copy of the dataframe hard-truncated to <= T. If a future
     refactor accidentally lets a later bar or indicator value leak into
     snapshot construction, this assertion is what catches it.

Run directly: python3 test_no_lookahead.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from market_snapshot import build_snapshot
from mtf_indicators import compute_mtf_features


def _make_synthetic_bars(n_minutes: int = 600, spike_at: int = 400) -> pd.DataFrame:
    start = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)  # a Monday, ~9am ET
    rows = []
    price = 100.0
    for i in range(n_minutes):
        ts = start + timedelta(minutes=i)
        # gentle drift, then a sharp deliberate spike at `spike_at` that must
        # never be visible to a snapshot built before that index
        if i == spike_at:
            price += 50.0
        else:
            price += 0.01 * ((i % 7) - 3)
        o = price
        h = price + 0.5
        l = price - 0.5
        c = price + 0.05
        rows.append(dict(timestamp=ts, open=o, high=h, low=l, close=c, volume=100 + i))
        price = c
    return pd.DataFrame(rows)


def test_recent_bars_never_include_future() -> None:
    df = _make_synthetic_bars()
    features = compute_mtf_features(df)
    cutoff_idx = 250  # well before the spike at 400
    T = df["timestamp"].iloc[cutoff_idx]

    feat_row = features.loc[T] if T in features.index else None
    snap = build_snapshot(
        timestamp=T, symbol="TEST", price=float(df["close"].iloc[cutoff_idx]),
        bars_df=df, feat_row=feat_row, levels={"L1": 100.0},
        level_name="L1", setup_type="cross", setup_direction="long",
    )

    assert all(b[0] <= T for b in snap.recent_bars), (
        "MarketSnapshot.recent_bars leaked a bar timestamped after T"
    )
    print("PASS: recent_bars never includes a bar after T")


def test_snapshot_identical_whether_or_not_future_data_exists() -> None:
    df_full = _make_synthetic_bars()
    cutoff_idx = 250
    T = df_full["timestamp"].iloc[cutoff_idx]

    df_truncated = df_full[df_full["timestamp"] <= T].reset_index(drop=True)

    features_full = compute_mtf_features(df_full)
    features_trunc = compute_mtf_features(df_truncated)

    feat_row_full = features_full.loc[T] if T in features_full.index else None
    feat_row_trunc = features_trunc.loc[T] if T in features_trunc.index else None

    kwargs = dict(
        timestamp=T, symbol="TEST", price=float(df_full["close"].iloc[cutoff_idx]),
        levels={"L1": 100.0}, level_name="L1", setup_type="cross", setup_direction="long",
    )
    snap_full = build_snapshot(bars_df=df_full, feat_row=feat_row_full, **kwargs)
    snap_trunc = build_snapshot(bars_df=df_truncated, feat_row=feat_row_trunc, **kwargs)

    assert snap_full.recent_bars == snap_trunc.recent_bars, "recent_bars differ with/without future data"
    assert snap_full.macd_15m_hist == snap_trunc.macd_15m_hist, "macd_15m_hist leaked future data"
    assert snap_full.macd_15m_bottom_recent == snap_trunc.macd_15m_bottom_recent
    assert snap_full.macd_15m_top_recent == snap_trunc.macd_15m_top_recent
    assert snap_full.rsi_60m == snap_trunc.rsi_60m, "rsi_60m leaked future data"
    assert snap_full.macd_4h_hist == snap_trunc.macd_4h_hist, "macd_4h_hist leaked future data"
    assert snap_full.rsi_4h == snap_trunc.rsi_4h, "rsi_4h leaked future data"
    assert snap_full.atr == snap_trunc.atr, "atr leaked future data"
    print("PASS: snapshot at T is identical whether or not future bars exist in the source data")


def test_spike_after_cutoff_does_not_move_snapshot_price() -> None:
    df = _make_synthetic_bars(spike_at=260)  # spike shortly after our cutoff
    cutoff_idx = 250
    T = df["timestamp"].iloc[cutoff_idx]
    price_at_T = float(df["close"].iloc[cutoff_idx])

    features = compute_mtf_features(df)
    feat_row = features.loc[T] if T in features.index else None
    snap = build_snapshot(
        timestamp=T, symbol="TEST", price=price_at_T, bars_df=df, feat_row=feat_row,
        levels={"L1": 100.0}, level_name="L1", setup_type="fail", setup_direction="short",
    )
    assert snap.price == price_at_T
    assert snap.recent_bars[-1][0] == T
    assert snap.recent_bars[-1][4] == price_at_T  # close of the last visible bar is T's close, not the later spike
    print("PASS: a spike placed just after T never appears in the snapshot built at T")


if __name__ == "__main__":
    test_recent_bars_never_include_future()
    test_snapshot_identical_whether_or_not_future_data_exists()
    test_spike_after_cutoff_does_not_move_snapshot_price()
    print("\nAll no-lookahead checks passed.")
