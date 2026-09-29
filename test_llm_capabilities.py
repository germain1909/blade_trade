#!/usr/bin/env python3
"""
Phase 2 Step 0 — verify the local LM Studio / Qwen3-VL endpoint before
building anything on top of it.

TEST 1: text connectivity (trivial completion)
TEST 2: structured JSON output — tries response_format json_object/json_schema
        if the server honors it, falls back to prompt-only + manual parse,
        and reports which path actually worked (don't assume compliance)
TEST 3: vision — a real point-in-time-safe chart (last N bars ending at a
        chosen historical timestamp T, no bars after T) sent as an image,
        asking the model to describe chart structure

Uses the fixed endpoint/model given for this project — does not guess:
    base_url = http://127.0.0.1:1234/v1
    model    = qwen/qwen3-vl-4b

Run directly: python3 test_llm_capabilities.py
"""

from __future__ import annotations

import base64
import io
import json
import time

import pandas as pd
from openai import OpenAI

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-4b"

client = OpenAI(base_url=BASE_URL, api_key="lm-studio", timeout=60.0)


def test_1_text() -> None:
    print("=" * 70)
    print("TEST 1 — Text connectivity")
    print("=" * 70)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Respond with exactly: OK"},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    latency = time.time() - t0
    content = resp.choices[0].message.content
    print(f"  latency: {latency:.2f}s")
    print(f"  raw response: {content!r}")
    ok = content is not None and "OK" in content.upper()
    print(f"  PASS: {ok}")
    print()


def test_2_structured_json() -> None:
    print("=" * 70)
    print("TEST 2 — Structured JSON output")
    print("=" * 70)
    prompt = 'Return ONLY a JSON object, no other text, exactly: {"status": "OK", "value": 1}'

    # Attempt A: response_format json_object — most reliable when the server honors it
    print("  Attempt A: response_format={'type': 'json_object'}")
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        latency = time.time() - t0
        raw = resp.choices[0].message.content
        print(f"    latency: {latency:.2f}s  raw: {raw!r}")
        parsed = json.loads(raw)
        print(f"    parsed OK: {parsed}")
        print("    -> response_format json_object IS supported and reliable")
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        print("    -> response_format json_object not usable, falling back")

    # Attempt B: plain prompt, manual json.loads with brace-slicing (matches the
    # fail-closed pattern already used in prompts/take_skip_v1.py)
    print("  Attempt B: plain prompt + manual parse (no response_format)")
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=50,
    )
    latency = time.time() - t0
    raw = resp.choices[0].message.content
    print(f"    latency: {latency:.2f}s  raw: {raw!r}")
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end]) if start != -1 and end != 0 else None
        print(f"    parsed: {parsed}")
        print(f"    PASS (manual parse): {parsed == {'status': 'OK', 'value': 1}}")
    except Exception as exc:
        print(f"    manual parse FAILED: {exc}")
    print()


def _make_point_in_time_chart_png(csv_path: str, end_ts: str, lookback_bars: int = 40) -> tuple[bytes, dict]:
    """Builds a simple candlestick-style chart using ONLY bars with
    timestamp <= end_ts — no future bars, no mplfinance dependency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    cutoff = pd.Timestamp(end_ts, tz="UTC")
    truncated = df[df["timestamp"] <= cutoff]
    window = truncated.tail(lookback_bars).reset_index(drop=True)
    assert len(window) > 0, "no bars found at/before end_ts — check end_ts against the file's range"
    assert window["timestamp"].max() <= cutoff, "point-in-time violation: a bar after cutoff leaked into the chart"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    for i, row in window.iterrows():
        color = "#2F9C8D" if row["close"] >= row["open"] else "#C85A42"
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1)
        ax.plot([i, i], [row["open"], row["close"]], color=color, linewidth=4)
    ax.set_title(f"Point-in-time chart, last {len(window)} bars ending {window['timestamp'].iloc[-1]}")
    ax.set_xlim(-1, len(window))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    meta = {
        "bars_in_chart": len(window),
        "first_bar": window["timestamp"].iloc[0].isoformat(),
        "last_bar": window["timestamp"].iloc[-1].isoformat(),
        "cutoff_requested": cutoff.isoformat(),
        "max_bar_ts_leq_cutoff": bool(window["timestamp"].max() <= cutoff),
    }
    return buf.read(), meta


def test_3_vision() -> None:
    print("=" * 70)
    print("TEST 3 — Vision")
    print("=" * 70)
    # Use a timestamp inside the known-dense NQ window, well clear of the gappy period.
    end_ts = "2025-10-15 14:00:00+00:00"
    png_bytes, meta = _make_point_in_time_chart_png("data/NQZ5_1m.csv", end_ts, lookback_bars=40)
    print(f"  chart built: {meta}")
    assert meta["max_bar_ts_leq_cutoff"], "point-in-time check failed — aborting before sending to the model"

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    print(f"  image size: {len(png_bytes)} bytes, base64 length: {len(b64)}")

    messages = [
        {"role": "system", "content": "You are a market chart analyst."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "This is a candlestick chart of recent 1-minute NQ futures price action. "
                    "Describe the basic chart structure you see: overall direction, whether it "
                    "looks like it's trending, ranging, or choppy, and roughly how many candles "
                    "are shown. Keep it to 3-4 sentences."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]

    t0 = time.time()
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2, max_tokens=300)
        latency = time.time() - t0
        content = resp.choices[0].message.content
        print(f"  latency: {latency:.2f}s")
        print(f"  response:\n{content}")
        print(f"  PASS: {bool(content and len(content.strip()) > 0)}")
    except Exception as exc:
        latency = time.time() - t0
        print(f"  FAILED after {latency:.2f}s: {type(exc).__name__}: {exc}")
    print()


if __name__ == "__main__":
    print(f"Testing {MODEL} @ {BASE_URL}\n")
    test_1_text()
    test_2_structured_json()
    test_3_vision()
