#!/usr/bin/env python3
"""
TEMPORARY instrumentation for a one-off QuantHarness smoke test against the
local LM Studio model. Not a permanent part of the pipeline -- delete after
use. Makes zero changes to indicator_agent.py/pattern_agent.py/trend_agent.py/
decision_agent.py/graph_setup.py/graph_util.py: all timing is captured by
monkey-patching BaseChatModel.invoke at runtime and inferring which agent is
calling from the Python call stack (frame filenames), so none of QuantHarness's
own prompts/logic/behavior are touched.

Purpose: does the existing pipeline run end-to-end against the local model,
does each agent produce sensible output, how long does each call/agent take,
and is total latency reasonable for BladeTrade's ~15-minute review cadence.
NOT a backtest, NOT a trading-performance test.
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime, timedelta

import requests
import yfinance as yf
from langchain_core.language_models.chat_models import BaseChatModel

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "qwen/qwen3-vl-4b"
AGENT_FILES = ("indicator_agent.py", "pattern_agent.py", "trend_agent.py", "decision_agent.py", "graph_setup.py")

call_log: list[dict] = []


def _agent_from_stack() -> str:
    for frame in traceback.extract_stack():
        for candidate in AGENT_FILES:
            if frame.filename.endswith(candidate):
                return candidate[:-3]
    return "unknown"


def _is_vision_input(input_) -> bool:
    try:
        msgs = input_ if isinstance(input_, list) else getattr(input_, "to_messages", lambda: [input_])()
        for m in msgs:
            content = getattr(m, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image_url":
                        return True
    except Exception:
        pass
    return False


_original_invoke = BaseChatModel.invoke


def _instrumented_invoke(self, input_, config=None, **kwargs):
    agent = _agent_from_stack()
    is_vision = _is_vision_input(input_)
    call_number = len(call_log) + 1
    t0 = time.time()
    success = True
    error = None
    usage = None
    try:
        result = _original_invoke(self, input_, config=config, **kwargs)
        meta = getattr(result, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or getattr(result, "usage_metadata", None)
        return result
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.time() - t0
        record = {
            "call_number": call_number, "agent": agent, "vision": is_vision,
            "start": t0, "elapsed_s": round(elapsed, 2), "success": success, "error": error,
            "usage": usage,
        }
        call_log.append(record)
        kind = "VISION" if is_vision else "text  "
        status = "OK" if success else "FAIL"
        print(f"  [call {call_number:>2}] agent={agent:<16} {kind}  {elapsed:>6.2f}s  {status}"
              + (f"  ERROR={error}" if error else ""))


BaseChatModel.invoke = _instrumented_invoke


def verify_lm_studio() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        print(f"LM Studio reachable at {BASE_URL}. Loaded models: {models}")
        if MODEL not in models:
            print(f"WARNING: expected model {MODEL!r} not found in loaded models.")
        return True
    except Exception as exc:
        print(f"LM Studio NOT reachable at {BASE_URL}: {exc}")
        return False


def main() -> None:
    print("=" * 90)
    print("QuantHarness local-LLM smoke test -- integration/latency only, not a backtest")
    print("=" * 90)

    print("\nStep 1: verify LM Studio is reachable with the existing configuration...")
    if not verify_lm_studio():
        print("ABORT: LM Studio unreachable.")
        sys.exit(1)

    print("\nStep 2: fetch a minimal-but-sufficient data window (4 calendar days of 1h NQ bars --")
    print("empirically confirmed to yield ~48 bars, clearing web_interface.py's hardcoded")
    print("`df.tail(45)` floor; 3 days only yielded 25 bars due to the weekend gap)...")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=4)
    raw = yf.download("NQ=F", start=start_dt, end=end_dt, interval="1h", progress=False)
    print(f"  fetched {len(raw)} raw 1h bars ({start_dt} to {end_dt})")

    # Reuse the exact same column-shaping web_interface.py's own fetch_yfinance_data_with_datetime
    # does for a 1h timeframe, so this is a faithful stand-in for what the real /api/analyze path
    # would produce -- not a redesigned data path.
    df = raw.reset_index()
    if isinstance(df.columns, type(raw.columns)) and hasattr(df.columns, "get_level_values"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Datetime": "Datetime", "index": "Datetime"})
    if "Datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Datetime"})

    print(f"\nStep 3: run the SAME, unmodified QuantHarness analysis pipeline "
          f"(web_interface.analyzer.run_analysis)...\n")

    from web_interface import analyzer

    t_start = time.time()
    results = analyzer.run_analysis(df, "Nasdaq Futures", "1h")
    total_elapsed = time.time() - t_start

    print("\n" + "=" * 90)
    print(f"TOTAL run_analysis time: {total_elapsed:.2f}s   ({len(call_log)} LLM calls)")
    print("=" * 90)

    by_agent: dict[str, list[dict]] = {}
    for rec in call_log:
        by_agent.setdefault(rec["agent"], []).append(rec)

    print("\nPer-agent totals:")
    for agent, recs in by_agent.items():
        agent_total = sum(r["elapsed_s"] for r in recs)
        n_fail = sum(1 for r in recs if not r["success"])
        print(f"  {agent:<16} calls={len(recs):<3} total={agent_total:>7.2f}s  failures={n_fail}")

    print("\nFinal analysis result (success/error + top-level keys):")
    print(f"  success: {results.get('success')}")
    if not results.get("success"):
        print(f"  error: {results.get('error')}")
    else:
        for k in results:
            if k not in ("pattern_chart", "trend_chart"):
                v = results[k]
                v_str = str(v)
                print(f"  {k}: {v_str[:300]}")

    import json
    with open("smoke_test_results.json", "w") as f:
        json.dump({
            "total_elapsed_s": total_elapsed,
            "call_log": call_log,
            "results_success": results.get("success"),
            "results_error": results.get("error") if not results.get("success") else None,
            "results_keys": [k for k in results if k not in ("pattern_chart", "trend_chart")],
            "results_non_image": {k: v for k, v in results.items() if k not in ("pattern_chart", "trend_chart")},
        }, f, indent=2, default=str)
    print("\nFull call log + results written to smoke_test_results.json")


if __name__ == "__main__":
    main()
