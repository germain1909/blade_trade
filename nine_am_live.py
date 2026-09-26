#!/usr/bin/env python3
"""
9AM Live Playbook — Run at 10:00 AM ET

Fetches today's 9AM 1-hour candle from TopStep, analyzes it against
historical patterns, and prints the playbook for the 10AM hour.

Usage:
    python3 nine_am_live.py                  # MNQ (default)
    python3 nine_am_live.py --contract CON.F.US.MES.U26
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from trading_agent.config import load_config
from trading_agent.projectx_client import ProjectXClient, Bar

_ET = ZoneInfo("America/New_York")


def get_nine_am_candle(client: ProjectXClient, contract_id: str) -> dict:
    """Fetch today's 1m bars from 9:00-10:00 ET and build the 9AM OHLC."""
    now_et = datetime.now(_ET)
    today = now_et.date()

    nine_start = datetime(today.year, today.month, today.day, 9, 0, tzinfo=_ET)
    nine_end   = datetime(today.year, today.month, today.day, 10, 0, tzinfo=_ET)

    if now_et < nine_end:
        print(f"WARNING: 9AM candle not complete yet (current time: {now_et.strftime('%H:%M ET')})")
        print(f"         Results will be partial.\n")

    bars = client.retrieve_bars(
        contract_id=contract_id,
        start_time=nine_start.astimezone(timezone.utc),
        end_time=nine_end.astimezone(timezone.utc),
        unit=ProjectXClient.UNIT_MINUTE,
        unit_number=1,
    )

    if not bars:
        print(f"No bars returned for {contract_id} 9:00-10:00 ET today.")
        sys.exit(1)

    nine_o = bars[0].open
    nine_h = max(b.high for b in bars)
    nine_l = min(b.low for b in bars)
    nine_c = bars[-1].close
    nine_range = nine_h - nine_l

    return {
        "open": nine_o,
        "high": nine_h,
        "low": nine_l,
        "close": nine_c,
        "range": nine_range,
        "bars": len(bars),
    }


def analyze_candle(candle: dict) -> dict:
    """Classify the 9AM candle and return the analysis."""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    rng = candle["range"]

    body = c - o
    if rng > 0:
        body_pct = abs(body) / rng
        close_pos = (c - l) / rng
    else:
        body_pct = 0
        close_pos = 0.5

    # Candle type
    if body_pct < 0.1:
        candle_type = "DOJI"
    elif body > 0:
        candle_type = "BULLISH"
    else:
        candle_type = "BEARISH"

    # Close position
    if close_pos > 0.66:
        close_zone = "UPPER"
    elif close_pos < 0.33:
        close_zone = "LOWER"
    else:
        close_zone = "MIDDLE"

    return {
        "candle_type": candle_type,
        "close_zone": close_zone,
        "close_pos_pct": close_pos * 100,
        "body_pct": body_pct * 100,
        "midpoint": (h + l) / 2.0,
    }


def print_playbook(candle: dict, analysis: dict, contract: str) -> None:
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    rng = candle["range"]
    mid = analysis["midpoint"]
    ct = analysis["candle_type"]
    cz = analysis["close_zone"]

    print(f"{'='*60}")
    print(f"  9AM PLAYBOOK — {contract}")
    print(f"  {datetime.now(_ET).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}")

    print(f"\n  9AM CANDLE:")
    print(f"    Open:  {o:.2f}")
    print(f"    High:  {h:.2f}  ← resistance")
    print(f"    Low:   {l:.2f}  ← support")
    print(f"    Close: {c:.2f}")
    print(f"    Range: {rng:.2f} pts")
    print(f"    Mid:   {mid:.2f}")

    print(f"\n  READING:")
    print(f"    Type:     {ct}")
    print(f"    Close:    {cz} third ({analysis['close_pos_pct']:.0f}% of range)")

    print(f"\n  {'─'*50}")

    # Playbook based on historical stats
    if ct == "BULLISH" and cz == "UPPER":
        print(f"  BIAS: STRONG LONG")
        print(f"  Historical: 9H breaks 85% of the time on NQ")
        print(f"")
        print(f"  PLAY 1 (primary): Buy breakout above {h:.2f}")
        print(f"    Entry:  1m close above {h:.2f}")
        print(f"    Stop:   {mid:.2f} (midpoint)")
        print(f"    Target: {h + (h - mid):.2f} (1R) or {h + 2*(h - mid):.2f} (2R)")
        print(f"")
        print(f"  PLAY 2 (dip buy): Buy rejection at {l:.2f}")
        print(f"    Entry:  1m rejection wick at {l:.2f}, close above")
        print(f"    Stop:   {l - rng*0.25:.2f}")
        print(f"    Target: {h:.2f}")

    elif ct == "BEARISH" and cz == "LOWER":
        print(f"  BIAS: STRONG SHORT")
        print(f"  Historical: 9L breaks 77% of the time on NQ")
        print(f"")
        print(f"  PLAY 1 (primary): Sell breakout below {l:.2f}")
        print(f"    Entry:  1m close below {l:.2f}")
        print(f"    Stop:   {mid:.2f} (midpoint)")
        print(f"    Target: {l - (mid - l):.2f} (1R) or {l - 2*(mid - l):.2f} (2R)")
        print(f"")
        print(f"  PLAY 2 (rip sell): Sell rejection at {h:.2f}")
        print(f"    Entry:  1m rejection wick at {h:.2f}, close below")
        print(f"    Stop:   {h + rng*0.25:.2f}")
        print(f"    Target: {l:.2f}")

    elif ct == "BULLISH" and cz != "UPPER":
        print(f"  BIAS: LEAN LONG (not strong — close not in upper third)")
        print(f"  Historical: High breaks ~70% after bullish 9AM")
        print(f"")
        print(f"  PLAY: Buy breakout above {h:.2f}")
        print(f"    Entry:  1m close above {h:.2f}")
        print(f"    Stop:   {mid:.2f}")
        print(f"    Target: {h + (h - mid):.2f}")
        print(f"")
        print(f"  CAUTION: Close not in upper third — wait for confirmation.")

    elif ct == "BEARISH" and cz != "LOWER":
        print(f"  BIAS: LEAN SHORT (not strong — close not in lower third)")
        print(f"  Historical: Low breaks ~63% after bearish 9AM")
        print(f"")
        print(f"  PLAY: Sell breakout below {l:.2f}")
        print(f"    Entry:  1m close below {l:.2f}")
        print(f"    Stop:   {mid:.2f}")
        print(f"    Target: {l - (mid - l):.2f}")
        print(f"")
        print(f"  CAUTION: Close not in lower third — wait for confirmation.")

    elif ct == "DOJI":
        print(f"  BIAS: NONE — WAIT")
        print(f"  Historical: Doji = coin flip. 50/50 either direction.")
        print(f"")
        print(f"  PLAY: Wait for the FIRST clean break")
        print(f"    If 1m closes above {h:.2f} → long, stop {mid:.2f}")
        print(f"    If 1m closes below {l:.2f} → short, stop {mid:.2f}")
        print(f"    If neither by 10:30 → sit out")

    else:
        print(f"  BIAS: NEUTRAL")
        print(f"  Mixed signals. Wait for a clear break.")
        print(f"    Above {h:.2f} → long")
        print(f"    Below {l:.2f} → short")

    print(f"\n  KEY LEVELS:")
    print(f"    {h:.2f}  ← 9AM High (break = long)")
    print(f"    {c:.2f}  ← 9AM Close")
    print(f"    {mid:.2f}  ← Midpoint (stop level)")
    print(f"    {o:.2f}  ← 9AM Open")
    print(f"    {l:.2f}  ← 9AM Low (break = short)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=os.getenv("TOPSTEP_NQ_CONTRACT_ID", "CON.F.US.MNQ.U26"),
                    help="Contract ID")
    args = ap.parse_args()

    config = load_config()
    client = ProjectXClient(config)
    client.authenticate()

    print(f"Fetching 9AM candle for {args.contract}...")
    candle = get_nine_am_candle(client, args.contract)
    analysis = analyze_candle(candle)
    print_playbook(candle, analysis, args.contract)


if __name__ == "__main__":
    main()
