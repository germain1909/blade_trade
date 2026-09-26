#!/usr/bin/env python3
"""
Download futures 1-minute bars from Massive API and save as CSV.

Usage:
    python3 download_massive.py ESZ5                    # full contract history
    python3 download_massive.py ESZ5 --limit 100000     # more bars
    python3 download_massive.py MESM26                   # MES June 2026
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.massive.com/futures/v1/aggs"


def fetch_all(ticker: str, api_key: str, limit: int = 50000) -> list[dict]:
    all_bars = []
    url = f"{BASE_URL}/{ticker}?resolution=1min&limit={limit}&apiKey={api_key}"
    page = 1

    while url:
        print(f"  Page {page}: fetching...", end=" ", flush=True)
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK":
            print(f"Error: {data}")
            break

        results = data.get("results", [])
        print(f"{len(results):,} bars")
        all_bars.extend(results)

        next_url = data.get("next_url")
        if next_url and results:
            url = f"{next_url}&apiKey={api_key}"
            page += 1
            time.sleep(0.2)
        else:
            break

    return all_bars


def save_csv(bars: list[dict], path: Path) -> None:
    bars_sorted = sorted(bars, key=lambda b: b["window_start"])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars_sorted:
            ts = datetime.fromtimestamp(
                b["window_start"] / 1e9, tz=timezone.utc)
            w.writerow([
                ts.isoformat(),
                b["open"],
                b["high"],
                b["low"],
                b["close"],
                b["volume"],
            ])

    print(f"Saved {len(bars_sorted):,} bars → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", help="Contract ID (e.g. ESZ5, MESM26)")
    ap.add_argument("--limit", type=int, default=50000,
                    help="Bars per page (max 50000)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output path (default: data/{ticker}_1m.csv)")
    args = ap.parse_args()

    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("Set MASSIVE_API_KEY in .env")
        return

    print(f"Downloading {args.ticker} 1m bars...")
    bars = fetch_all(args.ticker, api_key, args.limit)

    if not bars:
        print("No data returned.")
        return

    ts_first = datetime.fromtimestamp(
        min(b["window_start"] for b in bars) / 1e9, tz=timezone.utc)
    ts_last = datetime.fromtimestamp(
        max(b["window_start"] for b in bars) / 1e9, tz=timezone.utc)
    print(f"Range: {ts_first.date()} → {ts_last.date()}  ({len(bars):,} bars)")

    out = Path(args.out) if args.out else Path(f"data/{args.ticker}_1m.csv")
    save_csv(bars, out)


if __name__ == "__main__":
    main()
