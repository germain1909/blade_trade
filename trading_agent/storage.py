from __future__ import annotations

import csv
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .projectx_client import Bar

logger = logging.getLogger(__name__)

_CSV_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(bars: list[Bar], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_FIELDS)
        for b in bars:
            writer.writerow([b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume])
    logger.info("Saved %d bars → %s", len(bars), path)


def load_csv(path: Path) -> list[Bar]:
    bars: list[Bar] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            bars.append(Bar(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            ))
    logger.info("Loaded %d bars ← %s", len(bars), path)
    return bars


def csv_path(
    data_dir: Path,
    contract_id: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1m",
) -> Path:
    """Build a canonical CSV path: data/<contract>_<start>_<end>_<tf>.csv"""
    safe = contract_id.replace(".", "_").replace("/", "_")
    fname = f"{safe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{timeframe}.csv"
    return data_dir / fname


# ---------------------------------------------------------------------------
# SQLite  (optional — use when you want queryable storage)
# ---------------------------------------------------------------------------

def save_sqlite(bars: list[Bar], db_path: Path, table: str = "bars") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                timestamp TEXT PRIMARY KEY,
                open      REAL NOT NULL,
                high      REAL NOT NULL,
                low       REAL NOT NULL,
                close     REAL NOT NULL,
                volume    INTEGER NOT NULL
            )
        """)
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table}" VALUES (?,?,?,?,?,?)',
            [
                (b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume)
                for b in bars
            ],
        )
    logger.info("Saved %d bars → %s [table=%s]", len(bars), db_path, table)


def load_sqlite(db_path: Path, table: str = "bars") -> list[Bar]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f'SELECT timestamp,open,high,low,close,volume FROM "{table}" ORDER BY timestamp'
        ).fetchall()
    bars = [
        Bar(
            timestamp=datetime.fromisoformat(r[0]),
            open=float(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            volume=int(r[5]),
        )
        for r in rows
    ]
    logger.info("Loaded %d bars ← %s [table=%s]", len(bars), db_path, table)
    return bars


# ---------------------------------------------------------------------------
# Monthly Excel
# ---------------------------------------------------------------------------

def save_monthly_excel(
    bars: list[Bar],
    data_dir: Path,
    prefix: str,
    timeframe: str = "1m",
) -> list[Path]:
    """
    Group bars by calendar month and save one Excel file per month.

    Files are named:  <prefix>_<YYYY-MM>_<timeframe>.xlsx
    e.g.              MES_2025-06_1m.xlsx

    Returns the list of paths written.
    """
    if not bars:
        logger.warning("save_monthly_excel: no bars to save.")
        return []

    # Group bars by YYYY-MM
    monthly: dict[str, list[Bar]] = defaultdict(list)
    for bar in bars:
        key = bar.timestamp.strftime("%Y-%m")
        monthly[key].append(bar)

    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for month_key in sorted(monthly):
        month_bars = monthly[month_key]
        df = pd.DataFrame([
            {
                "timestamp": b.timestamp.isoformat(),
                "open":      b.open,
                "high":      b.high,
                "low":       b.low,
                "close":     b.close,
                "volume":    b.volume,
            }
            for b in month_bars
        ])

        path = data_dir / f"{prefix}_{month_key}_{timeframe}.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        logger.info("Saved %d bars → %s", len(month_bars), path)
        written.append(path)

    logger.info("Wrote %d monthly Excel files to %s", len(written), data_dir)
    return written
