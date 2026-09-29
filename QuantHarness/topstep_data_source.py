"""
Adapter: TopstepClient's per-session 1-minute bars -> the exact
Datetime/Open/High/Low/Close/Volume dataframe shape web_interface.py's
run_analysis() already expects from the Yahoo Finance path. Purely
additive -- does not touch the existing yfinance code path at all.

Topstep's history API is session-scoped (one call per CME session, 23:00
UTC day N -> 22:00 UTC day N+1), so covering a date range means looping
over each session date in range and concatenating.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# topstep/ lives one directory above QuantHarness/ (a sibling package in the parent
# blade_trade repo) -- add the parent to sys.path so this resolves regardless of
# whether the process's cwd is QuantHarness/ (e.g. when web_interface.py runs).
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from topstep.client import TopstepClient

_RESAMPLE_RULE = {
    "1m": "1min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d",
    "1w": "1w", "1mo": "1ME",
}

_client: Optional[TopstepClient] = None


def _get_client() -> TopstepClient:
    global _client
    if _client is None:
        _client = TopstepClient()
    return _client


def _session_dates_covering(start_dt: datetime, end_dt: datetime) -> list[date]:
    """Every CME session date (per TopstepClient.cme_session_utc's convention,
    23:00 UTC day N -> 22:00 UTC day N+1) whose window overlaps [start_dt, end_dt]."""
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    # A session starting on session_date covers [23:00 session_date, 22:00 session_date+1).
    # Bars timestamped at/after start_dt-1day could still belong to a session we need.
    first_session = (start_dt - timedelta(days=1)).date()
    last_session = end_dt.date()

    dates = []
    d = first_session
    while d <= last_session:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def fetch_topstep_ohlc(
    symbol_root: str,
    start_dt: datetime,
    end_dt: datetime,
    timeframe: str,
    live: bool = True,
) -> pd.DataFrame:
    """Returns a dataframe with columns Datetime/Open/High/Low/Close/Volume
    (capitalized, Datetime as a plain column, ascending order) -- the same
    shape web_interface.py's yfinance path produces, ready for run_analysis().

    Raises ValueError if the symbol can't be resolved to an active Topstep
    contract, or RuntimeError if no bars were returned for the range.
    """
    client = _get_client()

    contract_id = client.get_latest_contract_id(symbol_root, live=live)
    if not contract_id:
        raise ValueError(
            f"Topstep: could not resolve an active contract for symbol root {symbol_root!r}. "
            f"Topstep only covers CME futures (e.g. NQ, MNQ, ES, MES, GC, MGC, CL, 6E) -- "
            f"non-futures assets (equities, indices, crypto) aren't available through this client."
        )

    session_dates = _session_dates_covering(start_dt, end_dt)

    all_bars = []
    for session_date in session_dates:
        bars = client.get_minute_bars_for_cme_session(
            contract_id=contract_id, session_date=session_date, live=live, unit_number=1,
        )
        all_bars.extend(bars)

    if not all_bars:
        raise RuntimeError(
            f"Topstep: no 1-minute bars returned for {symbol_root} ({contract_id}) "
            f"between {start_dt} and {end_dt}."
        )

    df = pd.DataFrame(all_bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df.sort_values("t").drop_duplicates(subset="t")

    start_utc = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
    end_utc = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)
    df = df[(df["t"] >= start_utc) & (df["t"] <= end_utc)]
    if df.empty:
        raise RuntimeError(
            f"Topstep: bars were returned for surrounding sessions, but none fell inside "
            f"the exact requested window {start_dt} to {end_dt}."
        )

    rule = _RESAMPLE_RULE.get(timeframe)
    if rule is None:
        raise ValueError(f"Topstep adapter: unsupported timeframe {timeframe!r}")

    if rule != "1min":
        resampled = (
            df.set_index("t")
            .resample(rule, label="right", closed="right")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
            .reset_index()
        )
    else:
        resampled = df.reset_index(drop=True)

    resampled = resampled.rename(columns={"t": "Datetime"})
    return resampled[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
