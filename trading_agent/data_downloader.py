from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone

from .projectx_client import Bar, ProjectXClient

logger = logging.getLogger(__name__)

# Roll-month config per symbol. Equity index micros (MES, MNQ, M2K, MYM) trade
# quarterly (Mar/Jun/Sep/Dec). Metals (GC, MGC) trade bimonthly — Sep/Mar are
# not active contract months, so they need their own roll schedule.
_QUARTERLY_MONTHS = (3, 6, 9, 12)
_BIMONTHLY_MONTHS = (2, 4, 6, 8, 10, 12)
_SYMBOL_ROLL_MONTHS = {
    "GC": _BIMONTHLY_MONTHS,
    "MGC": _BIMONTHLY_MONTHS,
}
_MONTH_CODE = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
_ROLL_DAY = 10  # switch front-month around the 10th of expiry month


# ---------------------------------------------------------------------------
# Single-contract download  —  one API call per trade day
# ---------------------------------------------------------------------------

def download_1m_bars(
    client: ProjectXClient,
    contract_id: str,
    start: datetime,
    end: datetime,
    live: bool = False,
) -> list[Bar]:
    """
    Download 1-minute bars for a single contract between start and end.

    Iterates Monday → Friday, making one API call per trade day.
    Each call fetches the full CME Globex session for that day:
        23:00 UTC the previous evening  →  22:00 UTC that day  (23 hours)

    Saturday and Sunday are skipped — no session starts those days.
    """
    start_date = _as_date(start)
    end_date   = _as_date(end)

    all_bars: list[Bar] = []
    cursor = start_date

    while cursor <= end_date:
        if cursor.weekday() < 5:  # 0=Mon … 4=Fri, skip Sat/Sun
            session_start, session_end = _session_window(cursor)
            logger.info("  %s  session %s", contract_id, cursor.isoformat())

            bars = client.retrieve_bars(
                contract_id=contract_id,
                start_time=session_start,
                end_time=session_end,
                unit=ProjectXClient.UNIT_MINUTE,
                unit_number=1,
                live=live,
                limit=2000,          # 23 h × 60 min = 1 380 bars max per session
                include_partial_bar=False,
            )
            logger.info("    → %d bars", len(bars))
            all_bars.extend(bars)

        cursor += timedelta(days=1)

    result = _dedup_sorted(all_bars)
    logger.info(
        "Done: %d bars for %s (%s → %s)",
        len(result), contract_id, start_date, end_date,
    )
    return result


# ---------------------------------------------------------------------------
# Continuous (multi-contract) download  —  handles quarterly rollovers
# ---------------------------------------------------------------------------

def download_continuous_1m_bars(
    client: ProjectXClient,
    symbol: str,
    start: datetime,
    end: datetime,
    live: bool = False,
) -> list[Bar]:
    """
    Download 1-minute bars across quarterly contract rollovers.

    For each weekday in start→end:
      1. Determine the correct front-month contract for that date.
      2. Fetch that day's CME session bars.

    Works for quarterly CME micro futures: MES, MNQ, M2K, MYM.
    Returns a single merged sorted list across all contracts.
    """
    start_date = _as_date(start)
    end_date   = _as_date(end)

    all_bars: list[Bar] = []
    cursor = start_date
    current_contract: str | None = None

    while cursor <= end_date:
        if cursor.weekday() < 5:
            contract_id, _ = _front_month_contract(symbol.upper(), cursor)

            if contract_id != current_contract:
                logger.info("Contract roll → %s  on %s", contract_id, cursor.isoformat())
                current_contract = contract_id

            session_start, session_end = _session_window(cursor)
            logger.info("  %s  session %s", contract_id, cursor.isoformat())

            bars = client.retrieve_bars(
                contract_id=contract_id,
                start_time=session_start,
                end_time=session_end,
                unit=ProjectXClient.UNIT_MINUTE,
                unit_number=1,
                live=live,
                limit=2000,
                include_partial_bar=False,
            )
            logger.info("    → %d bars", len(bars))
            all_bars.extend(bars)

        cursor += timedelta(days=1)

    result = _dedup_sorted(all_bars)
    logger.info(
        "Continuous download complete: %d bars for %s (%s → %s)",
        len(result), symbol, start_date, end_date,
    )
    return result


# ---------------------------------------------------------------------------
# Session window
# ---------------------------------------------------------------------------

def _session_window(trade_date: date) -> tuple[datetime, datetime]:
    """
    CME Globex 23-hour session for a trade date.

    Opens at 23:00 UTC the previous calendar evening (e.g. Sunday night for Monday).
    Closes at 22:00 UTC on the trade date.

    Monday June 10  →  June 9 23:00 UTC  →  June 10 22:00 UTC
    """
    prev = trade_date - timedelta(days=1)
    start = datetime(prev.year, prev.month, prev.day, 23, 0, 0, tzinfo=timezone.utc)
    end   = datetime(trade_date.year, trade_date.month, trade_date.day, 22, 0, 0, tzinfo=timezone.utc)
    return start, end


# ---------------------------------------------------------------------------
# Quarterly roll helpers
# ---------------------------------------------------------------------------

def _front_month_contract(symbol: str, on_date: date) -> tuple[str, date]:
    """Return (contract_id, last_active_date) for the front-month contract."""
    y = on_date.year
    roll_months = _SYMBOL_ROLL_MONTHS.get(symbol, _QUARTERLY_MONTHS)

    for exp_month in roll_months:
        roll_date = date(y, exp_month, _ROLL_DAY)
        if on_date < roll_date:
            code = _MONTH_CODE[exp_month]
            contract_id = f"CON.F.US.{symbol}.{code}{y % 100:02d}"
            return contract_id, roll_date - timedelta(days=1)

    # Past the last roll of the year → first roll month of next year
    next_month = roll_months[0]
    code = _MONTH_CODE[next_month]
    contract_id = f"CON.F.US.{symbol}.{code}{(y + 1) % 100:02d}"
    last_day = date(y + 1, next_month, _ROLL_DAY) - timedelta(days=1)
    return contract_id, last_day


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_date(dt: datetime) -> date:
    return dt.date() if isinstance(dt, datetime) else dt


def _dedup_sorted(bars: list[Bar]) -> list[Bar]:
    seen: set[datetime] = set()
    out: list[Bar] = []
    for bar in sorted(bars, key=lambda b: b.timestamp):
        if bar.timestamp not in seen:
            seen.add(bar.timestamp)
            out.append(bar)
    return out
