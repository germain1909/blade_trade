from __future__ import annotations

"""
Pre-session Points of Interest (POI) Scanner
=============================================

Identifies the 3 highest-confluence price levels for the upcoming session
by stacking institutional reference levels and finding where they cluster.

Levels sourced:
  - PDH / PDL / PDC / PD midpoint    previous complete Globex day
  - Asia high / low / mid             19:00 ET (prior) → 23:55 ET (prior)
  - London high / low / mid           03:30 ET → 09:00 ET
  - ATR levels ±25% … ±100%          daily open ± (pct × 14-period ATR)

Confluence = number of individual levels landing within `tolerance_ticks`
of each other.  The 3 groups with the highest confluence score become
the session POIs, ranked closest-to-furthest from current price as a
tiebreaker.

Usage (backtesting against saved bars):
    from trading_agent.poi_scanner import scan_poi, print_poi_report

    pois, levels = scan_poi(
        bars          = all_1m_bars,          # list[Bar], sorted oldest→newest
        scan_time     = datetime(2026, 6, 8, 8, 30, tzinfo=ET),
        current_price = 7420.00,
        daily_atr     = 86.13,
    )
    print_poi_report(pois, levels)

Usage (live — pass bars from ProjectX API):
    bars = client.retrieve_bars(contract_id, start, end)
    pois, levels = scan_poi(bars, scan_time=datetime.now(ET), ...)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .projectx_client import Bar

_ET = ZoneInfo("America/New_York")

_ATR_PCTS: list[float] = [-1.00, -0.75, -0.50, -0.25, 0.00, +0.25, +0.50, +0.75, +1.00]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionLevels:
    """All raw price levels extracted from session data."""
    pdh:          float
    pdl:          float
    pdc:          float
    pd_mid:       float
    asia_high:    float
    asia_low:     float
    asia_mid:     float
    london_high:  float
    london_low:   float
    london_mid:   float
    day_open:     float
    daily_atr:    float
    atr_levels:   dict[str, float]   # "+75%" → price
    scan_time:    datetime
    current_price: float

    # Coverage metadata (for diagnostics)
    prev_day_bar_count:    int = 0
    asia_bar_count:        int = 0
    london_bar_count:      int = 0


@dataclass
class LevelRef:
    name:  str
    price: float


@dataclass
class POI:
    """A clustered confluence zone."""
    rank:             int
    price:            float        # centroid of the cluster
    confluence_score: int
    levels:           list[LevelRef]
    bias:             str          # "resistance" | "support" | "pivot"
    distance_pts:     float        # absolute distance from current price
    above_price:      bool
    nearest_atr:      Optional[str]

    @property
    def label(self) -> str:
        return " / ".join(l.name for l in self.levels)


# ─────────────────────────────────────────────────────────────────────────────
# Session time helpers
# ─────────────────────────────────────────────────────────────────────────────

def _et(ts: datetime) -> datetime:
    return ts.astimezone(_ET)


def _slice(bars: list[Bar], start: datetime, end: datetime) -> list[Bar]:
    """Return bars whose ET timestamp falls in [start, end)."""
    return [b for b in bars if start <= _et(b.timestamp) < end]


def _prev_globex_session(scan_time: datetime) -> tuple[datetime, datetime]:
    """
    Return (start, end) ET datetimes for the most recently COMPLETED
    CME Globex session before scan_time.

    A Globex session opens at 18:00 ET on day D-1 and closes at 17:00 ET
    on day D.  We walk backwards from scan_time to find the last closed one.
    """
    st = _et(scan_time)
    d  = st.date()

    # If scan is before 18:00 ET today, the session that closed most recently
    # ended at 17:00 ET today (or yesterday if we haven't hit 17:00 yet).
    # Walk back until we find a weekday close.
    for offset in range(1, 8):
        close_day   = d - timedelta(days=offset - 1)
        close_et    = datetime(close_day.year, close_day.month, close_day.day,
                               17, 0, tzinfo=_ET)
        open_day    = close_day - timedelta(days=1)
        open_et     = datetime(open_day.year, open_day.month, open_day.day,
                               18, 0, tzinfo=_ET)
        if close_et <= st and close_day.weekday() < 5:   # Mon-Fri
            return open_et, close_et

    raise ValueError(f"Could not find a previous Globex session before {scan_time}")


def _current_asia_window(scan_time: datetime) -> tuple[datetime, datetime]:
    """
    Asia session for the current overnight: 19:00 ET yesterday → 23:55 ET yesterday.
    If scan_time is before 23:55 ET yesterday, we use whatever bars exist up to scan_time.
    """
    st    = _et(scan_time)
    today = st.date()
    prev       = today - timedelta(days=1)
    asia_start = datetime(prev.year, prev.month, prev.day, 19,  0, tzinfo=_ET)
    asia_end   = datetime(prev.year, prev.month, prev.day, 23, 55, tzinfo=_ET)
    return asia_start, min(asia_end, st)


def _current_london_window(scan_time: datetime) -> tuple[datetime, datetime]:
    """London session: 03:30 ET → 09:00 ET today."""
    st    = _et(scan_time)
    today = st.date()
    lon_start = datetime(today.year, today.month, today.day, 3, 30, tzinfo=_ET)
    lon_end   = datetime(today.year, today.month, today.day, 9,  0, tzinfo=_ET)
    return lon_start, min(lon_end, st)


# ─────────────────────────────────────────────────────────────────────────────
# Session level extraction
# ─────────────────────────────────────────────────────────────────────────────

def _session_hl(bars: list[Bar]) -> tuple[float, float, float]:
    """Return (high, low, last_close) for a bar list.  Raises if empty."""
    if not bars:
        raise ValueError("No bars in session window")
    h = max(b.high  for b in bars)
    l = min(b.low   for b in bars)
    c = bars[-1].close
    return h, l, c


def compute_session_levels(
    bars:          list[Bar],
    scan_time:     datetime,
    current_price: float,
    daily_atr:     float,
    day_open:      Optional[float] = None,
) -> SessionLevels:
    """
    Slice `bars` into session windows and compute all reference levels.

    Parameters
    ----------
    bars          : 1-minute bars sorted oldest → newest (at least 2 days)
    scan_time     : the moment you're running the scan (e.g. 08:30 ET)
    current_price : latest market price (used for bias and distance)
    daily_atr     : 14-period Wilder ATR of the daily range (pass from
                    _build_daily_states or compute externally)
    day_open      : today's opening price; defaults to current_price if None
    """
    if day_open is None:
        day_open = current_price

    # ── Previous Globex session ──────────────────────────────────────────────
    pd_start, pd_end = _prev_globex_session(scan_time)
    pd_bars          = _slice(bars, pd_start, pd_end)
    if not pd_bars:
        raise ValueError(
            f"No bars found for previous Globex session "
            f"({pd_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{pd_end.strftime('%Y-%m-%d %H:%M')} ET). "
            f"Ensure your bar data covers at least 2 trading days."
        )
    pdh, pdl, pdc = _session_hl(pd_bars)

    # ── Asia session ─────────────────────────────────────────────────────────
    as_start, as_end = _current_asia_window(scan_time)
    asia_bars        = _slice(bars, as_start, as_end)
    if asia_bars:
        ash, asl, _ = _session_hl(asia_bars)
    else:
        # Fallback: use overnight portion of previous session
        ash = pdh; asl = pdl

    # ── London session ───────────────────────────────────────────────────────
    lon_start, lon_end = _current_london_window(scan_time)
    lon_bars           = _slice(bars, lon_start, lon_end)
    if lon_bars:
        lnh, lnl, _ = _session_hl(lon_bars)
    else:
        lnh = ash; lnl = asl

    # ── ATR levels from today's open ─────────────────────────────────────────
    atr_levels = {
        (f"{'+' if p >= 0 else ''}{int(p * 100)}%"): round(day_open + p * daily_atr, 2)
        for p in _ATR_PCTS
    }

    return SessionLevels(
        pdh           = round(pdh, 2),
        pdl           = round(pdl, 2),
        pdc           = round(pdc, 2),
        pd_mid        = round((pdh + pdl) / 2, 2),
        asia_high     = round(ash, 2),
        asia_low      = round(asl, 2),
        asia_mid      = round((ash + asl) / 2, 2),
        london_high   = round(lnh, 2),
        london_low    = round(lnl, 2),
        london_mid    = round((lnh + lnl) / 2, 2),
        day_open      = round(day_open, 2),
        daily_atr     = round(daily_atr, 2),
        atr_levels    = atr_levels,
        scan_time     = scan_time,
        current_price = current_price,
        prev_day_bar_count = len(pd_bars),
        asia_bar_count     = len(asia_bars),
        london_bar_count   = len(lon_bars),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POI confluence finder
# ─────────────────────────────────────────────────────────────────────────────

def _all_level_refs(lvl: SessionLevels) -> list[LevelRef]:
    """Flatten all named levels into a single list."""
    refs: list[LevelRef] = [
        LevelRef("PDH",         lvl.pdh),
        LevelRef("PDL",         lvl.pdl),
        LevelRef("PDC",         lvl.pdc),
        LevelRef("PD Mid",      lvl.pd_mid),
        LevelRef("Asia High",   lvl.asia_high),
        LevelRef("Asia Low",    lvl.asia_low),
        LevelRef("Asia Mid",    lvl.asia_mid),
        LevelRef("London High", lvl.london_high),
        LevelRef("London Low",  lvl.london_low),
        LevelRef("London Mid",  lvl.london_mid),
    ]
    for label, price in lvl.atr_levels.items():
        refs.append(LevelRef(f"ATR {label}", price))
    return refs


def _cluster(refs: list[LevelRef], tolerance: float) -> list[list[LevelRef]]:
    """
    Group refs into clusters where every member is within `tolerance` of
    the cluster centroid.  Simple single-pass greedy approach.
    """
    sorted_refs = sorted(refs, key=lambda r: r.price)
    clusters: list[list[LevelRef]] = []

    for ref in sorted_refs:
        placed = False
        for cluster in clusters:
            centroid = sum(r.price for r in cluster) / len(cluster)
            if abs(ref.price - centroid) <= tolerance:
                cluster.append(ref)
                placed = True
                break
        if not placed:
            clusters.append([ref])

    return clusters


def _nearest_atr(price: float, atr_levels: dict[str, float]) -> Optional[str]:
    best_label = None
    best_dist  = float("inf")
    for label, lvl_price in atr_levels.items():
        d = abs(price - lvl_price)
        if d < best_dist:
            best_dist  = d
            best_label = label
    return best_label


def _bias(price: float, current: float, atr: float) -> str:
    pivot_zone = atr * 0.10   # within 10% of ATR = "pivot"
    if abs(price - current) <= pivot_zone:
        return "pivot"
    return "resistance" if price > current else "support"


def find_poi(
    levels:            SessionLevels,
    tolerance_ticks:   int = 8,
    tick_size:         float = 0.25,
    top_n:             int = 3,
    min_atr_distance:  float = 0.0,    # skip levels too close to current price
) -> list[POI]:
    """
    Return the top `top_n` POIs ranked by confluence score.
    Ties broken by distance from current price (closer = higher priority).

    Parameters
    ----------
    tolerance_ticks  : levels within this many ticks cluster together
    min_atr_distance : fraction of daily ATR — skip levels closer than this
                       (e.g. 0.10 skips anything within 10% ATR of current px)
    """
    tolerance = tolerance_ticks * tick_size
    refs      = _all_level_refs(levels)
    clusters  = _cluster(refs, tolerance)

    pois: list[POI] = []
    for cluster in clusters:
        if len(cluster) < 1:
            continue
        centroid = sum(r.price for r in cluster) / len(cluster)
        dist     = abs(centroid - levels.current_price)

        # Optionally skip levels too close to current price
        if min_atr_distance > 0 and dist < min_atr_distance * levels.daily_atr:
            continue

        pois.append(POI(
            rank             = 0,
            price            = round(centroid, 2),
            confluence_score = len(cluster),
            levels           = sorted(cluster, key=lambda r: r.price),
            bias             = _bias(centroid, levels.current_price, levels.daily_atr),
            distance_pts     = round(dist, 2),
            above_price      = centroid > levels.current_price,
            nearest_atr      = _nearest_atr(centroid, levels.atr_levels),
        ))

    # Rank: primary = confluence score desc, secondary = distance asc
    pois.sort(key=lambda p: (-p.confluence_score, p.distance_pts))
    pois = pois[:top_n]
    for i, p in enumerate(pois):
        p.rank = i + 1

    return pois


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def scan_poi(
    bars:            list[Bar],
    scan_time:       datetime,
    current_price:   float,
    daily_atr:       float,
    day_open:        Optional[float] = None,
    tolerance_ticks: int   = 8,
    tick_size:       float = 0.25,
    top_n:           int   = 3,
) -> tuple[list[POI], SessionLevels]:
    """
    One-call interface: compute session levels then find top POIs.

    Returns (pois, levels) so callers can inspect raw levels too.
    """
    levels = compute_session_levels(
        bars          = bars,
        scan_time     = scan_time,
        current_price = current_price,
        daily_atr     = daily_atr,
        day_open      = day_open,
    )
    pois = find_poi(
        levels           = levels,
        tolerance_ticks  = tolerance_ticks,
        tick_size        = tick_size,
        top_n            = top_n,
    )
    return pois, levels


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_poi_report(pois: list[POI], levels: SessionLevels) -> None:
    st = levels.scan_time.astimezone(_ET)
    sep = "═" * 72

    print(sep)
    print(f"  POI SCAN  —  {st.strftime('%A %Y-%m-%d %H:%M ET')}")
    print(f"  Current price: {levels.current_price:.2f}  |  "
          f"Daily open: {levels.day_open:.2f}  |  ATR: {levels.daily_atr:.2f}")
    print(sep)

    print()
    print("  SESSION REFERENCE LEVELS")
    print("  " + "─" * 60)
    rows = [
        ("Prev Day High",  levels.pdh),
        ("Prev Day Low",   levels.pdl),
        ("Prev Day Close", levels.pdc),
        ("Prev Day Mid",   levels.pd_mid),
        ("Asia High",      levels.asia_high),
        ("Asia Low",       levels.asia_low),
        ("Asia Mid",       levels.asia_mid),
        ("London High",    levels.london_high),
        ("London Low",     levels.london_low),
        ("London Mid",     levels.london_mid),
    ]
    for name, price in rows:
        dist  = price - levels.current_price
        arrow = "↑" if dist > 0 else ("↓" if dist < 0 else "=")
        print(f"  {name:<18}  {price:>9.2f}   {arrow} {abs(dist):>6.2f} pts")

    print()
    print("  ATR LEVELS FROM OPEN")
    print("  " + "─" * 60)
    for label, price in sorted(levels.atr_levels.items(),
                                key=lambda x: float(x[0].replace('%','').replace('+',''))):
        dist  = price - levels.current_price
        arrow = "↑" if dist > 0 else ("↓" if dist < 0 else "→")
        print(f"  ATR {label:<8}  {price:>9.2f}   {arrow} {abs(dist):>6.2f} pts")

    print()
    print(sep)
    print("  TOP POINTS OF INTEREST")
    print(sep)

    BIAS_ICON = {"resistance": "▼ SHORT", "support": "▲ LONG", "pivot": "◆ EITHER"}

    for poi in pois:
        icon  = BIAS_ICON.get(poi.bias, poi.bias)
        arrow = "↑" if poi.above_price else "↓"
        print()
        print(f"  POI #{poi.rank}  —  {poi.price:.2f}   [{icon}]   "
              f"Confluence: {poi.confluence_score}   "
              f"{arrow} {poi.distance_pts:.2f} pts from current")
        print(f"  Nearest ATR level: {poi.nearest_atr or '—'}")
        print(f"  Contributing levels:")
        for ref in poi.levels:
            diff = ref.price - poi.price
            print(f"    • {ref.name:<18}  {ref.price:.2f}"
                  f"  ({'+' if diff >= 0 else ''}{diff:.2f})")
        print()
        print(f"  TRADE PLAN:")
        if poi.bias == "resistance":
            print(f"    Short on rejection at {poi.price:.2f}")
            print(f"    Long on reclaim above {poi.price:.2f}")
        elif poi.bias == "support":
            print(f"    Long on hold at {poi.price:.2f}")
            print(f"    Short on break below {poi.price:.2f}")
        else:
            print(f"    Watch for MACD cross at {poi.price:.2f} — trade in direction of break")
        print("  " + "─" * 60)

    print()
    print(f"  Data coverage:  Prev day {levels.prev_day_bar_count} bars  |  "
          f"Asia {levels.asia_bar_count} bars  |  London {levels.london_bar_count} bars")
    print(sep)
