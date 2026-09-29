#!/usr/bin/env python3
"""
Explicit, configurable session-window definitions for the Opportunity
Scanner — nothing about session boundaries is a bare literal buried inside
scanner/tracker logic; everything reads from this object.

Asia's freeze point (midnight ET) and London's start (immediately after,
no gap) are fixed by project definition. Everything else is a starting
default to confirm/override once real fire-rate data is in hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime


@dataclass
class SessionConfig:
    # Globex trading-day boundary — also the PDH/PDL window, same convention
    # already used by weekly_levels.py / trading_agent/poi_scanner.py
    # (bars at/after this hour belong to the NEXT calendar day's Globex day).
    globex_day_start_et: dtime = dtime(18, 0)   # prior calendar day
    globex_day_end_et:   dtime = dtime(17, 0)   # this calendar day

    asia_start_et:   dtime = dtime(18, 0)       # Globex open
    asia_end_et:     dtime = dtime(0, 0)        # fixed — freezes at midnight ET

    london_start_et: dtime = dtime(0, 0)        # fixed — begins the instant Asia freezes, no gap
    london_end_et:   dtime = dtime(9, 0)        # DEFAULT ONLY — confirm/override before trusting results

    ny_start_et:     dtime = dtime(8, 0)        # DEFAULT ONLY — overlaps London by design (08:00-09:00)
    ny_end_et:       dtime = dtime(17, 0)       # DEFAULT ONLY

    def active_sessions(self, t: dtime) -> frozenset[str]:
        """Which session labels are active at ET time t. A set, not a single
        label, so overlaps (e.g. London/NY 08:00-09:00) are preserved rather
        than collapsed to one arbitrary choice."""
        sessions = set()
        if self._in_window(t, self.asia_start_et, self.asia_end_et):
            sessions.add("asia")
        if self._in_window(t, self.london_start_et, self.london_end_et):
            sessions.add("london")
        if self._in_window(t, self.ny_start_et, self.ny_end_et):
            sessions.add("ny")
        if not sessions:
            sessions.add("overnight")
        return frozenset(sessions)

    @staticmethod
    def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
        if start <= end:
            return start <= t < end
        return t >= start or t < end  # window wraps midnight
