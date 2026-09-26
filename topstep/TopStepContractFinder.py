from __future__ import annotations
from dataclasses import dataclass
from datetime import date


@dataclass
class TopstepContractFinder:
    """
    Heuristic contract guesser for common futures symbols.

    Supports:
      - MGC / GC  (micro/mini gold, G-J-M-Z cycle with custom roll windows)
      - CL / MCL  (crude, monthly with mid-month style roll)
      - NQ / MNQ  (Nasdaq, quarterly H/M/U/Z)
      - 6E / M6E  (Euro FX, quarterly H/M/U/Z)

    The defaults are chosen to roughly match typical front-month conventions
    and can be tuned per your data provider if needed.
    """

    # --- MGC / GC "front month" boundaries ---
    mgc_mid_feb_day: int = 15   # switch G -> J
    mgc_mid_may_day: int = 15   # switch J -> M
    mgc_mid_oct_day: int = 15   # switch M -> Z

    # --- CL monthly roll ---
    # crude roughly rolls mid-month to next/next+1 month
    cl_mid_month_roll_day: int = 15

    # Month code mapping (CME-style)
    MONTH_CODES = {
        1: "F",  # Jan
        2: "G",  # Feb
        3: "H",  # Mar
        4: "J",  # Apr
        5: "K",  # May
        6: "M",  # Jun
        7: "N",  # Jul
        8: "Q",  # Aug
        9: "U",  # Sep
        10: "V", # Oct
        11: "X", # Nov
        12: "Z", # Dec
    }

    # ---------- public API ----------

    def guess_contract(self, symbol: str, dt: date) -> str:
        """
        Return a full contract code like 'MGCJ25', 'CLM25', 'NQM25', '6EU25'
        for the given symbol and date.
        """
        sym = symbol.upper()

        if sym in ("MGC", "GC"):
            month_code, year = self._guess_mgc(dt)
        elif sym in ("CL", "MCL"):
            month_code, year = self._guess_cl(dt)
        elif sym in ("NQ", "MNQ"):
            month_code, year = self._guess_quarterly(dt)
        elif sym in ("6E", "M6E"):
            month_code, year = self._guess_quarterly(dt)
        else:
            raise ValueError(f"Unsupported symbol {symbol!r}")

        return f"{sym}{month_code}{year % 100:02d}"

    # ---------- internal helpers ----------

    def _guess_mgc(self, dt: date) -> tuple[str, int]:
        """
        Approximate front-month contract for MGC/GC on a given date.

        Schedule (simplified):
          G (Feb):  early Jan   – mid-Feb
          J (Apr):  mid-Feb     – mid-May
          M (Jun):  mid-May     – mid-Oct
          Z (Dec):  mid-Oct     – end Dec
        """
        y, m, d = dt.year, dt.month, dt.day

        if (m == 1) or (m == 2 and d < self.mgc_mid_feb_day):
            # Feb contract of current year
            return "G", y

        if (m == 2 and d >= self.mgc_mid_feb_day) or m in (3, 4) or (
            m == 5 and d < self.mgc_mid_may_day
        ):
            # Apr contract
            return "J", y

        if (m == 5 and d >= self.mgc_mid_may_day) or m in (6, 7, 8, 9) or (
            m == 10 and d < self.mgc_mid_oct_day
        ):
            # Jun contract
            return "M", y

        # mid-Oct through Dec = Dec contract
        return "Z", y

    def _guess_cl(self, dt: date) -> tuple[str, int]:
        """
        Approximate front-month contract for CL/MCL on a given date.

        Very simple rule of thumb:
          - If day < roll_day: use next calendar month
          - If day >= roll_day: use two months ahead
        This roughly matches mid-month crude roll conventions.
        """
        y, m, d = dt.year, dt.month, dt.day

        if d < self.cl_mid_month_roll_day:
            cy, cm = self._add_months(y, m, 1)
        else:
            cy, cm = self._add_months(y, m, 2)

        return self.MONTH_CODES[cm], cy

    def _guess_quarterly(self, dt: date) -> tuple[str, int]:
        """
        Approximate front-month for quarterly contracts (H/M/U/Z),
        used for NQ/MNQ and 6E/M6E.

        Windows (approx, adapted from CME rollover guides):
          H (Mar):  Dec 14 (prev yr) – Mar 14
          M (Jun):  Mar 15 – Jun 13
          U (Sep):  Jun 14 – Sep 12
          Z (Dec):  Sep 13 – Dec 13
          After Dec 13 → next year's H
        """
        y, m, d = dt.year, dt.month, dt.day

        if m in (1, 2) or (m == 3 and d <= 14):
            # March contract this year
            return "H", y

        if (m == 3 and d >= 15) or m in (4, 5) or (m == 6 and d <= 13):
            # June contract this year
            return "M", y

        if (m == 6 and d >= 14) or m in (7, 8) or (m == 9 and d <= 12):
            # Sep contract this year
            return "U", y

        # Z or next year's H
        if (m == 9 and d >= 13) or m in (10, 11) or (m == 12 and d <= 13):
            # December of this year
            return "Z", y

        # After Dec 13 → next year's March
        return "H", y + 1

    @classmethod
    def _add_months(cls, year: int, month: int, offset: int) -> tuple[int, int]:
        """Helper to add months while carrying year."""
        month += offset
        year += (month - 1) // 12
        month = (month - 1) % 12 + 1
        return year, month
