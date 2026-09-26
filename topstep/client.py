from dotenv import load_dotenv
import os
import time
import requests
from datetime import datetime, timedelta, timezone, date, time as dt_time
from typing import List, Dict, Any, Optional


load_dotenv() # automatically finds .env file
TOPSTEP_BASE = os.getenv("TOPSTEP_BASE", "https://api.topstepx.com")  
TOPSTEP_USERNAME = os.getenv("TOPSTEP_USERNAME")
TOPSTEP_API_KEY = os.getenv("TOPSTEP_API_KEY")

class TopstepClient:
    def __init__(self):
        if not TOPSTEP_USERNAME or not TOPSTEP_API_KEY:
            raise ValueError("TOPSTEP_USERNAME or TOPSTEP_API_KEY is missing. Check your .env file.")

        # one persistent session per client instance
        self._session = requests.Session()
        self._token = None
        self._token_exp = 0  # epoch timestamp when token expires
        self._base_url = TOPSTEP_BASE

    def login_with_key(self):
        """
        Logs in using a username + apiKey.
        Matches the Postman loginKey endpoint.
        """
        url = f"{TOPSTEP_BASE}/api/Auth/loginKey"

        payload = {
            "userName": TOPSTEP_USERNAME,
            "apiKey": TOPSTEP_API_KEY,
        }

        headers = {
            "accept": "text/plain",
            "Content-Type": "application/json",
        }

        resp = self._session.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", False):
            msg = data.get("errorMessage", "Unknown login error")
            raise RuntimeError(f"Topstep login failed: {msg}")

        token = data["token"]      # <-- THIS is the correct token
        self._token = token
        self._token_exp = time.time() + 23 * 3600

        print("Topstep: Login success!")

    def _ensure_token(self):
        now = time.time()

        # 1. No token stored yet
        if not self._token:
            print("Topstep: No token found. Logging in...")
            self.login_with_key()
            self.validate()
            return

        # 2. Token expired
        elif now > self._token_exp:
            print("Topstep: Token expired. Logging in again...")
            self.login_with_key()
            self.validate()
            return

        # 3. Token exists and is not expired
        else:
            # Convert expiration timestamp to readable date/time
            exp_str = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(self._token_exp)
            )

            print(f"Topstep: Token is valid. It will expire at {exp_str}.")

            # Optional: Validate token with the server
            if not self.validate():
                print("Topstep: Server rejected token. Re-authenticating...")
                self.login_with_key()
                self.validate()

    def validate(self) -> bool:
        """
        Validates the current Bearer token and receives a new validated token.
        Returns True if validation succeeded, False otherwise.
        """
        if not self._token:
            print("Topstep: Validate called with no token present.")
            return False

        print("Topstep: Validating token...")

        url = f"{TOPSTEP_BASE}/api/Auth/validate"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }

        resp = self._session.post(url, headers=headers, timeout=15)
        resp.raise_for_status()

        data = resp.json()

        # If success = false → validation failure
        if not data.get("success", False):
            msg = data.get("errorMessage", "Unknown validation error")
            print(f"Topstep: Token validation failed: {msg}")
            return False

        # Validation success → set new token (Topstep always returns one)
        new_token = data.get("token")
        if new_token:
            self._token = new_token
            self._token_exp = time.time() + 23 * 3600
            print("Topstep: Token validated and refreshed.")
        else:
            print("Topstep: Token valid (no refresh token returned).")

        return True
    
    def get_latest_contract_id(self, search_text: str, live: bool = False) -> str | None:
        """
        Returns ONLY the contract ID string for the active contract.
        Example: 'CON.F.US.MGC.Z25'
        """
        self._ensure_token()

        url = f"{self._base_url}/api/Contract/search"

        payload = {
            "live": live,
            "searchText": search_text
        }

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        resp = self._session.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        contracts = data.get("contracts", [])

        if not contracts:
            print(f"No contracts returned for searchText='{search_text}'.")
            return None

        # They already give only the active contract
        contract_id = contracts[0].get("id")
        print(f"Topstep: Active contract for '{search_text}' → {contract_id}")

        return contract_id


    def get_yesterdays_daily_bar_for_symbol(self, symbol_root: str, live: bool = False):
        """
        Retrieve the most recent CLOSED daily bar for the active contract
        of a given futures symbol root (e.g. 'MGC', 'GC', 'CL').

        Returns:
            A dict like:
            {
                "t": "2025-11-21T00:00:00+00:00",
                "o": 4075.0,
                "h": 4101.1,
                "l": 4019.0,
                "c": 4062.8,
                "v": 455342
            }
            or None if no bars found.
        """
        # 1) Get active contractId for this symbol (e.g. "CON.F.US.MGC.Z25")
        contract_id = self.get_latest_contract_id(symbol_root, live=live)
        if not contract_id:
            print(f"Topstep: Could not resolve active contract for '{symbol_root}'.")
            return None

        # 2) Ensure we have a valid token
        self._ensure_token()

        # # 3) Build a date range wide enough to include the last trading day
        # end_dt = datetime.now(timezone.utc).replace(microsecond=0)
        # start_dt = end_dt - timedelta(days=7)  # wide window, we'll still only take 1 bar


        # 3) Build exact 1-day window for "yesterday"
        #    startTime = yesterday 00:00:00Z
        #    endTime   = (yesterday + 1 day) 00:00:00Z
        today_utc = datetime.now(timezone.utc).date()
        yesterday = today_utc - timedelta(days=1)

        start_dt = datetime.combine(yesterday, dt_time(0, 0, 0), tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1)  # next calendar day (Sat after Fri, etc.)

        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": start_dt.isoformat().replace("+00:00", "Z"),
            "endTime": end_dt.isoformat().replace("+00:00", "Z"),
            "unit": 4,                 # 4 = Day
            "unitNumber": 1,           # 1 day per bar
            "limit": 1,                # only need the latest completed bar
            "includePartialBar": False # excludes today's forming bar
        }

        print(payload)
        url = f"{self._base_url}/api/History/retrieveBars"

        headers = {
            "accept": "application/json",   # response body is JSON with "bars": [...]
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        print(f"Topstep: Getting most recent closed daily bar for {symbol_root} ({contract_id})...")
        resp = self._session.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        bars = data.get("bars", [])

        if not bars:
            print("Topstep: No daily bars returned.")
            return None

        # Because limit=1, this should just be the most recent completed day
        bar = bars[-1]
        print('yesterdays bar',bar)
        return bar
    
    
    def get_daily_bar_for_contract_on_date(
        self,
        contract_id: str,
        target_date: date,
        live: bool = False,
    ):
        """
        Retrieve the CLOSED daily bar for a specific contractId
        on a given calendar date (UTC).

        Args:
            contract_id: e.g. "CON.F.US.ENQ.Z25"
            target_date: datetime.date for the day you care about
            live: whether to pull from the live or sim environment

        Returns:
            A dict like:
            {
                "t": "2025-05-19T00:00:00+00:00",
                "o": ...,
                "h": ...,
                "l": ...,
                "c": ...,
                "v": ...
            }
            or None if no bar for that date is found.
        """
        # Ensure auth
        self._ensure_token()

        # Build a small window around the target date to be safe
        # start_dt = datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        # end_dt = datetime.combine(target_date + timedelta(days=2), datetime.min.time(), tzinfo=timezone.utc)

        # EXACT 1-DAY WINDOW
        # startTime = target_date 00:00:00Z
        # endTime   = target_date + 1 day 00:00:00Z
        start_dt = datetime.combine(target_date, dt_time(0, 0, 0), tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1)


        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": start_dt.isoformat().replace("+00:00", "Z"),
            "endTime": end_dt.isoformat().replace("+00:00", "Z"),
            "unit": 4,                 # 4 = Day
            "unitNumber": 1,           # 1 day per bar
            "limit": 10,               # small window, a few bars max
            "includePartialBar": False # closed bars only
        }

        url = f"{self._base_url}/api/History/retrieveBars"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        print(
            f"Topstep: Getting daily bar for {contract_id} "
            f"on {target_date.isoformat()}..."
        )

        resp = self._session.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        bars = data.get("bars", [])

        if not bars:
            print("Topstep: No daily bars returned.")
            return None

        # Since window is exactly one day, the returned bar *is* the one we want
        bar = bars[-1]
        print("Topstep: Found daily bar:", bar)
        return bar
    
    @staticmethod
    def cme_session_utc(session_date: date):
        """
        Given a CME 'session date' (the trade date),
        return the UTC start/end for the metals session:

            23:00 UTC on session_date
            → 22:00 UTC on session_date + 1

        This matches 18:00 ET → 17:00 ET, but we stay in UTC
        so we avoid all timezone/DST headaches.
        """
        # 23:00 UTC on the given date
        start_utc = datetime(
            session_date.year,
            session_date.month,
            session_date.day,
            23, 0, 1,
            tzinfo=timezone.utc,
        )

        # 23-hour session -> next day 22:00 UTC
        end_utc = start_utc + timedelta(hours=22)

        return start_utc, end_utc
    

    def get_minute_bars_for_cme_session(
        self,
        contract_id: str,
        session_date: date,
        live: bool = False,
        unit_number: int = 1,  # 1 = 1-minute bars, 10 = 10-minute, etc.
    ) -> List[Dict[str, Any]]:
        """
        Fetch CLOSED intraday minute bars for a CME futures contract
        over a single metals session:

            23:00 UTC on session_date
            → 22:00 UTC on session_date + 1

        Args:
            contract_id: e.g. "CON.F.US.MGC.Z25"
            session_date: the CME trade date for the session
            live: query live vs sim environment
            unit_number: minutes per bar (1 for 1-minute, 10 for 10-minute)

        Returns:
            A list of bar dicts like:
            {
                "t": "2025-11-25T21:59:00+00:00",
                "o": ...,
                "h": ...,
                "l": ...,
                "c": ...,
                "v": ...
            }
            Possibly empty if no bars exist for that session.
        """
        # Ensure token is valid
        self._ensure_token()

        # Get session start/end in UTC
        start_utc, end_utc = self.cme_session_utc(session_date)

        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": start_utc.isoformat().replace("+00:00", "Z"),
            "endTime": end_utc.isoformat().replace("+00:00", "Z"),
            "unit": 2,                    # 2 = Minute
            "unitNumber": unit_number,    # e.g. 1 = 1m, 10 = 10m
            "limit": 2000,                # safe upper bound for intraday bars
            "includePartialBar": False,   # only closed bars
        }
        print('germain1',payload)
        url = f"{self._base_url}/api/History/retrieveBars"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        print(
            f"Topstep: Getting {unit_number}-minute bars for {contract_id} "
            f"session {session_date.isoformat()} "
            f"{payload['startTime']} → {payload['endTime']}..."
        )

        resp = self._session.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        bars = data.get("bars", []) or []

        # Sort ascending by time just to be sure
        bars_sorted = sorted(bars, key=lambda b: b["t"])

        print(f"Topstep: Retrieved {len(bars_sorted)} bars.")
        return bars_sorted
    
    