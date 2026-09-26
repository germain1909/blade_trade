from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from .config import Config

logger = logging.getLogger(__name__)

_TOKEN_TTL = 23 * 3600  # seconds — refresh just before the 24-hour expiry


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass
class Account:
    id: int
    name: str
    balance: float
    can_trade: bool
    is_visible: bool


@dataclass
class Contract:
    id: str
    name: str
    description: str
    tick_size: float
    tick_value: float
    active_contract: bool
    symbol_id: str


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class ProjectXClient:
    """Thin wrapper around the ProjectX Gateway REST API."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._session = requests.Session()
        self._token: str | None = None
        self._token_acquired_at: float | None = None  # monotonic seconds

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """Obtain a JWT session token using username + API key."""
        logger.info("Authenticating with ProjectX API …")
        data = self._post_raw("/api/Auth/loginKey", {
            "userName": self._cfg.username,
            "apiKey": self._cfg.api_key,
        })
        self._set_token(data["token"])
        logger.info("Authentication successful.")

    def validate(self) -> None:
        """Validate the current token and receive a refreshed one."""
        logger.info("Refreshing session token …")
        data = self._post("/api/Auth/validate", {})
        new = data.get("newToken") or data.get("token")
        if new:
            self._set_token(new)
            logger.info("Token refreshed.")

    def ensure_authenticated(self) -> None:
        """Authenticate if needed, or refresh if near expiry."""
        if self._token is None:
            self.authenticate()
        elif (
            self._token_acquired_at is not None
            and (time.monotonic() - self._token_acquired_at) > _TOKEN_TTL
        ):
            self.validate()

    # ------------------------------------------------------------------
    # Accounts
    # ------------------------------------------------------------------

    def search_accounts(self, only_active: bool = True) -> list[Account]:
        self.ensure_authenticated()
        data = self._post("/api/Account/search", {"onlyActiveAccounts": only_active})
        return [
            Account(
                id=a["id"],
                name=a["name"],
                balance=float(a["balance"]),
                can_trade=bool(a["canTrade"]),
                is_visible=bool(a["isVisible"]),
            )
            for a in data.get("accounts", []) or []
        ]

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def search_contracts(self, search_text: str, live: bool = False) -> list[Contract]:
        """Full-text search; returns up to 20 contracts."""
        self.ensure_authenticated()
        data = self._post("/api/Contract/search", {"searchText": search_text, "live": live})
        return [self._parse_contract(c) for c in data.get("contracts", []) or []]

    def search_contract_by_id(self, contract_id: str) -> Contract:
        self.ensure_authenticated()
        data = self._post("/api/Contract/searchById", {"contractId": contract_id})
        # The envelope sometimes nests under "contract", sometimes flat
        raw = data.get("contract") or {
            k: v for k, v in data.items()
            if k not in {"success", "errorCode", "errorMessage"}
        }
        return self._parse_contract(raw)

    def available_contracts(self, live: bool = False) -> list[Contract]:
        """List all available contracts for the given environment."""
        self.ensure_authenticated()
        data = self._post("/api/Contract/available", {"live": live})
        return [self._parse_contract(c) for c in data.get("contracts", []) or []]

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------

    # Bar unit constants
    UNIT_SECOND = 1
    UNIT_MINUTE = 2
    UNIT_HOUR   = 3
    UNIT_DAY    = 4
    UNIT_WEEK   = 5
    UNIT_MONTH  = 6

    def retrieve_bars(
        self,
        contract_id: str,
        start_time: datetime,
        end_time: datetime,
        unit: int = UNIT_MINUTE,
        unit_number: int = 1,
        live: bool = False,
        limit: int = 20_000,
        include_partial_bar: bool = False,
    ) -> list[Bar]:
        """
        Fetch OHLCV bars.  Max 20,000 bars per request.
        For 1-minute bars: unit=UNIT_MINUTE (2), unit_number=1.
        """
        self.ensure_authenticated()
        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": _to_iso(start_time),
            "endTime": _to_iso(end_time),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": include_partial_bar,
        }
        # print("DEBUG payload:", payload)
        data = self._post("/api/History/retrieveBars", payload)
        # print("DEBUG response:", data)
        return [
            Bar(
                timestamp=datetime.fromisoformat(b["t"]),
                open=float(b["o"]),
                high=float(b["h"]),
                low=float(b["l"]),
                close=float(b["c"]),
                volume=int(b["v"]),
            )
            for b in data.get("bars", []) or []
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_token(self, token: str) -> None:
        self._token = token
        self._token_acquired_at = time.monotonic()

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _post_raw(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST and raise on HTTP or API error (no token required)."""
        url = f"{self._cfg.base_url}{path}"
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries + 1):
            resp = self._session.post(url, json=body, headers=self._headers(), timeout=30)
            if resp.status_code == 429 and attempt < max_retries:
                wait = float(resp.headers.get("Retry-After", backoff))
                logger.warning("429 rate limited on %s — retrying in %.1fs", path, wait)
                time.sleep(wait)
                backoff *= 2
                continue
            resp.raise_for_status()
            break
        data: dict[str, Any] = resp.json()
        if not data.get("success"):
            raise RuntimeError(
                f"API error [{path}]: code={data.get('errorCode')} — {data.get('errorMessage')}"
            )
        return data

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if not self._token:
            raise RuntimeError("Not authenticated — call authenticate() first.")
        return self._post_raw(path, body)

    @staticmethod
    def _parse_contract(c: dict[str, Any]) -> Contract:
        return Contract(
            id=c["id"],
            name=c["name"],
            description=c.get("description", ""),
            tick_size=float(c["tickSize"]),
            tick_value=float(c["tickValue"]),
            active_contract=bool(c.get("activeContract", False)),
            symbol_id=str(c.get("symbolId", "")),
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_iso(dt: datetime) -> str:
    """Return ISO-8601 string in UTC with Z suffix."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")
