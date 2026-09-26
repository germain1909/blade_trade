from __future__ import annotations

"""
SignalR WebSocket connection for TopstepX real-time data.

Two hubs:
  Market hub  (rtc.topstepx.com/hubs/market)
    → GatewayQuote(contractId, data)   — best bid/ask/last per symbol
    → GatewayTrade(contractId, data)   — individual trade prints (for bar building)

  User hub    (rtc.topstepx.com/hubs/user)
    → GatewayUserOrder(data)           — order status changes / fills
    → GatewayUserPosition(data)        — position updates
    → GatewayUserTrade(data)           — trade fill confirmations with PnL

Exact field names taken from official ProjectX RTC API docs.

GatewayQuote fields : lastPrice, bestBid, bestAsk, timestamp, ...
GatewayTrade fields : price, volume, timestamp, type, symbolId
GatewayUserOrder    : id, status, fillVolume, filledPrice, customTag, ...
GatewayUserTrade    : price, profitAndLoss, fees, side, size, orderId, ...
GatewayUserPosition : type (1=Long 2=Short), size, averagePrice, contractId

NOTE: Market hub events pass TWO arguments [contractId, data_dict].
      User hub events pass ONE argument [data_dict].
      signalrcore delivers all hub-method arguments as a Python list.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from .config import Config
from .projectx_client import Bar

logger = logging.getLogger(__name__)

OnBarCallback      = Callable[[Bar], None]
OnQuoteCallback    = Callable[[float, float, float], None]  # bid, ask, last
OnOrderCallback    = Callable[[dict], None]
OnPositionCallback = Callable[[dict], None]
OnTradeCallback    = Callable[[dict], None]   # GatewayUserTrade (fill PnL)


# ─────────────────────────────────────────────────────────────────────────────
# 1-minute bar accumulator from tick stream
# ─────────────────────────────────────────────────────────────────────────────

class _BarAccumulator:
    """Accumulate GatewayTrade prints into completed 1-minute OHLCV bars."""

    def __init__(self) -> None:
        self._minute_start: Optional[datetime] = None
        self._open = self._high = self._low = self._close = 0.0
        self._volume: int = 0

    @staticmethod
    def _floor_minute(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0, tzinfo=ts.tzinfo)

    def update(self, ts: datetime, price: float, volume: int) -> Optional[Bar]:
        """
        Feed one trade print.
        Returns a completed Bar when the 1-minute boundary is crossed,
        otherwise None.
        """
        minute = self._floor_minute(ts)

        if self._minute_start is None:
            self._minute_start = minute
            self._open = self._high = self._low = self._close = price
            self._volume = volume
            return None

        if minute == self._minute_start:
            if price > self._high: self._high = price
            if price < self._low:  self._low  = price
            self._close   = price
            self._volume += volume
            return None

        # Minute rolled — emit the completed bar and start fresh
        completed = Bar(
            timestamp = self._minute_start,
            open      = self._open,
            high      = self._high,
            low       = self._low,
            close     = self._close,
            volume    = self._volume,
        )
        self._minute_start = minute
        self._open = self._high = self._low = self._close = price
        self._volume = volume
        return completed


# ─────────────────────────────────────────────────────────────────────────────
# Live connection
# ─────────────────────────────────────────────────────────────────────────────

class LiveConnection:
    """
    Manages both TopstepX SignalR hubs for a single session.

    Parameters
    ----------
    config       : app Config (rtc_base_url)
    account_id   : numeric account ID (needed for user hub subscriptions)
    contract_id  : e.g. "CON.F.US.MES.M25"
    jwt_token    : bearer token from ProjectXClient.authenticate()
    on_bar       : called with a completed 1m Bar each time a minute closes
    on_quote     : called with (bid, ask, last) on each GatewayQuote tick
    on_order     : called with the raw order dict on each GatewayUserOrder
    on_fill      : called with the raw trade dict on each GatewayUserTrade
    on_position  : called with the raw position dict on each GatewayUserPosition
    """

    def __init__(
        self,
        config:       Config,
        account_id:   int,
        contract_id:  str,
        jwt_token:    str,
        on_bar:       Optional[OnBarCallback]      = None,
        on_quote:     Optional[OnQuoteCallback]    = None,
        on_order:     Optional[OnOrderCallback]    = None,
        on_fill:      Optional[OnTradeCallback]    = None,
        on_position:  Optional[OnPositionCallback] = None,
    ) -> None:
        self._cfg          = config
        self._account_id   = account_id
        self._contract_id  = contract_id
        self._jwt          = jwt_token

        self._on_bar       = on_bar
        self._on_quote     = on_quote
        self._on_order     = on_order
        self._on_fill      = on_fill
        self._on_position  = on_position

        self._accumulator  = _BarAccumulator()
        self._market_conn  = None
        self._user_conn    = None
        self._stop_event   = threading.Event()
        self._last_price: float = 0.0
        self._quote_count: int  = 0
        self._trade_count: int  = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect both hubs and block until stop() is called."""
        try:
            from signalrcore.hub_connection_builder import HubConnectionBuilder
        except ImportError:
            raise RuntimeError(
                "signalrcore is not installed. Run:  pip install signalrcore"
            )

        self._market_conn = self._build_hub(
            url      = f"{self._cfg.rtc_base_url}/hubs/market",
            on_open  = self._market_on_open,
            handlers = {
                "GatewayQuote": self._handle_quote,
                "GatewayTrade": self._handle_trade,
            },
        )
        self._user_conn = self._build_hub(
            url      = f"{self._cfg.rtc_base_url}/hubs/user",
            on_open  = self._user_on_open,
            handlers = {
                "GatewayUserOrder":    self._handle_order,
                "GatewayUserPosition": self._handle_position,
                "GatewayUserTrade":    self._handle_fill,
            },
        )

        self._market_conn.start()
        self._user_conn.start()
        logger.info("LiveConnection: both hubs started")

        self._stop_event.wait()

        # Clean unsubscribe before disconnecting
        try:
            self._market_conn.send("UnsubscribeContractQuotes", [self._contract_id])
            self._market_conn.send("UnsubscribeContractTrades", [self._contract_id])
        except Exception:
            pass
        try:
            self._user_conn.send("UnsubscribeOrders",    [self._account_id])
            self._user_conn.send("UnsubscribePositions", [self._account_id])
            self._user_conn.send("UnsubscribeTrades",    [self._account_id])
            self._user_conn.send("UnsubscribeAccounts",  [])
        except Exception:
            pass

        self._market_conn.stop()
        self._user_conn.stop()
        logger.info("LiveConnection: both hubs stopped")

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def last_price(self) -> float:
        return self._last_price

    # ── Hub builder ───────────────────────────────────────────────────────────

    def _build_hub(self, url: str, on_open: Callable, handlers: dict):
        from signalrcore.hub_connection_builder import HubConnectionBuilder

        conn = (
            HubConnectionBuilder()
            .with_url(
                f"{url}?access_token={self._jwt}",
                options={
                    "verify_ssl":         True,
                    "skip_negotiation":   True,    # WebSocket only — matches JS docs
                    "headers":            {},
                },
            )
            .with_automatic_reconnect({
                "type":                "raw",
                "keep_alive_interval": 10,
                "reconnect_interval":   5,
                "max_attempts":        20,
            })
            .build()
        )
        conn.on_open(on_open)
        conn.on_error(lambda e: logger.error(f"SignalR error [{url}]: {e}"))
        for event, handler in handlers.items():
            conn.on(event, handler)
        return conn

    # ── Market hub ────────────────────────────────────────────────────────────

    def _market_on_open(self) -> None:
        logger.info(f"Market hub open — subscribing {self._contract_id}")
        # Each subscription sends a single string arg (contract_id)
        self._market_conn.send("SubscribeContractQuotes", [self._contract_id])
        self._market_conn.send("SubscribeContractTrades", [self._contract_id])

    def _handle_quote(self, args: list) -> None:
        """GatewayQuote(contractId, data) — args = [contract_id_str, data_dict]"""
        if len(args) < 2:
            return
        q = args[1]
        if not isinstance(q, dict):
            return
        bid  = float(q.get("bestBid",   0) or 0)
        ask  = float(q.get("bestAsk",   0) or 0)
        last = float(q.get("lastPrice", 0) or 0)
        if not last:
            return
        self._last_price = last
        self._quote_count += 1
        logger.info(f"QUOTE #{self._quote_count}  last={last}  bid={bid}  ask={ask}")

        # Build bars from quote ticks as fallback when GatewayTrade is absent.
        raw_ts = q.get("timestamp") or q.get("ts") or ""
        try:
            if raw_ts:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            else:
                ts = datetime.now(timezone.utc)
            completed = self._accumulator.update(ts, last, 1)
            if completed is not None:
                logger.debug(
                    f"BAR(q) {completed.timestamp.isoformat()}  "
                    f"O={completed.open}  H={completed.high}  "
                    f"L={completed.low}   C={completed.close}  V={completed.volume}"
                )
                if self._on_bar:
                    self._on_bar(completed)
        except Exception as exc:
            logger.debug(f"Quote bar build error: {exc}")

        if self._on_quote:
            self._on_quote(bid, ask, last)

    def _handle_trade(self, args: list) -> None:
        """GatewayTrade(contractId, data) — args = [contract_id_str, data_dict]"""
        if len(args) < 2:
            return
        t = args[1]
        if not isinstance(t, dict):
            return
        try:
            raw_ts = t.get("timestamp") or t.get("ts") or ""
            ts     = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            price  = float(t["price"])
            volume = int(t.get("volume", 1))
        except Exception as exc:
            logger.debug(f"Bad GatewayTrade payload: {t}  ({exc})")
            return

        self._last_price = price
        self._trade_count += 1
        logger.info(f"TRADE #{self._trade_count}  price={price}  vol={volume}")
        completed = self._accumulator.update(ts, price, volume)
        if completed is not None:
            logger.debug(
                f"BAR  {completed.timestamp.isoformat()}  "
                f"O={completed.open}  H={completed.high}  "
                f"L={completed.low}   C={completed.close}  V={completed.volume}"
            )
            if self._on_bar:
                self._on_bar(completed)

    # ── User hub ──────────────────────────────────────────────────────────────

    def _user_on_open(self) -> None:
        logger.info(f"User hub open — subscribing account {self._account_id}")
        self._user_conn.send("SubscribeAccounts",  [])
        self._user_conn.send("SubscribeOrders",    [self._account_id])
        self._user_conn.send("SubscribePositions", [self._account_id])
        self._user_conn.send("SubscribeTrades",    [self._account_id])

    def _handle_order(self, args: list) -> None:
        """GatewayUserOrder(data) — args = [data_dict]"""
        if not args:
            return
        order = args[0]
        status_map = {0: "None", 1: "Open", 2: "Filled", 3: "Cancelled",
                      4: "Expired", 5: "Rejected", 6: "Pending"}
        status = status_map.get(order.get("status", -1), str(order.get("status")))
        logger.info(
            f"ORDER  id={order.get('id')}  status={status}  "
            f"filled={order.get('fillVolume')}/{order.get('size')}  "
            f"fillPx={order.get('filledPrice')}  tag={order.get('customTag')!r}"
        )
        if self._on_order:
            self._on_order(order)

    def _handle_fill(self, args: list) -> None:
        """GatewayUserTrade(data) — trade fill with PnL"""
        if not args:
            return
        fill = args[0]
        side = "BUY" if fill.get("side") == 0 else "SELL"
        logger.info(
            f"FILL   id={fill.get('id')}  {side} {fill.get('size')}x "
            f"@ {fill.get('price')}  PnL={fill.get('profitAndLoss')}  "
            f"fees={fill.get('fees')}  orderId={fill.get('orderId')}"
        )
        if self._on_fill:
            self._on_fill(fill)

    def _handle_position(self, args: list) -> None:
        """GatewayUserPosition(data)"""
        if not args:
            return
        pos = args[0]
        pos_type = {0: "Flat", 1: "Long", 2: "Short"}.get(pos.get("type", 0), "?")
        logger.debug(
            f"POSITION  {pos.get('contractId')}  {pos_type}  "
            f"size={pos.get('size')}  avgPx={pos.get('averagePrice')}"
        )
        if self._on_position:
            self._on_position(pos)
