from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .projectx_client import Bar
from .strategy import Signal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    Dollar-denominated exit levels for a fixed-size position.

    Defaults match the initial backtest spec:
        5 MES contracts, $500 take-profit, $200 stop-loss.

    MES / ES: dollars_per_point = 5.0  ($5 per index point)
    MNQ / NQ: dollars_per_point = 2.0  ($2 per index point)
    """
    contracts:         int   = 5
    take_profit_usd:   float = 500.0
    stop_loss_usd:     float = 200.0
    dollars_per_point: float = 5.0

    @property
    def tp_points(self) -> float:
        """Take-profit distance in index points."""
        return self.take_profit_usd / (self.contracts * self.dollars_per_point)

    @property
    def sl_points(self) -> float:
        """Stop-loss distance in index points."""
        return self.stop_loss_usd / (self.contracts * self.dollars_per_point)


# ---------------------------------------------------------------------------
# Exit event
# ---------------------------------------------------------------------------

class ExitReason(Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS   = "stop_loss"


@dataclass
class Exit:
    reason:      ExitReason
    direction:   Signal
    entry_price: float
    exit_price:  float
    entry_time:  datetime
    exit_time:   datetime
    pnl_usd:     float
    contracts:   int


# ---------------------------------------------------------------------------
# Internal position state
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    direction:   Signal
    entry_price: float
    entry_time:  datetime
    tp_price:    float
    sl_price:    float
    contracts:   int


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Tracks one open position at a time and exits via fixed TP/SL levels.

    Quick-start:
        risk = RiskManager(RiskConfig(contracts=5, take_profit_usd=500, stop_loss_usd=200))

        risk.open_position(Signal.LONG, entry_price=5250.00, timestamp=bar.timestamp)

        for bar in subsequent_bars:
            exit = risk.on_bar(bar)
            if exit:
                print(exit.reason, exit.pnl_usd)

    TP/SL levels are computed from the entry price each time a position is opened,
    so you can change config.take_profit_usd / config.stop_loss_usd at any time
    before the next entry takes effect.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._cfg = config or RiskConfig()
        self._pos: _Position | None = None

    # ------------------------------------------------------------------
    # Config — readable and writable so callers can adjust TP/SL
    # ------------------------------------------------------------------

    @property
    def config(self) -> RiskConfig:
        return self._cfg

    @config.setter
    def config(self, value: RiskConfig) -> None:
        self._cfg = value

    # ------------------------------------------------------------------
    # Position state
    # ------------------------------------------------------------------

    @property
    def in_position(self) -> bool:
        return self._pos is not None

    @property
    def position_direction(self) -> Signal | None:
        return self._pos.direction if self._pos else None

    @property
    def position(self) -> _Position | None:
        return self._pos

    # ------------------------------------------------------------------
    # Open / check / reset
    # ------------------------------------------------------------------

    def open_position(
        self,
        direction: Signal,
        entry_price: float,
        timestamp: datetime,
        stop_price: float | None = None,    # override: use directly instead of RiskConfig
        target_price: float | None = None,  # override: use directly instead of RiskConfig
    ) -> None:
        """Open a new position. Raises if one is already open."""
        if self._pos is not None:
            raise RuntimeError("Already in a position — exit it before opening a new one.")
        if direction not in (Signal.LONG, Signal.SHORT):
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

        tp_pts = self._cfg.tp_points
        sl_pts = self._cfg.sl_points

        if stop_price is not None:
            sl_price = stop_price
        elif direction is Signal.LONG:
            sl_price = entry_price - sl_pts
        else:
            sl_price = entry_price + sl_pts

        if target_price is not None:
            tp_price = target_price
        elif direction is Signal.LONG:
            tp_price = entry_price + tp_pts
        else:
            tp_price = entry_price - tp_pts

        self._pos = _Position(
            direction=direction,
            entry_price=entry_price,
            entry_time=timestamp,
            tp_price=tp_price,
            sl_price=sl_price,
            contracts=self._cfg.contracts,
        )

    def on_bar(self, bar: Bar) -> Exit | None:
        """
        Check whether this bar touches a TP or SL level.

        Uses the bar's high and low for fill detection — the position may have
        been entered on a prior bar's close.

        When both levels are touched in the same bar, stop-loss wins
        (conservative: assume the adverse tick hit first).
        """
        if self._pos is None:
            return None

        pos = self._pos

        if pos.direction is Signal.LONG:
            sl_hit = bar.low  <= pos.sl_price
            tp_hit = bar.high >= pos.tp_price
        else:
            sl_hit = bar.high >= pos.sl_price
            tp_hit = bar.low  <= pos.tp_price

        if sl_hit:
            return self._close(pos, pos.sl_price, bar.timestamp, ExitReason.STOP_LOSS)
        if tp_hit:
            return self._close(pos, pos.tp_price, bar.timestamp, ExitReason.TAKE_PROFIT)

        return None

    def reset(self) -> None:
        """Discard any open position — call before a new backtest run."""
        self._pos = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _close(
        self,
        pos:        _Position,
        exit_price: float,
        exit_time:  datetime,
        reason:     ExitReason,
    ) -> Exit:
        if pos.direction is Signal.LONG:
            pnl = (exit_price - pos.entry_price) * pos.contracts * self._cfg.dollars_per_point
        else:
            pnl = (pos.entry_price - exit_price) * pos.contracts * self._cfg.dollars_per_point

        self._pos = None
        return Exit(
            reason=reason,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl_usd=round(pnl, 2),
            contracts=pos.contracts,
        )
