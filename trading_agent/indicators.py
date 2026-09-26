from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .projectx_client import Bar


# ---------------------------------------------------------------------------
# EMA helpers (private)
# ---------------------------------------------------------------------------

class _EMA:
    """Standard EMA  α = 2 / (n + 1)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._alpha = 2.0 / (period + 1)
        self.value: float | None = None
        self._n = 0

    def update(self, price: float) -> float | None:
        self._n += 1
        if self._n == 1:
            self.value = price
        else:
            self.value = self._alpha * price + (1.0 - self._alpha) * self.value  # type: ignore[operator]
        return self.value if self._n >= self.period else None

    def reset(self) -> None:
        self.__init__(self.period)


class _WilderEMA:
    """Wilder's smoothed MA  α = 1 / n  (used inside RSI)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.value: float | None = None
        self._n = 0
        self._sum = 0.0

    def update(self, price: float) -> float | None:
        self._n += 1
        if self._n < self.period:
            self._sum += price
            return None
        if self._n == self.period:
            self._sum += price
            self.value = self._sum / self.period
        else:
            self.value = (self.value * (self.period - 1) + price) / self.period  # type: ignore[operator]
        return self.value

    def reset(self) -> None:
        self.__init__(self.period)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class RSI:
    """RSI using Wilder's smoothed moving average."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.value: float | None = None
        self._gain_ema = _WilderEMA(period)
        self._loss_ema = _WilderEMA(period)
        self._prev: float | None = None

    def update(self, close: float) -> float | None:
        if self._prev is None:
            self._prev = close
            return None

        delta = close - self._prev
        self._prev = close

        gain = self._gain_ema.update(max(delta, 0.0))
        loss = self._loss_ema.update(max(-delta, 0.0))

        if gain is None or loss is None:
            return None

        if loss == 0.0:
            self.value = 100.0
        else:
            self.value = 100.0 - 100.0 / (1.0 + gain / loss)
        return self.value

    def reset(self) -> None:
        self.__init__(self.period)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

@dataclass
class MACDResult:
    macd: float
    signal: float
    histogram: float


class MACD:
    """MACD using three standard EMAs: fast, slow, and signal."""

    def __init__(self, fast: int = 5, slow: int = 16, signal: int = 40) -> None:
        self._fast_period   = fast
        self._slow_period   = slow
        self._signal_period = signal
        self._fast   = _EMA(fast)
        self._slow   = _EMA(slow)
        self._signal = _EMA(signal)
        self.result: MACDResult | None = None

    def update(self, close: float) -> MACDResult | None:
        fast_val = self._fast.update(close)
        slow_val = self._slow.update(close)

        if fast_val is None or slow_val is None:
            return None

        macd_val = fast_val - slow_val
        sig_val  = self._signal.update(macd_val)

        if sig_val is None:
            return None

        self.result = MACDResult(
            macd=macd_val,
            signal=sig_val,
            histogram=macd_val - sig_val,
        )
        return self.result

    def reset(self) -> None:
        self.__init__(self._fast_period, self._slow_period, self._signal_period)


# ---------------------------------------------------------------------------
# 5-minute bar accumulator — no lookahead guarantee
# ---------------------------------------------------------------------------

@dataclass
class Bar5m:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_count: int


class FiveMinAccumulator:
    """
    Collects 1-minute bars and emits a confirmed Bar5m only when the first
    bar of the NEXT 5-minute period arrives.  This is the key no-lookahead
    guarantee: the 5m bar's close is always a past, fully-formed candle.
    """

    def __init__(self) -> None:
        self._buf: list[Bar] = []

    def update(self, bar: Bar) -> Bar5m | None:
        bucket = _5m_floor(bar.timestamp)

        if not self._buf:
            self._buf.append(bar)
            return None

        current_bucket = _5m_floor(self._buf[0].timestamp)
        if bucket == current_bucket:
            self._buf.append(bar)
            return None

        confirmed = _build_5m(self._buf, current_bucket)
        self._buf = [bar]
        return confirmed

    def reset(self) -> None:
        self._buf.clear()


class MinuteBarAccumulator:
    """
    Generic N-minute bar accumulator.  Works identically to FiveMinAccumulator
    but for any period (e.g. 15, 30, 60 minutes).

    A completed bar is only emitted when the first bar of the NEXT period
    arrives — no lookahead bias.
    """

    def __init__(self, minutes: int) -> None:
        self._minutes = minutes
        self._buf: list[Bar] = []

    def update(self, bar: Bar) -> Bar5m | None:
        bucket = _nm_floor(bar.timestamp, self._minutes)

        if not self._buf:
            self._buf.append(bar)
            return None

        current_bucket = _nm_floor(self._buf[0].timestamp, self._minutes)
        if bucket == current_bucket:
            self._buf.append(bar)
            return None

        confirmed = _build_5m(self._buf, current_bucket)
        self._buf = [bar]
        return confirmed

    def reset(self) -> None:
        self._buf.clear()


def _5m_floor(ts: datetime) -> datetime:
    """Floor a datetime to its 5-minute boundary."""
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)


def _nm_floor(ts: datetime, minutes: int) -> datetime:
    """Floor a datetime to the nearest N-minute boundary."""
    floored_min = (ts.minute // minutes) * minutes
    return ts.replace(minute=floored_min, second=0, microsecond=0)


def _build_5m(bars: list[Bar], ts: datetime) -> Bar5m:
    return Bar5m(
        timestamp=ts,
        open=bars[0].open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=bars[-1].close,
        volume=sum(b.volume for b in bars),
        bar_count=len(bars),
    )


# ---------------------------------------------------------------------------
# VWAP — session-anchored, call reset() at session start
# ---------------------------------------------------------------------------

class VWAP:
    """Typical-price-weighted VWAP. Resets whenever reset() is called."""

    def __init__(self) -> None:
        self._cum_tp_vol: float = 0.0
        self._cum_vol:    float = 0.0
        self.value: float | None = None

    def update(self, bar: Bar) -> float | None:
        tp = (bar.high + bar.low + bar.close) / 3.0
        self._cum_tp_vol += tp * bar.volume
        self._cum_vol    += bar.volume
        if self._cum_vol > 0:
            self.value = self._cum_tp_vol / self._cum_vol
        return self.value

    def reset(self) -> None:
        self._cum_tp_vol = 0.0
        self._cum_vol    = 0.0
        self.value       = None
