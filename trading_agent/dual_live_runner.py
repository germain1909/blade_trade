from __future__ import annotations

"""
Dual Strategy Live Runner
=========================

Runs Pivot+Supertrend and ORB strategies in parallel with shared risk.

  1. DAILY PREP  (5pm ET)
     - Fetch prior day bars, compute daily pivots + ATR
     - Save plan to disk

  2. SESSION START
     - Load pivot plan
     - Warm up Supertrend engine + ORB indicators with historical bars
     - Connect SignalR

  3. MAIN LOOP (per 1m bar)
     - Update 30m Supertrend engine
     - Check 5m rejection at pivot levels
     - Feed ORB strategy
     - First valid signal across either strategy → bracket order
     - Shared daily risk limits

Usage:
    from trading_agent.dual_live_runner import DualLiveSession

    session = DualLiveSession(config, account_id=123, contract_id="CON.F.US.MES.U26")
    session.daily_prep()
    session.run()
"""

import json
import logging
import queue
import threading
from collections import defaultdict, deque
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .config import Config
from .execution import (
    SIDE_BUY, SIDE_SELL, flatten_position, get_net_position, place_bracket_order,
)
from .indicators import MinuteBarAccumulator, Bar5m
from .live_signalr import LiveConnection
from .notifier import SlackNotifier
from .projectx_client import Bar, ProjectXClient
from .strategy import ORBreakoutStrategy, ORBreakoutConfig, Signal
from .supertrend_strategy import (
    PivotSupertrendConfig, _SupertrendEngine, _auto_select,
    _compute_traditional_pivots, _nearest_levels, _find_rejection,
    _in_window, _clamp,
)

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")
_PLAN_DIR   = Path("data")
_WARMUP_HRS = 12


# ── Config ───────────────────────────────────────────────────────────────────

class DualStrategyConfig:
    def __init__(
        self,
        # Pivot + Supertrend
        pivot_cfg: Optional[PivotSupertrendConfig] = None,
        # ORB
        orb_cfg: Optional[ORBreakoutConfig] = None,
        # Shared risk
        contracts:          int   = 1,
        max_trades_per_day: int   = 6,
        daily_loss_limit:   float = -500.0,
        daily_profit_limit: float = 1500.0,
        trade_start_et:     dtime = dtime(8, 30),
        entry_cutoff_et:    dtime = dtime(15, 0),
        trade_end_et:       dtime = dtime(16, 0),
    ):
        self.pivot_cfg = pivot_cfg or PivotSupertrendConfig(
            st_bar_minutes=30, rej_bar_minutes=5,
            zone_buffer_pts=8.0, min_rr=1.5,
            contracts=contracts,
            trade_start_et=trade_start_et,
            entry_cutoff_et=entry_cutoff_et,
            trade_end_et=trade_end_et,
        )
        self.orb_cfg = orb_cfg or ORBreakoutConfig(
            windows=((trade_start_et.hour, trade_start_et.minute),),
            window_hours=int((trade_end_et.hour - trade_start_et.hour)),
            or_minutes=30,
            stop_ticks=24,
            tick_size=0.25,
            target_rr=2.0,
        )
        self.contracts          = contracts
        self.max_trades_per_day = max_trades_per_day
        self.daily_loss_limit   = daily_loss_limit
        self.daily_profit_limit = daily_profit_limit
        self.trade_start_et     = trade_start_et
        self.entry_cutoff_et    = entry_cutoff_et
        self.trade_end_et       = trade_end_et


# ── Plan persistence ─────────────────────────────────────────────────────────

def _save_dual_plan(pivots: list[tuple[str, float]], daily_atr: float,
                    target_date: date) -> Path:
    _PLAN_DIR.mkdir(parents=True, exist_ok=True)
    path = _PLAN_DIR / f"dual_plan_{target_date}.json"
    data = {
        "date": str(target_date),
        "daily_atr": daily_atr,
        "pivots": [{"name": n, "price": p} for n, p in pivots],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Dual plan saved → {path}")
    return path


def _load_dual_plan(target_date: date) -> tuple[list[tuple[str, float]], float]:
    path = _PLAN_DIR / f"dual_plan_{target_date}.json"
    if not path.exists():
        raise FileNotFoundError(f"No dual plan for {target_date}. Run daily_prep() first.")
    data = json.loads(path.read_text())
    pivots = [(p["name"], p["price"]) for p in data["pivots"]]
    return pivots, float(data["daily_atr"])


# ── Live session ─────────────────────────────────────────────────────────────

class DualLiveSession:
    def __init__(
        self,
        config:      Config,
        account_id:  int,
        contract_id: str,
        strategy_cfg: Optional[DualStrategyConfig] = None,
    ) -> None:
        self._cfg         = config
        self._account_id  = account_id
        self._contract_id = contract_id
        self._scfg        = strategy_cfg or DualStrategyConfig()

        self._client = ProjectXClient(config)

        # ── Pivot + Supertrend ────────────────────────────────────────────────
        pcfg = self._scfg.pivot_cfg
        if pcfg.st_auto_params:
            lb, ct, pw = _auto_select(pcfg.st_bar_minutes)
        else:
            lb, ct, pw = pcfg.st_lookback, pcfg.st_threshold, pcfg.st_pred_weight

        self._st_engine = _SupertrendEngine(lb, ct, pw)
        self._st_accum  = MinuteBarAccumulator(pcfg.st_bar_minutes)
        self._rej_accum = MinuteBarAccumulator(pcfg.rej_bar_minutes)
        self._st_ready  = False
        self._pivots: list[tuple[str, float]] = []
        self._used_levels: set[str] = set()

        # ── ORB ───────────────────────────────────────────────────────────────
        self._orb = ORBreakoutStrategy(self._scfg.orb_cfg)

        # ── Shared trade state ────────────────────────────────────────────────
        self._in_trade    = False
        self._trade_dir:  Optional[str]   = None
        self._trade_stop: Optional[float] = None
        self._trade_tgt:  Optional[float] = None
        self._trade_src:  str             = ""
        self._day_pnl     = 0.0
        self._day_trades  = 0
        self._locked      = False

        # ── Bar queue ─────────────────────────────────────────────────────────
        self._bar_queue: queue.Queue[Optional[Bar]] = queue.Queue()
        self._last_price: float = 0.0

        self._trades: list[dict] = []
        self._live_conn: Optional[LiveConnection] = None
        self._slack = SlackNotifier()

    # ── Public API ────────────────────────────────────────────────────────────

    def daily_prep(self, target_date: Optional[date] = None,
                   history_contract_id: Optional[str] = None) -> None:
        self._client.authenticate()
        today = target_date or datetime.now(_ET).date()
        logger.info(f"dual daily_prep for {today}")

        hist_id  = history_contract_id or self._contract_id
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=5)
        bars = self._client.retrieve_bars(
            contract_id=hist_id, start_time=start_dt, end_time=end_dt,
            unit=ProjectXClient.UNIT_MINUTE, unit_number=1,
        )
        if not bars:
            raise RuntimeError(f"No bars returned for {hist_id}")

        daily_ohlc = self._aggregate_daily(bars)
        sorted_dates = sorted(daily_ohlc.keys())

        if len(sorted_dates) < 2:
            raise RuntimeError("Not enough daily data for pivots")

        prev = daily_ohlc[sorted_dates[-2]]
        pivots = _compute_traditional_pivots(prev["high"], prev["low"], prev["close"])

        logger.info(f"Pivots for {today}:")
        for name, price in pivots:
            logger.info(f"  {name}: {price:.2f}")

        daily_bars = self._build_daily_bars(daily_ohlc)
        daily_atr = self._compute_atr(daily_bars)
        logger.info(f"Daily ATR: {daily_atr:.2f}")

        _save_dual_plan(pivots, daily_atr, today)

    def run(self, target_date: Optional[date] = None) -> None:
        self._client.authenticate()
        today = target_date or datetime.now(_ET).date()

        self._pivots, daily_atr = _load_dual_plan(today)
        logger.info(
            f"Session {today}  ATR={daily_atr}  "
            + "  ".join(f"{n}={p:.2f}" for n, p in self._pivots)
        )

        # ── Warm up ──────────────────────────────────────────────────────────
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=_WARMUP_HRS)
        warmup_bars = self._client.retrieve_bars(
            contract_id=self._contract_id, start_time=start_dt, end_time=end_dt,
            unit=ProjectXClient.UNIT_MINUTE, unit_number=1,
        )
        for bar in warmup_bars:
            self._process_bar(bar, live=False)

        logger.info(f"Warmup complete on {len(warmup_bars)} bars  "
                     f"ST_dir={'BULL' if self._st_engine.t_dir == 1 else 'BEAR'}  "
                     f"ST_ready={self._st_ready}")

        # ── Notify ───────────────────────────────────────────────────────────
        pivot_lines = [f"{n} @ {p:.2f}" for n, p in self._pivots]
        self._slack.send_message(
            f"*Dual session started*\n"
            f"Contract: {self._contract_id}\n"
            f"Strategies: Pivot+Supertrend, ORB\n"
            f"Pivots:\n" + "\n".join(f"  • {l}" for l in pivot_lines)
        )

        # ── SignalR ──────────────────────────────────────────────────────────
        def _start_signalr() -> threading.Thread:
            jwt = self._client._token or ""
            self._live_conn = LiveConnection(
                config=self._cfg, account_id=self._account_id,
                contract_id=self._contract_id, jwt_token=jwt,
                on_bar=self._on_bar, on_quote=self._on_quote,
                on_order=self._on_order, on_fill=self._on_fill,
                on_position=lambda p: None,
            )
            t = threading.Thread(target=self._live_conn.start, daemon=True)
            t.start()
            return t

        sr_thread = _start_signalr()
        logger.info("SignalR started — waiting for bars")

        # ── Main loop ────────────────────────────────────────────────────────
        session_started = False
        try:
            while True:
                try:
                    bar = self._bar_queue.get(timeout=60)
                except queue.Empty:
                    now_et = datetime.now(_ET).time()
                    in_session = _in_window(now_et, self._scfg.trade_start_et,
                                           self._scfg.trade_end_et)
                    if in_session:
                        session_started = True
                    elif session_started:
                        logger.info("Session window closed")
                        break
                    # Before session starts — just keep waiting
                    self._client.ensure_authenticated()
                    if not sr_thread.is_alive():
                        logger.warning("SignalR died — reconnecting")
                        sr_thread = _start_signalr()
                    continue

                if bar is None:
                    break
                # Track session start from bars too
                bar_et = bar.timestamp.astimezone(_ET).time()
                if _in_window(bar_et, self._scfg.trade_start_et,
                              self._scfg.trade_end_et):
                    session_started = True
                self._process_bar(bar, live=True)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
        finally:
            self._shutdown()

    # ── SignalR callbacks ────────────────────────────────────────────────────

    def _on_bar(self, bar: Bar) -> None:
        self._bar_queue.put(bar)

    def _on_quote(self, bid: float, ask: float, last: float) -> None:
        if last:
            self._last_price = last

    def _on_order(self, order: dict) -> None:
        if order.get("status") == 5 and self._in_trade:
            logger.error(f"Order REJECTED: {order}")
            self._emergency_flatten()

    def _on_fill(self, fill: dict) -> None:
        pnl  = float(fill.get("profitAndLoss") or 0)
        fees = float(fill.get("fees") or 0)

        if pnl == 0:
            logger.info(f"FILL (entry) price={fill.get('price')} fees=${fees:.2f}")
            return

        net = pnl - fees
        self._day_pnl    += net
        self._day_trades += 1
        self._trades.append({
            "price": fill.get("price"), "pnl": net,
            "strategy": self._trade_src, "timestamp": fill.get("creationTimestamp"),
        })
        logger.info(f"FILL (close) net=${net:+,.2f}  day_pnl=${self._day_pnl:,.2f}  "
                     f"strategy={self._trade_src}")

        self._slack.send_trade_closed(
            symbol=self._contract_id.split(".")[-1], pnl=net,
            result=f"{'WIN' if net > 0 else 'LOSS'} ({self._trade_src})",
        )

        self._in_trade  = False
        self._trade_dir = None
        self._trade_src = ""

        if (self._day_pnl <= self._scfg.daily_loss_limit
                or self._day_pnl >= self._scfg.daily_profit_limit
                or self._day_trades >= self._scfg.max_trades_per_day):
            logger.info("Daily limit reached")
            self._locked = True
            self._send_summary()

    # ── Core bar processing ──────────────────────────────────────────────────

    def _process_bar(self, bar: Bar, live: bool = True) -> None:
        ts_et    = bar.timestamp.astimezone(_ET)
        bar_time = ts_et.time()
        in_session = _in_window(bar_time, self._scfg.trade_start_et,
                                self._scfg.trade_end_et)
        can_enter  = _in_window(bar_time, self._scfg.trade_start_et,
                                self._scfg.entry_cutoff_et)
        pcfg = self._scfg.pivot_cfg

        # ── 1. Update Supertrend on 30m bars ─────────────────────────────────
        completed_st = self._st_accum.update(bar)
        if completed_st is not None:
            self._st_engine.update(completed_st)
            if self._st_engine.flip is not None:
                self._st_ready = True
                if live and self._in_trade and self._trade_src == "PIVOT":
                    if ((self._trade_dir == "long" and self._st_engine.flip == -1)
                            or (self._trade_dir == "short" and self._st_engine.flip == 1)):
                        logger.info("Supertrend flipped against pivot trade — flattening")
                        self._flatten_and_record("st_flip")

        # ── 2. Check pivot rejection on 5m bars ─────────────────────────────
        completed_rej = self._rej_accum.update(bar)
        if (live and completed_rej is not None and in_session and can_enter
                and not self._in_trade and not self._locked
                and self._st_ready and self._pivots):
            rej = _find_rejection(
                completed_rej, self._pivots, self._st_engine.t_dir,
                pcfg.zone_buffer_pts, self._used_levels)

            if rej is not None:
                name, level_price = rej
                direction = "long" if self._st_engine.t_dir == 1 else "short"

                if direction == "long" and not pcfg.allow_longs:
                    pass
                elif direction == "short" and not pcfg.allow_shorts:
                    pass
                else:
                    slip = pcfg.slippage_ticks * pcfg.tick_size
                    entry_px = (completed_rej.close + slip if direction == "long"
                                else completed_rej.close - slip)
                    pivot_prices = [p for _, p in self._pivots]
                    tp, sl = _nearest_levels(pivot_prices, entry_px, direction)

                    if tp is not None and sl is not None:
                        if direction == "short" and pcfg.short_tp_pct < 1.0:
                            tp = entry_px - abs(entry_px - tp) * pcfg.short_tp_pct

                        risk   = abs(entry_px - sl)
                        reward = abs(tp - entry_px)
                        if risk > 0 and reward / risk >= pcfg.min_rr:
                            self._enter_trade(direction, entry_px, sl, tp,
                                              "PIVOT", f"Pivot {name}")
                            self._used_levels.add(name)

        # ── 3. Feed ORB strategy ─────────────────────────────────────────────
        if live:
            orb_signal = self._orb.on_bar(bar)
            if (orb_signal in (Signal.LONG, Signal.SHORT) and in_session
                    and can_enter and not self._in_trade and not self._locked):
                direction = "long" if orb_signal is Signal.LONG else "short"
                orb_stop   = self._orb.last_stop_price()
                orb_target = self._orb.last_target_price()
                if orb_stop is not None and orb_target is not None:
                    meta = self._orb.last_trade_metadata()
                    label = meta.get("bias_source", "ORB")
                    self._enter_trade(direction, bar.close, orb_stop, orb_target,
                                      "ORB", label)

        # ── Day rollover ─────────────────────────────────────────────────────
        bar_date = ts_et.date()
        if not hasattr(self, '_cur_date') or bar_date != self._cur_date:
            self._cur_date    = bar_date
            self._used_levels = set()
            if live:
                self._day_trades = 0
                self._day_pnl    = 0.0
                self._locked     = False
                self._orb.reset()

    def _enter_trade(self, direction: str, entry: float, stop: float,
                     target: float, strategy: str, label: str) -> None:
        tick = self._scfg.pivot_cfg.tick_size
        sl_ticks = max(1, round(abs(entry - stop) / tick))
        tp_ticks = max(1, round(abs(target - entry) / tick))
        side = SIDE_BUY if direction == "long" else SIDE_SELL
        tag  = f"{strategy}_{direction[0].upper()}"

        logger.info(
            f"ENTRY [{strategy}] {direction.upper()} {label}  "
            f"ep={entry:.2f}  sl={stop:.2f}({sl_ticks}t)  tp={target:.2f}({tp_ticks}t)"
        )

        try:
            place_bracket_order(
                client=self._client, account_id=self._account_id,
                contract_id=self._contract_id, side=side,
                size=self._scfg.contracts, stop_loss_ticks=sl_ticks,
                take_profit_ticks=tp_ticks, custom_tag=tag,
            )
        except Exception as exc:
            logger.error(f"place_bracket_order FAILED: {exc}")
            try:
                net = get_net_position(self._client, self._account_id,
                                       self._contract_id)
                if net != 0:
                    flatten_position(self._client, self._account_id,
                                     self._contract_id)
            except Exception as fe:
                logger.critical(f"Emergency flatten also failed: {fe}")
            return

        self._in_trade   = True
        self._trade_dir  = direction
        self._trade_stop = stop
        self._trade_tgt  = target
        self._trade_src  = strategy

        self._slack.send_trade_alert(
            symbol=self._contract_id.split(".")[-1],
            direction=direction.upper(), entry=entry,
            stop=stop, target=target, poi_label=f"[{strategy}] {label}",
        )

    def _flatten_and_record(self, reason: str) -> None:
        logger.info(f"Flattening position — {reason}")
        try:
            flatten_position(self._client, self._account_id, self._contract_id)
        except Exception as exc:
            logger.error(f"Flatten failed: {exc}")
        self._in_trade  = False
        self._trade_dir = None
        self._trade_src = ""

    def _emergency_flatten(self) -> None:
        try:
            flatten_position(self._client, self._account_id, self._contract_id)
        except Exception as exc:
            logger.critical(f"Emergency flatten failed: {exc}")
        self._in_trade  = False
        self._trade_dir = None
        self._trade_src = ""

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _aggregate_daily(self, bars: list[Bar]) -> dict[date, dict]:
        daily: dict[date, dict] = defaultdict(
            lambda: {"open": 0, "high": float("-inf"), "low": float("inf"), "close": 0})
        for b in bars:
            ts_et = b.timestamp.astimezone(_ET)
            d = ts_et.date()
            if ts_et.hour >= 17:
                d = d + timedelta(days=1)
            s = daily[d]
            if s["open"] == 0:
                s["open"] = b.open
            s["high"]  = max(s["high"],  b.high)
            s["low"]   = min(s["low"],   b.low)
            s["close"] = b.close
        return dict(daily)

    def _build_daily_bars(self, daily: dict[date, dict]) -> list[Bar]:
        result = []
        for d in sorted(daily.keys()):
            s = daily[d]
            if s["high"] == float("-inf"):
                continue
            result.append(Bar(
                timestamp=datetime(d.year, d.month, d.day, tzinfo=_ET),
                open=s["open"], high=s["high"], low=s["low"], close=s["close"],
                volume=0,
            ))
        return result

    def _compute_atr(self, daily_bars: list[Bar], period: int = 14) -> float:
        if len(daily_bars) < 2:
            return 0.0
        trs = []
        for i in range(1, len(daily_bars)):
            h, l, pc = daily_bars[i].high, daily_bars[i].low, daily_bars[i-1].close
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(trs[:period]) / period if len(trs) >= period else sum(trs) / len(trs)
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return round(atr, 2)

    def _send_summary(self) -> None:
        wins   = sum(1 for t in self._trades if t["pnl"] > 0)
        losses = len(self._trades) - wins
        wr     = wins / len(self._trades) if self._trades else 0.0
        self._slack.send_daily_summary(
            total_pnl=self._day_pnl, win_rate=wr,
            trades=self._day_trades, wins=wins, losses=losses,
        )

    def _shutdown(self) -> None:
        logger.info("Shutting down")
        try:
            if self._live_conn:
                self._live_conn.stop()
        except Exception:
            pass

        _PLAN_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _PLAN_DIR / f"dual_trades_{datetime.now(_ET).date()}.json"
        log_path.write_text(json.dumps({
            "trades": self._trades, "day_pnl": self._day_pnl,
            "day_trades": self._day_trades,
        }, indent=2))
        logger.info(f"Trade log saved → {log_path}")
        logger.info(f"Session complete  trades={self._day_trades}  "
                     f"day_pnl=${self._day_pnl:,.2f}")

        if not self._locked:
            self._send_summary()
