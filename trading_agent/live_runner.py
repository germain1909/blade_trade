from __future__ import annotations

"""
POI Live Session Runner
=======================

Full pipeline for one trading session:

  1. DAILY PREP  (run at 5 pm ET, once per day)
     - Fetch yesterday's 1m bars from the ProjectX REST API
     - Compute the 14-period Wilder ATR for today's session
     - Run the POI scanner to find 3 confluence levels
     - Save the session plan to disk (data/poi_plan_YYYY-MM-DD.json)

  2. SESSION START (run when the market opens, or at a scheduled time)
     - Load the saved POI plan
     - Fetch today's bars so far via REST to warm up the indicators
     - Spin up LiveConnection (SignalR) in a background thread
     - Enter the main processing loop

  3. MAIN LOOP (per completed 1-minute bar from SignalR tick stream)
     - Update 5m MACD and 1m EMA indicators
     - Feed into the POI strategy state machine
     - If an entry is confirmed → place a bracket order via the REST API
     - Log every transition and trade

  4. SESSION END
     - Disconnect SignalR
     - Save the trade log to disk (data/poi_trades_YYYY-MM-DD.json)

Usage
-----
    from trading_agent.live_runner import POILiveSession

    session = POILiveSession(config, account_id=123, contract_id="CON.F.US.MES.M25")
    session.daily_prep()          # run at 5pm ET
    session.run()                 # run at session start
"""

import json
import logging
import queue
import threading
from collections import deque, defaultdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .config import Config
from .execution import SIDE_BUY, SIDE_SELL, flatten_position, get_net_position, place_bracket_order
from .indicators import MinuteBarAccumulator, _EMA
from .live_signalr import LiveConnection
from .notifier import SlackNotifier
from .poi_scanner import POI, LevelRef, scan_poi, print_poi_report
from .poi_strategy import (
    POIStrategyConfig, POIZoneState, _POIState,
    _at_zone, _macd_aligned, _ema_aligned,
    _entry_direction, _clean_close, _next_poi_target, _in_window,
)
from .projectx_client import Bar, ProjectXClient

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

_PLAN_DIR   = Path("data")
_WARMUP_HRS = 6


# ─────────────────────────────────────────────────────────────────────────────
# ATR helper (Wilder's method on daily ranges)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_daily_atr(daily_bars: list[Bar], period: int = 14) -> float:
    """Wilder's ATR from a list of daily (or session-aggregated) bars."""
    if len(daily_bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(daily_bars)):
        h  = daily_bars[i].high
        l  = daily_bars[i].low
        pc = daily_bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[:period]) / period if len(trs) >= period else sum(trs) / len(trs)
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 2)


# ─────────────────────────────────────────────────────────────────────────────
# POI plan persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_plan(pois: list[POI], daily_atr: float, day_open: float,
               target_date: date) -> Path:
    _PLAN_DIR.mkdir(parents=True, exist_ok=True)
    path = _PLAN_DIR / f"poi_plan_{target_date}.json"
    data = {
        "date":      str(target_date),
        "daily_atr": daily_atr,
        "day_open":  day_open,
        "pois": [
            {
                "rank":             p.rank,
                "price":            p.price,
                "confluence_score": p.confluence_score,
                "bias":             p.bias,
                "above_price":      p.above_price,
                "nearest_atr":      p.nearest_atr,
                "levels":           [{"name": l.name, "price": l.price}
                                     for l in p.levels],
            }
            for p in pois
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"POI plan saved → {path}")
    return path


def _load_plan(target_date: date) -> tuple[list[dict], float, float]:
    path = _PLAN_DIR / f"poi_plan_{target_date}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No POI plan for {target_date}. Run daily_prep() first (expected: {path})"
        )
    data = json.loads(path.read_text())
    return data["pois"], float(data["daily_atr"]), float(data["day_open"])


def _pois_from_json(raw: list[dict]) -> list[POI]:
    pois = []
    for p in raw:
        pois.append(POI(
            rank             = p["rank"],
            price            = p["price"],
            confluence_score = p["confluence_score"],
            bias             = p["bias"],
            above_price      = p["above_price"],
            nearest_atr      = p.get("nearest_atr"),
            distance_pts     = 0.0,
            levels           = [LevelRef(l["name"], l["price"]) for l in p["levels"]],
        ))
    return pois


# ─────────────────────────────────────────────────────────────────────────────
# Live session
# ─────────────────────────────────────────────────────────────────────────────

class POILiveSession:
    """
    Orchestrates the full POI live trading pipeline.

    Parameters
    ----------
    config       : loaded Config (credentials + API URLs)
    account_id   : integer account ID from TopstepX
    contract_id  : e.g. "CON.F.US.MES.M25"
    strategy_cfg : POIStrategyConfig (optional — uses sensible defaults)
    """

    def __init__(
        self,
        config:       Config,
        account_id:   int,
        contract_id:  str,
        strategy_cfg: Optional[POIStrategyConfig] = None,
    ) -> None:
        self._cfg         = config
        self._account_id  = account_id
        self._contract_id = contract_id
        self._scfg        = strategy_cfg or POIStrategyConfig(
            zone_buffer_ticks        = 6,
            max_zone_bars            = 20,
            require_macd_slope_align = True,
            require_ema_slope_align  = True,
            entry_clean_close_ticks  = 8,
            stop_atr_pct             = 0.25,
            use_atr_stop             = True,
            target_r_multiple        = 2.0,
            contracts                = 1,
            slippage_ticks           = 1,
            max_trades_per_day       = 3,
            daily_loss_limit_usd     = -500.0,
            daily_profit_limit_usd   = 1500.0,
            trade_start_et           = dtime(18, 0),
            entry_cutoff_et          = dtime(16, 30),
            trade_end_et             = dtime(17, 0),
        )

        self._client = ProjectXClient(config)

        # ── Indicators (rebuilt each session) ────────────────────────────────
        self._ema1m      = _EMA(self._scfg.ema_length)
        self._ema_hist:  deque = deque(maxlen=self._scfg.ema_slope_lookback_bars + 1)
        self._acc5m      = MinuteBarAccumulator(5)
        self._macd_fast  = _EMA(self._scfg.macd_fast_length)
        self._macd_slow  = _EMA(self._scfg.macd_slow_length)
        self._macd_sig   = _EMA(self._scfg.macd_signal_length)
        self._macd_hist: deque = deque(maxlen=self._scfg.macd_slope_lookback_bars + 1)

        self._cur_ema:    Optional[float] = None
        self._cur_ema_sl: Optional[float] = None
        self._cur_macd_ln: Optional[float] = None
        self._cur_macd_sl: Optional[float] = None

        # ── POI state ─────────────────────────────────────────────────────────
        self._pois:       list[POI]       = []
        self._poi_states: list[_POIState] = []
        self._daily_atr:  float           = 0.0

        # ── Trade state ───────────────────────────────────────────────────────
        self._in_trade    = False
        self._trade_dir:  Optional[str]    = None
        self._trade_stop: Optional[float]  = None
        self._trade_tgt:  Optional[float]  = None
        self._active_poi: Optional[_POIState] = None
        self._day_pnl     = 0.0
        self._day_trades  = 0
        self._locked      = False

        # ── Bar queue (SignalR thread → main thread) ──────────────────────────
        self._bar_queue: queue.Queue[Optional[Bar]] = queue.Queue()
        self._last_price: float = 0.0

        self._logs:   list[dict] = []
        self._trades: list[dict] = []
        self._live_conn: Optional[LiveConnection] = None
        self._slack = SlackNotifier()

    # ── Public API ────────────────────────────────────────────────────────────

    def daily_prep(self, target_date: Optional[date] = None,
                   history_contract_id: Optional[str] = None) -> list[POI]:
        """
        Run at 5pm ET each day.
        Fetches previous-day bars, builds POIs, saves plan to disk.

        history_contract_id : use a different contract for historical bars
                              (e.g. the old front month after a roll).
                              Falls back to self._contract_id if not set.
        """
        self._client.authenticate()
        today = target_date or datetime.now(_ET).date()
        logger.info(f"daily_prep for {today}")

        hist_id  = history_contract_id or self._contract_id
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=5)   # extra days to survive rolls
        bars = self._client.retrieve_bars(
            contract_id  = hist_id,
            start_time   = start_dt,
            end_time     = end_dt,
            unit         = ProjectXClient.UNIT_MINUTE,
            unit_number  = 1,
        )
        if not bars:
            raise RuntimeError(
                f"No bars returned for {hist_id}. "
                f"After a roll, pass history_contract_id=<old contract>."
            )

        daily_atr = self._estimate_daily_atr(bars)

        # Bars are returned newest-first by the API — bars[0] is the most recent.
        current_price = bars[0].close
        logger.info(
            f"current_price={current_price}  "
            f"from bar at {bars[0].timestamp.astimezone(_ET):%H:%M ET}"
        )
        scan_time = datetime.now(_ET)

        pois, levels = scan_poi(
            bars          = bars,
            scan_time     = scan_time,
            current_price = current_price,
            daily_atr     = daily_atr,
        )

        print_poi_report(pois, levels)
        _save_plan(pois, daily_atr, levels.day_open, today)
        return pois

    def run(self, target_date: Optional[date] = None) -> None:
        """
        Start the live trading session.
        Loads the POI plan, warms up indicators, connects SignalR,
        then processes bars until the session ends.
        """
        self._client.authenticate()
        today = target_date or datetime.now(_ET).date()

        # ── Load POI plan ─────────────────────────────────────────────────────
        raw_pois, self._daily_atr, day_open = _load_plan(today)
        self._pois       = _pois_from_json(raw_pois)
        self._poi_states = [_POIState(poi=p) for p in self._pois]

        logger.info(
            f"Session {today}  ATR={self._daily_atr}  open={day_open}  "
            + "  ".join(f"POI#{p.rank}@{p.price}" for p in self._pois)
        )

        # ── Warm up indicators ────────────────────────────────────────────────
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=_WARMUP_HRS)
        warmup_bars = self._client.retrieve_bars(
            contract_id = self._contract_id,
            start_time  = start_dt,
            end_time    = end_dt,
            unit        = ProjectXClient.UNIT_MINUTE,
            unit_number = 1,
        )
        for bar in warmup_bars:
            self._process_bar(bar, live=False)

        ema_str  = f"{self._cur_ema:.2f}"   if self._cur_ema   is not None else "n/a"
        macd_str = f"{self._cur_macd_sl:.4f}" if self._cur_macd_sl is not None else "n/a"
        logger.info(
            f"Warmup complete on {len(warmup_bars)} bars  "
            f"EMA={ema_str}  MACD_slope={macd_str}"
        )

        # ── Notify session start ──────────────────────────────────────────────
        self._slack.send_session_started(
            contract = self._contract_id,
            pois     = [f"POI#{p.rank} @ {p.price}  [{p.label}]" for p in self._pois],
        )

        # ── Connect SignalR ───────────────────────────────────────────────────
        def _start_signalr_thread() -> threading.Thread:
            jwt = self._client._token or ""
            self._live_conn = LiveConnection(
                config      = self._cfg,
                account_id  = self._account_id,
                contract_id = self._contract_id,
                jwt_token   = jwt,
                on_bar      = self._on_bar_received,
                on_quote    = self._on_quote_received,
                on_order    = self._on_order_update,
                on_fill     = self._on_fill,
                on_position = self._on_position,
            )
            t = threading.Thread(target=self._live_conn.start, daemon=True, name="signalr")
            t.start()
            return t

        signalr_thread = _start_signalr_thread()
        logger.info("SignalR thread started — waiting for bars …")

        # ── Main processing loop ──────────────────────────────────────────────
        try:
            while True:
                try:
                    bar = self._bar_queue.get(timeout=60)
                except queue.Empty:
                    now_et = datetime.now(_ET).time()
                    if now_et >= self._scfg.trade_end_et:
                        logger.info("Session window closed — ending session")
                        break
                    self._client.ensure_authenticated()
                    if not signalr_thread.is_alive():
                        logger.warning("SignalR thread died — reconnecting with fresh token …")
                        signalr_thread = _start_signalr_thread()
                    continue

                if bar is None:
                    break
                self._process_bar(bar, live=True)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        finally:
            self._shutdown()

    # ── SignalR callbacks ─────────────────────────────────────────────────────

    def _on_bar_received(self, bar: Bar) -> None:
        self._bar_queue.put(bar)

    def _on_quote_received(self, bid: float, ask: float, last: float) -> None:
        if last:
            self._last_price = last

    def _on_order_update(self, order: dict) -> None:
        if order.get("status") == 5 and self._in_trade:
            logger.error(f"Entry order REJECTED: {order} — flattening as safety")
            self._emergency_flatten()

    def _on_fill(self, fill: dict) -> None:
        pnl  = float(fill.get("profitAndLoss") or 0)
        fees = float(fill.get("fees") or 0)

        # Only process CLOSING fills (entry fills have profitAndLoss=0/null).
        # Fees alone on an entry fill must not reset trade state.
        if pnl == 0:
            logger.info(
                f"FILL (entry)  price={fill.get('price')}  "
                f"fees=${fees:.2f}  orderId={fill.get('orderId')}"
            )
            return

        net = pnl - fees
        self._day_pnl    += net
        self._day_trades += 1
        self._trades.append({
            "fill_id":   fill.get("id"),
            "price":     fill.get("price"),
            "side":      fill.get("side"),
            "size":      fill.get("size"),
            "pnl":       net,
            "order_id":  fill.get("orderId"),
            "timestamp": fill.get("creationTimestamp"),
        })
        logger.info(f"FILL (close)  net=${net:+,.2f}  day_pnl=${self._day_pnl:,.2f}  "
                    f"day_trades={self._day_trades}")

        self._slack.send_trade_closed(
            symbol = self._contract_id.split(".")[-1],
            pnl    = net,
        )

        if self._active_poi:
            self._active_poi.state = POIZoneState.USED
        self._in_trade   = False
        self._trade_dir  = None
        self._active_poi = None

        if (self._day_pnl  <= self._scfg.daily_loss_limit_usd
                or self._day_pnl >= self._scfg.daily_profit_limit_usd
                or self._day_trades >= self._scfg.max_trades_per_day):
            logger.info("Daily limit reached — no more entries today")
            self._locked = True
            wins   = sum(1 for t in self._trades if t["pnl"] > 0)
            losses = len(self._trades) - wins
            wr     = wins / len(self._trades) if self._trades else 0.0
            self._slack.send_daily_summary(
                total_pnl = self._day_pnl,
                win_rate  = wr,
                trades    = self._day_trades,
                wins      = wins,
                losses    = losses,
            )
            self._slack.send_daily_limit_hit(
                reason  = "max trades" if self._day_trades >= self._scfg.max_trades_per_day
                          else ("profit target" if self._day_pnl >= self._scfg.daily_profit_limit_usd
                                else "loss limit"),
                day_pnl = self._day_pnl,
            )

    def _on_position(self, pos: dict) -> None:
        pass   # bracket orders manage TP/SL on exchange side

    # ── Core bar processing ───────────────────────────────────────────────────

    def _process_bar(self, bar: Bar, live: bool = True) -> None:
        """Feed one completed 1m bar through indicators and state machine."""
        ts_et    = bar.timestamp.astimezone(_ET)
        bar_time = ts_et.time()

        # ── 5M MACD ───────────────────────────────────────────────────────────
        completed_5m = self._acc5m.update(bar)
        if completed_5m is not None:
            fv = self._macd_fast.update(completed_5m.close)
            sv = self._macd_slow.update(completed_5m.close)
            if fv is not None and sv is not None:
                ml  = fv - sv
                sgv = self._macd_sig.update(ml)
                if sgv is not None:
                    self._cur_macd_ln = ml
                    self._macd_hist.append(ml)
                    if len(self._macd_hist) >= self._scfg.macd_slope_lookback_bars + 1:
                        self._cur_macd_sl = self._macd_hist[-1] - self._macd_hist[0]

            if live and _in_window(bar_time, self._scfg.trade_start_et,
                                   self._scfg.trade_end_et):
                self._update_poi_zone_states(completed_5m, bar.timestamp)
                self._log_heartbeat(bar.timestamp)

        # ── 1M EMA ────────────────────────────────────────────────────────────
        ev = self._ema1m.update(bar.close)
        if ev is not None:
            self._cur_ema = ev
            self._ema_hist.append(ev)
            if len(self._ema_hist) >= self._scfg.ema_slope_lookback_bars + 1:
                self._cur_ema_sl = self._ema_hist[-1] - self._ema_hist[0]

        if not live:
            return

        in_session = _in_window(bar_time, self._scfg.trade_start_et,
                                self._scfg.trade_end_et)
        can_enter  = _in_window(bar_time, self._scfg.trade_start_et,
                                self._scfg.entry_cutoff_et)

        if (not in_session or self._in_trade or self._locked
                or self._cur_ema is None or self._cur_macd_sl is None
                or not can_enter):
            return

        self._check_entry(bar)

    def _update_poi_zone_states(self, bar5m, bar_ts: datetime) -> None:
        """Advance each POI's zone state on every completed 5m bar."""
        cfg  = self._scfg
        zbuf = cfg.zone_buffer_ticks * cfg.tick_size

        for ps in self._poi_states:
            if ps.state in (POIZoneState.USED, POIZoneState.EXPIRED,
                            POIZoneState.ENTRY_CONFIRMED):
                continue

            at_z = _at_zone(bar5m, ps.poi.price, zbuf)

            if ps.state == POIZoneState.WATCHING:
                if at_z:
                    ps.zone_bars    = 1
                    ps.touched_time = bar_ts
                    self._set_poi_state(ps, POIZoneState.ZONE_TOUCHED, bar_ts,
                                        "zone_touched", bar5m.close)

            elif ps.state == POIZoneState.ZONE_TOUCHED:
                if not at_z:
                    self._set_poi_state(ps, POIZoneState.WATCHING, bar_ts,
                                        "left_zone", bar5m.close)
                    continue
                ps.zone_bars += 1
                macd_ok = _macd_aligned(ps.poi.bias, self._cur_macd_sl,
                                        cfg.macd_slope_threshold)
                ema_ok  = _ema_aligned(ps.poi.bias, self._cur_ema_sl,
                                       cfg.ema_slope_threshold)
                if macd_ok or ema_ok:
                    self._set_poi_state(ps, POIZoneState.SETUP_FORMING, bar_ts,
                                        "indicators_aligning", bar5m.close)
                if ps.zone_bars > cfg.max_zone_bars:
                    self._set_poi_state(ps, POIZoneState.EXPIRED, bar_ts,
                                        "zone_timeout", bar5m.close)

            elif ps.state == POIZoneState.SETUP_FORMING:
                if not at_z:
                    self._set_poi_state(ps, POIZoneState.WATCHING, bar_ts,
                                        "left_zone_before_entry", bar5m.close)
                    continue
                ps.zone_bars += 1
                if ps.zone_bars > cfg.max_zone_bars:
                    self._set_poi_state(ps, POIZoneState.EXPIRED, bar_ts,
                                        "zone_timeout", bar5m.close)

    def _log_heartbeat(self, bar_ts: datetime) -> None:
        """Append current POI states to a separate heartbeat log every 5m bar."""
        _PLAN_DIR.mkdir(parents=True, exist_ok=True)
        path = _PLAN_DIR / f"poi_heartbeat_{datetime.now(_ET).date()}.jsonl"
        entry = {
            "timestamp":  bar_ts.astimezone(_ET).isoformat(),
            "price":      self._last_price,
            "macd_slope": self._cur_macd_sl,
            "ema_slope":  self._cur_ema_sl,
            "ema":        self._cur_ema,
            "pois": [
                {
                    "poi_rank":  ps.poi.rank,
                    "poi_price": ps.poi.price,
                    "poi_bias":  ps.poi.bias,
                    "state":     ps.state.value,
                    "zone_bars": ps.zone_bars,
                }
                for ps in self._poi_states
            ],
        }
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _check_entry(self, bar: Bar) -> None:
        """Check each SETUP_FORMING POI for entry confirmation on this 1m bar."""
        cfg  = self._scfg
        tick = cfg.tick_size
        slip = cfg.slippage_ticks * tick
        cbuf = cfg.entry_clean_close_ticks * tick

        for ps in self._poi_states:
            if ps.state != POIZoneState.SETUP_FORMING:
                continue

            direction = _entry_direction(ps.poi.bias, bar, ps.poi.price,
                                         self._cur_macd_sl)
            if direction is None:
                continue
            if direction == "long"  and not cfg.allow_longs:
                continue
            if direction == "short" and not cfg.allow_shorts:
                continue

            macd_ok = _macd_aligned(ps.poi.bias, self._cur_macd_sl,
                                    cfg.macd_slope_threshold)
            ema_ok  = _ema_aligned(ps.poi.bias, self._cur_ema_sl,
                                   cfg.ema_slope_threshold)
            if not (macd_ok and ema_ok):
                continue

            if not _clean_close(direction, bar, ps.poi.price, cbuf):
                logger.debug(
                    f"POI#{ps.poi.rank}  unclean close  dir={direction}  "
                    f"bar=[{bar.low},{bar.high}]  level={ps.poi.price}"
                )
                continue

            # ── Build order parameters ────────────────────────────────────────
            ep       = bar.close + slip if direction == "long" else bar.close - slip
            atr_dist = min(cfg.stop_atr_pct * self._daily_atr, 25.0)
            stp      = ep - atr_dist if direction == "long" else ep + atr_dist
            risk_pts = abs(ep - stp)

            nxt_poi = _next_poi_target(direction, ep, self._poi_states)
            tgt     = (nxt_poi if nxt_poi is not None
                       else ep + risk_pts * cfg.target_r_multiple if direction == "long"
                       else ep - risk_pts * cfg.target_r_multiple)

            if direction == "long"  and tgt <= ep:
                continue
            if direction == "short" and tgt >= ep:
                continue

            sl_ticks = max(1, round(abs(ep - stp) / tick))
            tp_ticks = max(1, round(abs(ep - tgt) / tick))
            side     = SIDE_BUY if direction == "long" else SIDE_SELL
            tag      = f"POI#{ps.poi.rank}_{direction[0].upper()}"

            logger.info(
                f"ENTRY  POI#{ps.poi.rank}@{ps.poi.price}  {direction.upper()}  "
                f"ep={ep:.2f}  sl={stp:.2f}({sl_ticks}t)  tp={tgt:.2f}({tp_ticks}t)"
            )

            try:
                place_bracket_order(
                    client            = self._client,
                    account_id        = self._account_id,
                    contract_id       = self._contract_id,
                    side              = side,
                    size              = cfg.contracts,
                    stop_loss_ticks   = sl_ticks,
                    take_profit_ticks = tp_ticks,
                    custom_tag        = tag,
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
                continue

            self._set_poi_state(ps, POIZoneState.ENTRY_CONFIRMED,
                                bar.timestamp, f"entry_{direction}", bar.close)
            self._in_trade   = True
            self._trade_dir  = direction
            self._trade_stop = stp
            self._trade_tgt  = tgt
            self._active_poi = ps

            self._slack.send_trade_alert(
                symbol    = self._contract_id.split(".")[-1],
                direction = direction.upper(),
                entry     = ep,
                stop      = stp,
                target    = tgt,
                poi_label = ps.poi.label,
            )
            break   # one trade at a time

    def _set_poi_state(self, ps: _POIState, new: POIZoneState,
                       bar_ts: datetime, reason: str, price: float) -> None:
        old = ps.state
        ps.state = new
        self._logs.append({
            "timestamp":    bar_ts.isoformat(),
            "poi_rank":     ps.poi.rank,
            "poi_price":    ps.poi.price,
            "state_before": old.value,
            "state_after":  new.value,
            "reason":       reason,
            "price":        price,
            "macd_slope":   self._cur_macd_sl,
            "ema_slope":    self._cur_ema_sl,
        })
        logger.info(
            f"POI#{ps.poi.rank}@{ps.poi.price}  "
            f"{old.value} → {new.value}  [{reason}]  px={price:.2f}"
        )

    def _emergency_flatten(self) -> None:
        try:
            flatten_position(self._client, self._account_id, self._contract_id)
        except Exception as exc:
            logger.critical(f"Emergency flatten failed: {exc}")
        self._in_trade   = False
        self._trade_dir  = None
        self._active_poi = None

    def _shutdown(self) -> None:
        logger.info("Shutting down LiveConnection …")
        try:
            if self._live_conn is not None:
                self._live_conn.stop()
        except Exception:
            pass

        _PLAN_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _PLAN_DIR / f"poi_trades_{datetime.now(_ET).date()}.json"
        log_path.write_text(json.dumps({
            "trades":     self._trades,
            "logs":       self._logs,
            "day_pnl":    self._day_pnl,
            "day_trades": self._day_trades,
        }, indent=2))
        logger.info(f"Trade log saved → {log_path}")
        logger.info(
            f"Session complete  "
            f"trades={self._day_trades}  day_pnl=${self._day_pnl:,.2f}"
        )

        if not self._locked:   # didn't already send summary on daily limit hit
            wins   = sum(1 for t in self._trades if t["pnl"] > 0)
            losses = len(self._trades) - wins
            wr     = wins / len(self._trades) if self._trades else 0.0
            self._slack.send_daily_summary(
                total_pnl = self._day_pnl,
                win_rate  = wr,
                trades    = self._day_trades,
                wins      = wins,
                losses    = losses,
            )

    # ── ATR helper ────────────────────────────────────────────────────────────

    def _estimate_daily_atr(self, bars: list[Bar], period: int = 14) -> float:
        """
        Aggregate 1m bars into session buckets and compute Wilder ATR.
        Bars after 17:00 ET are assigned to the next calendar date to keep
        each ~23h Globex session as one 'daily' bar.
        """
        buckets: dict[date, list[Bar]] = defaultdict(list)
        for b in bars:
            ts_et = b.timestamp.astimezone(_ET)
            d     = ts_et.date()
            if ts_et.hour >= 17:
                d = d + timedelta(days=1)
            buckets[d].append(b)

        daily: list[Bar] = []
        for d in sorted(buckets.keys()):
            bs = buckets[d]
            if len(bs) < 10:
                continue
            daily.append(Bar(
                timestamp = bs[0].timestamp,
                open      = bs[0].open,
                high      = max(b.high for b in bs),
                low       = min(b.low  for b in bs),
                close     = bs[-1].close,
                volume    = sum(b.volume for b in bs),
            ))

        return _compute_daily_atr(daily, period)
