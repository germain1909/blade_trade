from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .projectx_client import Bar
from .risk_manager import Exit, ExitReason, RiskConfig, RiskManager
from .strategy import Signal, Strategy


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:  datetime
    exit_time:   datetime
    direction:   str        # "long" | "short"
    entry_price: float
    exit_price:  float
    contracts:   int
    pnl_usd:     float
    exit_reason: str        # "take_profit" | "stop_loss"
    bias_source: str = ""   # e.g. "PDH", "Asia High", "London Low"
    session:     str = ""   # "london" | "ny"
    atr_pct:     Optional[float] = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    trades:           list[Trade]
    equity_curve:     list[float]
    total_pnl:        float
    max_drawdown_usd: float
    win_rate:         float
    num_trades:       int
    num_wins:         int
    num_losses:       int
    avg_win_usd:      float
    avg_loss_usd:     float
    profit_factor:    float

    def summary(self) -> str:
        pf = f"{self.profit_factor:.2f}" if self.profit_factor != float("inf") else "∞"
        lines = [
            f"Trades        : {self.num_trades}  ({self.num_wins}W / {self.num_losses}L)",
            f"Win rate      : {self.win_rate * 100:.1f}%",
            f"Total PnL     : ${self.total_pnl:>10,.2f}",
            f"Max drawdown  : ${self.max_drawdown_usd:>10,.2f}",
            f"Avg win       : ${self.avg_win_usd:>10,.2f}",
            f"Avg loss      : ${self.avg_loss_usd:>10,.2f}",
            f"Profit factor : {pf}",
        ]
        return "\n".join(lines)

    def breakdown(self, group_by: str = "bias_source", title: str = "Case") -> str:
        """Trade statistics table grouped by `group_by` field (bias_source or session)."""
        if not self.trades:
            return "No trades."

        groups: dict[str, list[Trade]] = defaultdict(list)
        for t in self.trades:
            key = getattr(t, group_by, "") or "unknown"
            groups[key].append(t)

        col = 14
        header = (
            f"{title:<{col}}  {'N':>4}  {'W':>3}  {'L':>3}  "
            f"{'WR%':>5}  {'PnL':>10}  {'AvgW':>8}  {'AvgL':>8}  {'PF':>5}"
        )
        sep = "-" * len(header)
        rows = [header, sep]

        for src in sorted(groups.keys()):
            grp    = groups[src]
            wins   = [t for t in grp if t.pnl_usd > 0]
            losses = [t for t in grp if t.pnl_usd <= 0]
            g_win  = sum(t.pnl_usd for t in wins)
            g_loss = abs(sum(t.pnl_usd for t in losses))
            pnl    = sum(t.pnl_usd for t in grp)
            wr     = len(wins) / len(grp) * 100
            avg_w  = g_win  / len(wins)   if wins   else 0.0
            avg_l  = g_loss / len(losses) if losses else 0.0
            pf_val = g_win / g_loss        if g_loss  else float("inf")
            pf_str = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"
            rows.append(
                f"{src:<{col}}  {len(grp):>4}  {len(wins):>3}  {len(losses):>3}  "
                f"{wr:>5.1f}  ${pnl:>9,.2f}  ${avg_w:>7,.2f}  ${avg_l:>7,.2f}  {pf_str:>5}"
            )

        rows.append(sep)
        all_w  = [t for t in self.trades if t.pnl_usd > 0]
        all_l  = [t for t in self.trades if t.pnl_usd <= 0]
        g_win  = sum(t.pnl_usd for t in all_w)
        g_loss = abs(sum(t.pnl_usd for t in all_l))
        pf_val = g_win / g_loss if g_loss else float("inf")
        pf_str = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"
        rows.append(
            f"{'TOTAL':<{col}}  {self.num_trades:>4}  {self.num_wins:>3}  {self.num_losses:>3}  "
            f"{self.win_rate*100:>5.1f}  ${self.total_pnl:>9,.2f}  "
            f"${self.avg_win_usd:>7,.2f}  ${self.avg_loss_usd:>7,.2f}  {pf_str:>5}"
        )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    bars:        list[Bar],
    strategy:    Strategy,
    risk_config: Optional[RiskConfig] = None,
) -> BacktestResult:
    """
    Walk-forward single-pass backtest.

    Flow per bar:
      1. Check open position for TP/SL hit  (uses bar high/low)
      2. Ask strategy for a signal          (strategy sees this bar's close)
      3. If signal is LONG/SHORT and no position open, enter at this bar's close

    Only one position is held at a time.  New signals are ignored while in position.
    bias_source metadata is threaded from the strategy through to each Trade record
    so results can be broken down by which level triggered the trade.
    """
    risk = RiskManager(risk_config)
    strategy.reset()
    risk.reset()

    trades:   list[Trade] = []
    # Map entry_timestamp → strategy metadata captured at signal time
    pending: dict[datetime, dict] = {}

    equity       = 0.0
    peak         = 0.0
    max_drawdown = 0.0
    equity_curve: list[float] = []

    for bar in bars:
        # Step 1 — check TP/SL on the current bar before strategy processes it
        exit_event: Optional[Exit] = risk.on_bar(bar)
        if exit_event is not None:
            equity += exit_event.pnl_usd
            peak         = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            equity_curve.append(equity)

            meta = pending.pop(exit_event.entry_time, {})
            trades.append(Trade(
                entry_time=exit_event.entry_time,
                exit_time=exit_event.exit_time,
                direction=exit_event.direction.value,
                entry_price=exit_event.entry_price,
                exit_price=exit_event.exit_price,
                contracts=exit_event.contracts,
                pnl_usd=exit_event.pnl_usd,
                exit_reason=exit_event.reason.value,
                bias_source=meta.get("bias_source", ""),
                session=meta.get("session", ""),
                atr_pct=meta.get("atr_pct"),
            ))

        # Step 2 — strategy signal
        signal = strategy.on_bar(bar)

        # Step 3 — open position, using strategy's dynamic stop/target when available
        if signal in (Signal.LONG, Signal.SHORT) and not risk.in_position:
            risk.open_position(
                signal,
                bar.close,
                bar.timestamp,
                stop_price=strategy.last_stop_price(),
                target_price=strategy.last_target_price(),
            )
            # Capture entry metadata keyed by entry timestamp
            pending[bar.timestamp] = strategy.last_trade_metadata()

    # Aggregate stats
    wins       = [t for t in trades if t.pnl_usd > 0]
    losses     = [t for t in trades if t.pnl_usd <= 0]
    gross_win  = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        total_pnl=round(equity, 2),
        max_drawdown_usd=round(max_drawdown, 2),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        num_trades=len(trades),
        num_wins=len(wins),
        num_losses=len(losses),
        avg_win_usd=round(gross_win  / len(wins),   2) if wins   else 0.0,
        avg_loss_usd=round(gross_loss / len(losses), 2) if losses else 0.0,
        profit_factor=round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_excel_bars(paths: list[Path]) -> list[Bar]:
    """Merge and sort bars from one or more .xlsx or .csv files."""
    frames = []
    for p in paths:
        if p.suffix == ".csv":
            frames.append(pd.read_csv(p))
        else:
            frames.append(pd.read_excel(p, engine="openpyxl"))
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts = row.timestamp
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        bars.append(Bar(
            timestamp=ts,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume),
        ))
    return bars
