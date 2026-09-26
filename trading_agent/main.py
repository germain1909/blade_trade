#!/usr/bin/env python3
"""
trading_agent — milestone 1 CLI

Usage examples:
  python -m trading_agent accounts
  python -m trading_agent contracts --symbol MES
  python -m trading_agent download --symbol MES --start 2025-01-01 --end 2025-06-01
  python -m trading_agent download --symbol MES --start 2025-01-01 --end 2025-06-01 --sqlite
  python -m trading_agent download --symbol MES --start 2025-01-01 --end 2026-01-01 --continuous
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .backtester import BacktestResult, load_excel_bars, run_backtest
from .orb_research import (
    ORBConfig, export_trades_csv, print_research_summary, run_orb_research,
)
from .atr_levels import (
    ATRLevelConfig, export_atr_trades_csv, print_atr_summary, run_atr_research,
)
from .atr_state_machine import (
    ATRSMConfig, export_atr_sm_trades_csv, export_state_logs_csv,
    print_atr_sm_summary, run_atr_sm_research,
)
from .config import load_config
from .data_downloader import download_1m_bars, download_continuous_1m_bars
from .projectx_client import ProjectXClient
from .risk_manager import RiskConfig
from .storage import csv_path, load_csv, save_csv, save_sqlite, save_monthly_excel
from .strategy import ORBreakoutConfig, ORBreakoutStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_accounts(_args: argparse.Namespace) -> None:
    """List all active trading accounts."""
    cfg = load_config()
    client = ProjectXClient(cfg)
    client.authenticate()

    accounts = client.search_accounts(only_active=True)
    if not accounts:
        print("No active accounts found.")
        return

    print(f"\n{'ID':>10}  {'Name':<30}  {'Balance':>14}  Status")
    print("-" * 65)
    for a in accounts:
        status = "tradeable" if a.can_trade else "no-trade"
        print(f"{a.id:>10}  {a.name:<30}  {a.balance:>14,.2f}  {status}")


def cmd_contracts(args: argparse.Namespace) -> None:
    """Search for contracts by symbol."""
    cfg = load_config()
    client = ProjectXClient(cfg)
    client.authenticate()

    contracts = client.search_contracts(args.symbol, live=args.live)
    if not contracts:
        print(f"No contracts found for '{args.symbol}'.")
        return

    print(f"\n{'Active':<7}  {'Contract ID':<30}  {'Name':<10}  Description")
    print("-" * 90)
    for c in contracts:
        active = "  ✓  " if c.active_contract else "     "
        print(f"{active}  {c.id:<30}  {c.name:<10}  {c.description}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download historical 1-minute bars and save to CSV (or SQLite)."""
    cfg = load_config()
    client = ProjectXClient(cfg)
    client.authenticate()

    # 1. Active account (informational — not needed for bar download)
    accounts = client.search_accounts(only_active=True)
    if accounts:
        a = accounts[0]
        logger.info("Account: %s (id=%d, balance=%.2f)", a.name, a.id, a.balance)
    else:
        logger.warning("No active accounts found — continuing anyway.")

    # 2. Resolve contract — always use live=False for search (live=True returns nothing)
    contracts = client.search_contracts(args.symbol, live=False)
    if not contracts:
        logger.error("No contracts found for symbol '%s'. Try --symbol ES or MNQ.", args.symbol)
        sys.exit(1)

    contract = contracts[0]
    logger.info(
        "Contract: %s — %s  tick_size=%.4f  tick_value=%.2f",
        contract.id, contract.description, contract.tick_size, contract.tick_value,
    )

    # 3. Parse date range
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # 4. Download — continuous mode stitches across quarterly contract rollovers
    if args.continuous:
        bars = download_continuous_1m_bars(client, args.symbol, start, end, live=args.live)
    else:
        bars = download_1m_bars(client, contract.id, start, end, live=args.live)
    if not bars:
        logger.warning("No bars returned for the requested range.")
        sys.exit(0)

    # Filter to requested date range — session windows bleed into adjacent months
    # without this, a June download overwrites the May file with 60 bleed bars
    bars = [b for b in bars if start.date() <= b.timestamp.date() <= end.date()]
    if not bars:
        logger.warning("No bars within the requested date range after filtering.")
        sys.exit(0)

    # 5. Persist
    label = args.symbol.upper() if args.continuous else contract.id.replace(".", "_")

    if args.excel:
        paths = save_monthly_excel(bars, cfg.data_dir, prefix=label)
        print(f"\nSaved {len(bars):,} bars across {len(paths)} monthly Excel files in {cfg.data_dir}/")
        for p in paths:
            print(f"  {p.name}")
    elif args.sqlite:
        db_path = cfg.data_dir / f"{label}.db"
        table = f"{label}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_1m"
        save_sqlite(bars, db_path, table=table)
        print(f"\nSaved {len(bars):,} bars → {db_path}  [table={table}]")
    else:
        out = csv_path(cfg.data_dir, contract.id, start, end)
        save_csv(bars, out)
        print(f"\nSaved {len(bars):,} bars → {out}")

    if bars:
        print(f"Range : {bars[0].timestamp.isoformat()} → {bars[-1].timestamp.isoformat()}")
        print(f"Open  : {bars[0].open}   Close : {bars[-1].close}")


# ---------------------------------------------------------------------------
# Sub-command: orb (pure Opening Range Breakout research)
# ---------------------------------------------------------------------------

def cmd_orb(args: argparse.Namespace) -> None:
    cfg_app = load_config()
    prefix  = args.symbol.upper()
    xlsx_files = sorted(cfg_app.data_dir.glob(f"{prefix}_*_1m.xlsx"))
    if not xlsx_files:
        logger.error("No Excel files found for '%s' in %s.", prefix, cfg_app.data_dir)
        sys.exit(1)

    logger.info("Loading bars from %d file(s):", len(xlsx_files))
    for p in xlsx_files:
        logger.info("  %s", p.name)

    bars = load_excel_bars(xlsx_files)
    if not bars:
        logger.error("No bars loaded.")
        sys.exit(1)
    logger.info("Loaded %d bars  (%s → %s)", len(bars),
                bars[0].timestamp.date(), bars[-1].timestamp.date())

    cfg = ORBConfig(
        or_minutes              = args.or_minutes,
        sessions                = tuple(args.sessions),
        max_trades_per_session  = args.max_trades,
        session_loss_limit_r    = args.loss_limit,
        stop_methods            = tuple(args.stop_methods),
        fixed_stop_ticks        = args.stop_ticks,
        tick_size               = args.tick_size,
        or_pct_stop             = args.or_pct,
        targets_r               = tuple(args.targets),
        contracts               = args.contracts,
        dollars_per_point       = args.dpp,
    )

    trades_by_combo, session_stats, total_days = run_orb_research(bars, cfg)

    print_research_summary(trades_by_combo, session_stats, total_days, cfg)

    # CSV export — all trades for the default (first stop method, first target)
    if args.csv:
        all_trades = [t for tl in trades_by_combo.values() for t in tl]
        out = cfg_app.data_dir / f"{prefix}_orb_trades.csv"
        export_trades_csv(all_trades, out)
        print(f"  Exported {len(all_trades):,} trade rows → {out}")


# ---------------------------------------------------------------------------
# Sub-command: atr (ATR Percentage Level research)
# ---------------------------------------------------------------------------

def cmd_atr(args: argparse.Namespace) -> None:
    cfg_app = load_config()
    prefix  = args.symbol.upper()
    xlsx_files = sorted(cfg_app.data_dir.glob(f"{prefix}_*_1m.xlsx"))
    if not xlsx_files:
        logger.error("No Excel files found for '%s' in %s.", prefix, cfg_app.data_dir)
        sys.exit(1)

    logger.info("Loading bars from %d file(s):", len(xlsx_files))
    for p in xlsx_files:
        logger.info("  %s", p.name)

    bars = load_excel_bars(xlsx_files)
    if not bars:
        logger.error("No bars loaded.")
        sys.exit(1)
    logger.info("Loaded %d bars  (%s → %s)", len(bars),
                bars[0].timestamp.date(), bars[-1].timestamp.date())

    cfg = ATRLevelConfig(
        atr_period            = args.atr_period,
        ema_period            = args.ema_period,
        bar_minutes           = args.bar_minutes,
        contracts             = args.contracts,
        tick_size             = args.tick_size,
        slippage_ticks        = args.slippage_ticks,
        dollars_per_point     = args.dpp,
        breakeven_usd         = args.breakeven,
        max_trades_per_day    = args.max_trades,
        daily_loss_limit_usd  = args.daily_loss,
        daily_profit_limit_usd= args.daily_profit,
        blacklist_on_stop     = not args.no_blacklist,
    )

    trades = run_atr_research(bars, cfg)
    print_atr_summary(trades, cfg)

    if args.csv and trades:
        out = cfg_app.data_dir / f"{prefix}_atr_trades.csv"
        export_atr_trades_csv(trades, out)
        print(f"  Exported {len(trades):,} trade rows → {out}")

    if args.trades and trades:
        print(f"\n{'Entry (ET)':<22}  {'Dir':<6}  {'Level':>6}  "
              f"{'Entry':>8}  {'Exit':>8}  {'Reason':<16}  {'PnL':>9}")
        print("-" * 90)
        for t in trades:
            from zoneinfo import ZoneInfo
            _ET2 = ZoneInfo("America/New_York")
            et_ts = t.entry_time
            if et_ts.tzinfo is None:
                et_ts = et_ts.replace(tzinfo=__import__("datetime").timezone.utc)
            et_ts = et_ts.astimezone(_ET2)
            lbl = f"{'+' if t.entry_atr_pct >= 0 else ''}{int(t.entry_atr_pct * 100)}%"
            print(
                f"{str(et_ts)[:19]:<22}  {t.direction:<6}  {lbl:>6}  "
                f"{t.entry_price:>8.2f}  {t.exit_price:>8.2f}  {t.exit_reason:<16}  "
                f"${t.pnl_usd:>8,.2f}"
            )


# ---------------------------------------------------------------------------
# Sub-command: atr-sm (ATR Level State Machine strategy)
# ---------------------------------------------------------------------------

def cmd_atr_sm(args: argparse.Namespace) -> None:
    cfg_app = load_config()
    prefix  = args.symbol.upper()
    xlsx_files = sorted(cfg_app.data_dir.glob(f"{prefix}_*_1m.xlsx"))
    if not xlsx_files:
        logger.error("No Excel files found for '%s' in %s.", prefix, cfg_app.data_dir)
        import sys; sys.exit(1)

    logger.info("Loading bars from %d file(s):", len(xlsx_files))
    for p in xlsx_files:
        logger.info("  %s", p.name)

    bars = load_excel_bars(xlsx_files)
    if not bars:
        logger.error("No bars loaded.")
        import sys; sys.exit(1)
    logger.info("Loaded %d bars  (%s → %s)", len(bars),
                bars[0].timestamp.date(), bars[-1].timestamp.date())

    cfg = ATRSMConfig(
        atr_period                      = args.atr_period,
        ema_length                      = args.ema_length,
        rsi_length                      = args.rsi_length,
        rsi_ma_length                   = args.rsi_ma_length,
        slope_lookback_bars             = args.slope_lookback,
        retest_buffer_ticks             = args.retest_ticks,
        invalidation_buffer_ticks       = args.invalidation_ticks,
        stop_buffer_ticks               = args.stop_ticks,
        stop_atr_pct                    = args.stop_atr_pct,
        max_bars_after_break_for_retest = args.break_timeout,
        max_bars_after_retest_for_entry = args.retest_timeout,
        contracts                       = args.contracts,
        tick_size                       = args.tick_size,
        slippage_ticks                  = args.slippage_ticks,
        dollars_per_point               = args.dpp,
        max_trades_per_day              = args.max_trades,
        daily_loss_limit_usd            = args.daily_loss,
        daily_profit_limit_usd          = args.daily_profit,
        allow_longs                     = not args.no_longs,
        allow_shorts                    = not args.no_shorts,
    )

    trades, logs = run_atr_sm_research(bars, cfg)
    print_atr_sm_summary(trades, logs, cfg)

    if args.csv and trades:
        out = cfg_app.data_dir / f"{prefix}_atr_sm_trades.csv"
        export_atr_sm_trades_csv(trades, out)
        print(f"  Exported {len(trades):,} trade rows → {out}")

    if args.logs and logs:
        out = cfg_app.data_dir / f"{prefix}_atr_sm_logs.csv"
        export_state_logs_csv(logs, out)
        print(f"  Exported {len(logs):,} log rows  → {out}")


# ---------------------------------------------------------------------------
# Sub-command: backtest
# ---------------------------------------------------------------------------

def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a walk-forward backtest against downloaded Excel bar files."""
    cfg = load_config()

    prefix = args.symbol.upper()
    xlsx_files = sorted(cfg.data_dir.glob(f"{prefix}_*_1m.xlsx"))
    if not xlsx_files:
        logger.error(
            "No Excel files found for '%s' in %s.\n"
            "Run:  python -m trading_agent download --symbol %s --start ... --end ... --continuous --excel",
            prefix, cfg.data_dir, prefix,
        )
        sys.exit(1)

    logger.info("Loading bars from %d file(s):", len(xlsx_files))
    for p in xlsx_files:
        logger.info("  %s", p.name)

    bars = load_excel_bars(xlsx_files)
    if not bars:
        logger.error("No bars loaded from Excel files.")
        sys.exit(1)
    logger.info("Loaded %d bars  (%s → %s)", len(bars),
                bars[0].timestamp.date(), bars[-1].timestamp.date())

    strategy = ORBreakoutStrategy(ORBreakoutConfig(
        or_minutes          = args.or_minutes,
        enable_case1        = not args.no_case1,
        enable_case2        = not args.no_case2,
        enable_case3        = not args.no_case3,
        enable_case4        = not args.no_case4,
        require_macd        = not args.no_macd,
        macd_fast           = args.macd_fast,
        macd_slow           = args.macd_slow,
        macd_signal         = args.macd_signal,
        rsi_len             = args.rsi_len,
        require_rsi_pullback= not args.no_rsi,
        long_pullback_rsi   = args.rsi_long,
        short_pullback_rsi  = args.rsi_short,
        require_vwap        = not args.no_vwap,
        stop_mode           = args.stop_mode,
        stop_ticks          = args.stop_ticks,
        tick_size           = args.tick_size,
        target_rr           = args.rr,
        swing_lookback      = args.swing_lookback,
    ))

    risk_config = RiskConfig(
        contracts         = args.contracts,
        take_profit_usd   = args.tp,
        stop_loss_usd     = args.sl,
        dollars_per_point = args.dpp,
    )

    stop_desc = (
        "fixed %d ticks (%.2f pts)" % (args.stop_ticks, args.stop_ticks * args.tick_size)
        if args.stop_mode == "fixed"
        else "swing (%d-bar lookback)" % args.swing_lookback
    )
    cases_on = ", ".join(
        "C%d" % i for i, off in enumerate(
            [args.no_case1, args.no_case2, args.no_case3, args.no_case4], start=1
        ) if not off
    )
    print(f"\nBacktest : {prefix}  |  {len(bars):,} bars  |  {args.contracts} contracts")
    print(f"OR       : {args.or_minutes}m   Windows: 04-08 / 08-12 / 12-16 ET")
    print(f"Cases    : {cases_on}   Target: {args.rr}R   Stop: {stop_desc}")
    print(f"MACD 5m  : {args.macd_fast}/{args.macd_slow}/{args.macd_signal}  "
          f"{'on' if not args.no_macd else 'off'}   "
          f"VWAP: {'on' if not args.no_vwap else 'off'}   "
          f"RSI pullback: {'on' if not args.no_rsi else 'off'}")
    print("=" * 64)

    result = run_backtest(bars, strategy, risk_config)

    print(result.summary())
    print()
    print("By case:")
    print(result.breakdown(group_by="bias_source", title="Case"))
    print()
    print("By session window:")
    print(result.breakdown(group_by="session", title="Session"))

    if args.trades and result.trades:
        print(f"\n{'Entry (UTC)':<20}  {'Win':<7}  {'Case':<7}  {'Dir':^5}  "
              f"{'Entry':>8}  {'Exit':>8}  {'Reason':<11}  {'PnL':>8}")
        print("-" * 90)
        for t in result.trades:
            print(
                f"{str(t.entry_time)[:19]:<20}  {t.session:<7}  {t.bias_source:<7}  "
                f"{t.direction:^5}  {t.entry_price:>8.2f}  {t.exit_price:>8.2f}  "
                f"{t.exit_reason:<11}  ${t.pnl_usd:>7,.2f}"
            )


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m trading_agent",
        description="ProjectX / TopstepX trading agent — milestone 1",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # accounts
    sub.add_parser("accounts", help="List active trading accounts") \
       .set_defaults(func=cmd_accounts)

    # contracts
    ct = sub.add_parser("contracts", help="Search contracts by symbol")
    ct.add_argument("--symbol", default="MES", help="Symbol root, e.g. MES, ES, NQ")
    ct.add_argument("--live", action="store_true", help="Use live subscription")
    ct.set_defaults(func=cmd_contracts)

    # download
    dl = sub.add_parser("download", help="Download historical 1-minute bars")
    dl.add_argument("--symbol",  default="MES",        help="Symbol root, e.g. MES, ES, NQ")
    dl.add_argument("--start",   required=True,         help="Start date  YYYY-MM-DD")
    dl.add_argument("--end",     required=True,         help="End date    YYYY-MM-DD")
    dl.add_argument("--live",        action="store_true", help="Use live subscription")
    dl.add_argument("--sqlite",      action="store_true", help="Save to SQLite instead of CSV")
    dl.add_argument("--continuous",  action="store_true", help="Stitch across quarterly contract rollovers (MES, MNQ, M2K …)")
    dl.add_argument("--excel",       action="store_true", help="Save one Excel file per calendar month instead of a single CSV")
    dl.set_defaults(func=cmd_download)

    # orb — pure opening range breakout research
    orb = sub.add_parser("orb", help="Pure ORB research framework (no indicators)")
    orb.add_argument("--symbol",       default="MES",   help="Symbol prefix (e.g. MES)")
    orb.add_argument("--or-minutes",   type=int,   default=30,   dest="or_minutes",
                     help="Opening range duration in minutes (default 30)")
    orb.add_argument("--sessions",     nargs="+",
                     default=["london", "ny_open", "ny_pm"],
                     choices=["london", "ny_open", "ny_pm"],
                     help="Sessions to include (default: all three)")
    orb.add_argument("--max-trades",   type=int,   default=4,    dest="max_trades",
                     help="Max trades per session (default 4)")
    orb.add_argument("--loss-limit",   type=float, default=-2.0, dest="loss_limit",
                     help="Session R loss limit before halting (default -2.0)")
    orb.add_argument("--stop-methods", nargs="+",
                     default=["opposite_or", "fixed_ticks", "or_pct"],
                     dest="stop_methods",
                     help="Stop methods to test (default: all three)")
    orb.add_argument("--stop-ticks",   type=int,   default=24,   dest="stop_ticks")
    orb.add_argument("--tick-size",    type=float, default=0.25, dest="tick_size")
    orb.add_argument("--or-pct",       type=float, default=0.5,  dest="or_pct",
                     help="OR-percentage stop: distance = or_size × this (default 0.5)")
    orb.add_argument("--targets",      nargs="+",  type=float,
                     default=[1.0, 1.5, 2.0, 3.0],
                     help="Target R multiples to test (default: 1.0 1.5 2.0 3.0)")
    orb.add_argument("--contracts",    type=int,   default=1)
    orb.add_argument("--dpp",          type=float, default=5.0,
                     help="Dollars per point (MES=5, default 5)")
    orb.add_argument("--csv",          action="store_true",
                     help="Export all trades to CSV in the data directory")
    orb.set_defaults(func=cmd_orb)

    # atr — ATR Percentage Level research
    atr = sub.add_parser("atr", help="ATR Percentage Level strategy research")
    atr.add_argument("--symbol",        default="MES",  help="Symbol prefix (e.g. MES)")
    atr.add_argument("--atr-period",    type=int,   default=14,    dest="atr_period",
                     help="Wilder ATR period for daily ATR (default 14)")
    atr.add_argument("--ema-period",    type=int,   default=7,     dest="ema_period",
                     help="EMA period on the N-minute chart for trend filter (default 7)")
    atr.add_argument("--bar-minutes",   type=int,   default=15,    dest="bar_minutes",
                     help="Chart bar size in minutes (default 15)")
    atr.add_argument("--contracts",     type=int,   default=3,     help="Contracts per trade (default 3)")
    atr.add_argument("--tick-size",     type=float, default=0.25,  dest="tick_size",
                     help="Tick size in points (MES=0.25)")
    atr.add_argument("--slippage-ticks",type=int,   default=1,     dest="slippage_ticks",
                     help="Slippage in ticks at entry (default 1)")
    atr.add_argument("--dpp",           type=float, default=5.0,
                     help="Dollars per point (MES=5, default 5)")
    atr.add_argument("--breakeven",     type=float, default=100.0,
                     help="Unrealized USD profit at which stop moves to entry (default 100)")
    atr.add_argument("--max-trades",    type=int,   default=10,    dest="max_trades",
                     help="Max trades per day (default 10)")
    atr.add_argument("--daily-loss",    type=float, default=-1000.0, dest="daily_loss",
                     help="Daily loss limit USD — halt new entries after this (default -1000)")
    atr.add_argument("--daily-profit",  type=float, default=2000.0,  dest="daily_profit",
                     help="Daily profit limit USD — halt new entries after this (default 2000)")
    atr.add_argument("--no-blacklist",  action="store_true", dest="no_blacklist",
                     help="Disable per-day level blacklist after a full stop")
    atr.add_argument("--csv",           action="store_true",
                     help="Export all trades to CSV in the data directory")
    atr.add_argument("--trades",        action="store_true",
                     help="Print full trade log after summary")
    atr.set_defaults(func=cmd_atr)

    # atr-sm — ATR Level State Machine strategy
    sm = sub.add_parser("atr-sm", help="ATR Level State Machine strategy (break→retest→entry)")
    sm.add_argument("--symbol",            default="MES",   help="Symbol prefix (e.g. MES)")
    sm.add_argument("--atr-period",        type=int,   default=14,    dest="atr_period")
    sm.add_argument("--ema-length",        type=int,   default=15,    dest="ema_length",
                    help="5M EMA period (default 15)")
    sm.add_argument("--rsi-length",        type=int,   default=14,    dest="rsi_length",
                    help="15M RSI period (default 14)")
    sm.add_argument("--rsi-ma-length",     type=int,   default=14,    dest="rsi_ma_length",
                    help="EMA period applied to RSI values (default 14)")
    sm.add_argument("--slope-lookback",    type=int,   default=3,     dest="slope_lookback",
                    help="Bars back for slope comparison (default 3)")
    sm.add_argument("--retest-ticks",      type=int,   default=4,     dest="retest_ticks",
                    help="Ticks from level that counts as retest (default 4)")
    sm.add_argument("--invalidation-ticks",type=int,   default=4,     dest="invalidation_ticks",
                    help="Max ticks close past level before invalidation (default 4)")
    sm.add_argument("--stop-ticks",        type=int,   default=4,     dest="stop_ticks",
                    help="Ticks beyond retest extreme for stop (default 4, overridden by --stop-atr-pct)")
    sm.add_argument("--stop-atr-pct",      type=float, default=0.25,  dest="stop_atr_pct",
                    help="Stop distance as fraction of daily ATR from entry (default 0.25; set 0 to use retest extreme)")
    sm.add_argument("--break-timeout",     type=int,   default=20,    dest="break_timeout",
                    help="5M bars to wait for retest before invalidating (default 20)")
    sm.add_argument("--retest-timeout",    type=int,   default=10,    dest="retest_timeout",
                    help="5M bars to wait for entry after retest (default 10)")
    sm.add_argument("--contracts",         type=int,   default=3)
    sm.add_argument("--tick-size",         type=float, default=0.25,  dest="tick_size")
    sm.add_argument("--slippage-ticks",    type=int,   default=1,     dest="slippage_ticks")
    sm.add_argument("--dpp",               type=float, default=5.0,
                    help="Dollars per point (MES=5)")
    sm.add_argument("--max-trades",        type=int,   default=3,     dest="max_trades")
    sm.add_argument("--daily-loss",        type=float, default=-600.0, dest="daily_loss")
    sm.add_argument("--daily-profit",      type=float, default=1500.0, dest="daily_profit")
    sm.add_argument("--no-longs",          action="store_true", dest="no_longs")
    sm.add_argument("--no-shorts",         action="store_true", dest="no_shorts")
    sm.add_argument("--csv",               action="store_true",
                    help="Export trades to CSV")
    sm.add_argument("--logs",              action="store_true",
                    help="Export state-transition log to CSV")
    sm.set_defaults(func=cmd_atr_sm)

    # backtest
    bt = sub.add_parser("backtest", help="Backtest ORBreakoutStrategy against downloaded data")
    bt.add_argument("--symbol",          default="MES",    help="Symbol prefix (e.g. MES)")
    bt.add_argument("--contracts",  type=int,   default=5,      help="Contracts per trade")
    bt.add_argument("--tp",         type=float, default=500.0,  help="Fallback take-profit USD")
    bt.add_argument("--sl",         type=float, default=200.0,  help="Fallback stop-loss USD")
    bt.add_argument("--dpp",        type=float, default=5.0,    help="Dollars/point — MES=5")
    # Opening range
    bt.add_argument("--or-minutes", type=int,   default=30,     dest="or_minutes",
                    help="Opening range duration in minutes (default 30)")
    # Case toggles
    bt.add_argument("--no-case1",   action="store_true", dest="no_case1",
                    help="Disable Case 1: bullish breakout continuation")
    bt.add_argument("--no-case2",   action="store_true", dest="no_case2",
                    help="Disable Case 2: bearish breakout continuation")
    bt.add_argument("--no-case3",   action="store_true", dest="no_case3",
                    help="Disable Case 3: bull trap short")
    bt.add_argument("--no-case4",   action="store_true", dest="no_case4",
                    help="Disable Case 4: bear trap long")
    # Filters
    bt.add_argument("--no-macd",    action="store_true", dest="no_macd",  help="Disable MACD filter")
    bt.add_argument("--no-vwap",    action="store_true", dest="no_vwap",  help="Disable VWAP filter")
    bt.add_argument("--no-rsi",     action="store_true", dest="no_rsi",   help="Disable RSI pullback filter")
    bt.add_argument("--macd-fast",  type=int,   default=16,     dest="macd_fast")
    bt.add_argument("--macd-slow",  type=int,   default=40,     dest="macd_slow")
    bt.add_argument("--macd-signal",type=int,   default=9,      dest="macd_signal")
    bt.add_argument("--rsi-len",    type=int,   default=14,     dest="rsi_len")
    bt.add_argument("--rsi-long",   type=float, default=40.0,   dest="rsi_long",
                    help="RSI pullback threshold for longs (default 40)")
    bt.add_argument("--rsi-short",  type=float, default=60.0,   dest="rsi_short",
                    help="RSI pullback threshold for shorts (default 60)")
    # Exits
    bt.add_argument("--stop-mode",  default="fixed",             dest="stop_mode",
                    choices=["swing", "fixed"])
    bt.add_argument("--stop-ticks", type=int,   default=24,     dest="stop_ticks")
    bt.add_argument("--tick-size",  type=float, default=0.25,   dest="tick_size",
                    help="Tick size in points (MES=0.25, MGC=0.10)")
    bt.add_argument("--rr",         type=float, default=2.0,    help="Risk:reward target (default 2.0)")
    bt.add_argument("--swing-lookback", type=int, default=10,   dest="swing_lookback")
    bt.add_argument("--trades",     action="store_true",         help="Print full trade log")
    bt.set_defaults(func=cmd_backtest)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
