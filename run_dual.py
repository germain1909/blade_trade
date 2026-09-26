#!/usr/bin/env python3
"""
Dual strategy runner: Pivot+Supertrend + ORB

Usage:
    python run_dual.py prep               # 5pm ET — compute pivots, save plan
    python run_dual.py session            # session open — connect & trade
    python run_dual.py prep --gold        # prep for MGC
    python run_dual.py session --gold     # trade MGC
"""
import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

ACCOUNT_ID   = int(os.getenv("TOPSTEP_ACCOUNT_ID", "0"))
CONTRACT_ID  = os.getenv("TOPSTEP_CONTRACT_ID", "CON.F.US.MES.U26")
GOLD_CONTRACT_ID = os.getenv("TOPSTEP_GOLD_CONTRACT_ID", "CON.F.US.MGC.Q26")
HISTORY_CONTRACT = os.getenv("TOPSTEP_HISTORY_CONTRACT", "")

from trading_agent.config import load_config
from trading_agent.dual_live_runner import DualLiveSession, DualStrategyConfig
from trading_agent.supertrend_strategy import PivotSupertrendConfig
from trading_agent.strategy import ORBreakoutConfig
from datetime import time as dtime


# ── MES config ───────────────────────────────────────────────────────────────
mes_cfg = DualStrategyConfig(
    pivot_cfg=PivotSupertrendConfig(
        st_bar_minutes=30,
        rej_bar_minutes=5,
        zone_buffer_pts=8.0,
        min_rr=1.5,
        contracts=1,
        tick_size=0.25,
        dollars_per_point=5.0,
        slippage_ticks=1,
        trade_start_et=dtime(8, 30),
        entry_cutoff_et=dtime(15, 0),
        trade_end_et=dtime(16, 0),
    ),
    orb_cfg=ORBreakoutConfig(
        windows=((8, 30),),
        window_hours=7,
        or_minutes=30,
        stop_ticks=24,
        tick_size=0.25,
        target_rr=2.0,
    ),
    contracts=1,
    max_trades_per_day=6,
    daily_loss_limit=-500.0,
    daily_profit_limit=1500.0,
    trade_start_et=dtime(8, 30),
    entry_cutoff_et=dtime(15, 0),
    trade_end_et=dtime(16, 0),
)

# ── MGC config (3:00 AM - 12:00 PM ET) ──────────────────────────────────────
mgc_cfg = DualStrategyConfig(
    pivot_cfg=PivotSupertrendConfig(
        st_bar_minutes=30,
        rej_bar_minutes=5,
        zone_buffer_pts=15.0,
        min_rr=1.5,
        contracts=1,
        tick_size=0.10,
        dollars_per_point=10.0,
        slippage_ticks=2,
        trade_start_et=dtime(3, 0),
        entry_cutoff_et=dtime(11, 0),
        trade_end_et=dtime(12, 0),
    ),
    orb_cfg=ORBreakoutConfig(
        windows=((3, 0),),
        window_hours=9,
        or_minutes=30,
        stop_ticks=30,
        tick_size=0.10,
        target_rr=2.0,
    ),
    contracts=1,
    max_trades_per_day=6,
    daily_loss_limit=-500.0,
    daily_profit_limit=1500.0,
    trade_start_et=dtime(3, 0),
    entry_cutoff_et=dtime(11, 0),
    trade_end_et=dtime(12, 0),
)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("prep", "session"):
        print(__doc__)
        sys.exit(1)

    gold = "--gold" in sys.argv

    if ACCOUNT_ID == 0 and sys.argv[1] == "session":
        print("ERROR: Set TOPSTEP_ACCOUNT_ID in .env")
        sys.exit(1)

    contract = GOLD_CONTRACT_ID if gold else CONTRACT_ID
    cfg      = mgc_cfg if gold else mes_cfg
    label    = "MGC (3AM-12PM ET)" if gold else "MES (8:30AM-4PM ET)"

    config  = load_config()
    session = DualLiveSession(
        config=config,
        account_id=ACCOUNT_ID,
        contract_id=contract,
        strategy_cfg=cfg,
    )

    if sys.argv[1] == "prep":
        session.daily_prep(history_contract_id=HISTORY_CONTRACT or None)
        print(f"Daily prep complete ({label}) — pivots saved")

    elif sys.argv[1] == "session":
        print(f"Starting {label} — account={ACCOUNT_ID}  contract={contract}")
        print("Strategies: Pivot+Supertrend, ORB")
        print("Press Ctrl-C to stop cleanly.")
        session.run()


if __name__ == "__main__":
    main()
