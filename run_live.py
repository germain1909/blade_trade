#!/usr/bin/env python3
"""
Quickstart runner for the POI live trading pipeline.

Usage:
    python run_live.py prep               # 5pm ET — scan POIs, save plan
    python run_live.py session            # session open — connect & trade

Set ACCOUNT_ID and CONTRACT_ID below, or pass as env vars.
"""
import logging
import os
import sys
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)

# ── YOUR SETTINGS ─────────────────────────────────────────────────────────────
ACCOUNT_ID        = int(os.getenv("TOPSTEP_ACCOUNT_ID", "0"))
CONTRACT_ID       = os.getenv("TOPSTEP_CONTRACT_ID", "CON.F.US.MES.U26")   # active contract
HISTORY_CONTRACT  = os.getenv("TOPSTEP_HISTORY_CONTRACT", "")              # old contract for history after a roll
# ─────────────────────────────────────────────────────────────────────────────

from trading_agent.config import load_config
from trading_agent.live_runner import POILiveSession
from trading_agent.poi_strategy import POIStrategyConfig
from datetime import time as dtime

# Optional: customise strategy config here (or leave as default)
strategy_cfg = POIStrategyConfig(
    zone_buffer_ticks        = 8,
    max_zone_bars            = 20,
    require_macd_slope_align = True,
    require_ema_slope_align  = False,
    entry_clean_close_ticks  = 8,
    stop_atr_pct             = 0.25,
    use_atr_stop             = True,
    target_r_multiple        = 2.0,
    contracts                = 1,        # ← number of MES contracts
    slippage_ticks           = 1,
    max_trades_per_day       = 3,
    daily_loss_limit_usd     = -500.0,
    daily_profit_limit_usd   = 1500.0,
    trade_start_et           = dtime(9, 30),   # NY open
    entry_cutoff_et          = dtime(15, 30),  # no new entries after 3:30pm ET
    trade_end_et             = dtime(16, 0),   # flatten by 4pm ET
)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("prep", "session"):
        print(__doc__)
        sys.exit(1)

    if ACCOUNT_ID == 0 and sys.argv[1] == "session":
        print("ERROR: Set TOPSTEP_ACCOUNT_ID env var or edit ACCOUNT_ID in this file.")
        print("       Run:  python -m trading_agent accounts")
        sys.exit(1)

    config  = load_config()
    session = POILiveSession(
        config       = config,
        account_id   = ACCOUNT_ID,
        contract_id  = CONTRACT_ID,
        strategy_cfg = strategy_cfg,
    )

    if sys.argv[1] == "prep":
        pois = session.daily_prep(history_contract_id=HISTORY_CONTRACT or None)
        print(f"\nSaved {len(pois)} POIs to data/poi_plan_*.json")

    elif sys.argv[1] == "session":
        print(f"Starting live session — account={ACCOUNT_ID}  contract={CONTRACT_ID}")
        print("Press Ctrl-C to stop cleanly.")
        session.run()


if __name__ == "__main__":
    main()
