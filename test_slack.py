#!/usr/bin/env python3
"""
Quick test — sends a few sample messages to your Slack channel.
Run this to confirm your bot token and channel are wired up correctly.

    python3 test_slack.py
"""
from dotenv import load_dotenv
load_dotenv()

from trading_agent.notifier import SlackNotifier

notifier = SlackNotifier()

notifier.send_message("*Trading Bot Connected* — BladeTrade is online")

notifier.send_trade_alert(
    symbol    = "MES",
    direction = "LONG",
    entry     = 6350.25,
    stop      = 6345.25,
    target    = 6360.25,
    poi_label = "PDH / ATR +50%",
)

notifier.send_trade_closed(symbol="MES", pnl=125.00)
notifier.send_trade_closed(symbol="MES", pnl=-87.50)

notifier.send_daily_summary(
    total_pnl = 37.50,
    win_rate  = 0.667,
    trades    = 3,
    wins      = 2,
    losses    = 1,
)

print("Done — check your Slack channel.")
