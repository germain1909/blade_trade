from __future__ import annotations

"""
Slack notification service for the live trading system.

Reads SLACK_BOT_TOKEN and SLACK_CHANNEL from environment / .env.
All methods are fire-and-forget — a Slack failure never affects trading.

Setup:
  1. Create a Slack app at api.slack.com/apps
  2. Add the OAuth scope:  chat:write
  3. Install the app to your workspace and copy the Bot Token (xoxb-...)
  4. Invite the bot to your channel:  /invite @YourBotName
  5. Add to .env:
       SLACK_BOT_TOKEN=xoxb-...
       SLACK_CHANNEL=#trades
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Sends trading notifications to a Slack channel via slack_sdk.

    If SLACK_BOT_TOKEN is not set, all methods silently no-op so the
    trading system runs fine without Slack configured.
    """

    def __init__(
        self,
        token:   Optional[str] = None,
        channel: Optional[str] = None,
    ) -> None:
        self._token   = token   or os.getenv("SLACK_BOT_TOKEN", "")
        self._channel = channel or os.getenv("SLACK_CHANNEL", "#trades")
        self._user_id = os.getenv("SLACK_USER_ID", "")   # @mention on trade alerts
        self._client  = None

        if self._token:
            try:
                from slack_sdk import WebClient
                self._client = WebClient(token=self._token)
                logger.info(f"SlackNotifier ready → {self._channel}")
            except ImportError:
                logger.warning(
                    "slack_sdk not installed — Slack notifications disabled. "
                    "Run:  pip install slack_sdk"
                )
        else:
            logger.info("SLACK_BOT_TOKEN not set — Slack notifications disabled")

    # ── Core send ─────────────────────────────────────────────────────────────

    def send_message(self, text: str) -> None:
        """Send a plain text message. Never raises."""
        if not self._client:
            return
        try:
            self._client.chat_postMessage(channel=self._channel, text=text)
        except Exception as exc:
            logger.warning(f"Slack send_message failed: {exc}")

    # ── Trading events ────────────────────────────────────────────────────────

    def send_trade_alert(
        self,
        symbol:    str,
        direction: str,    # "LONG" | "SHORT"
        entry:     float,
        stop:      float,
        target:    float,
        poi_label: str = "",
    ) -> None:
        """
        Notify when a bracket order is placed.

        [MES LONG]
        Entry:  6350.25
        Stop:   6345.25
        Target: 6360.25
        Level:  PDH / ATR +50%
        """
        risk   = abs(entry - stop)
        rwd    = abs(target - entry)
        rr     = rwd / risk if risk > 0 else 0.0
        mention = f"<@{self._user_id}> " if self._user_id else ""
        lines = [
            f"{mention}*[{symbol} {direction.upper()}]*",
            f"Entry:  {entry:.2f}",
            f"Stop:   {stop:.2f}  ({risk:.2f} pts)",
            f"Target: {target:.2f}  ({rr:.1f}R)",
        ]
        if poi_label:
            lines.append(f"Level:  {poi_label}")
        self.send_message("\n".join(lines))

    def send_trade_closed(
        self,
        symbol: str,
        pnl:    float,
        result: str = "",   # "WIN" | "LOSS" | "BE" — auto-detected if blank
    ) -> None:
        """
        Notify when a bracket TP or SL fills.

        [MES CLOSED]
        Result: WIN
        PnL:   +$125.00
        """
        if not result:
            result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
        icon    = "✅" if pnl > 0 else ("❌" if pnl < 0 else "➖")
        mention = f"<@{self._user_id}> " if self._user_id else ""
        lines = [
            f"{mention}{icon} *[{symbol} CLOSED]*",
            f"Result: {result}",
            f"PnL:    {'+' if pnl >= 0 else ''}${pnl:,.2f}",
        ]
        self.send_message("\n".join(lines))

    def send_daily_summary(
        self,
        total_pnl: float,
        win_rate:  float,   # 0.0–1.0
        trades:    int,
        wins:      int = 0,
        losses:    int = 0,
    ) -> None:
        """
        End-of-session summary.

        Daily Summary
        PnL:      +$375.00
        Trades:   3  (2W / 1L)
        Win rate: 66.7%
        """
        icon  = "📈" if total_pnl >= 0 else "📉"
        lines = [
            f"{icon} *Daily Summary*",
            f"PnL:      {'+' if total_pnl >= 0 else ''}{total_pnl:,.2f}",
            f"Trades:   {trades}  ({wins}W / {losses}L)",
            f"Win rate: {win_rate * 100:.1f}%",
        ]
        self.send_message("\n".join(lines))

    def send_session_started(self, contract: str, pois: list[str]) -> None:
        lines = ["*Session started*", f"Contract: {contract}", "POIs:"]
        lines += [f"  • {p}" for p in pois]
        self.send_message("\n".join(lines))

    def send_daily_limit_hit(self, reason: str, day_pnl: float) -> None:
        lines = [
            "⚠️ *Daily limit reached — no more entries*",
            f"Reason:  {reason}",
            f"Day PnL: {'+' if day_pnl >= 0 else ''}{day_pnl:,.2f}",
        ]
        self.send_message("\n".join(lines))
