from __future__ import annotations

"""
Order execution layer for ProjectX / TopstepX.

All orders go through place_bracket_order() which submits a market order
with an attached stop-loss and take-profit bracket in a single call.

If the bracket cannot be confirmed, flatten_position() is called immediately
as a safety net — never leave a naked position.
"""

import logging
from typing import Optional
from .projectx_client import ProjectXClient

logger = logging.getLogger(__name__)

# ── Order type constants ──────────────────────────────────────────────────────
ORDER_LIMIT   = 1
ORDER_MARKET  = 2
ORDER_STOP    = 4

# ── Side constants ────────────────────────────────────────────────────────────
SIDE_BUY  = 0
SIDE_SELL = 1


def place_bracket_order(
    client:             ProjectXClient,
    account_id:         int,
    contract_id:        str,
    side:               int,            # SIDE_BUY or SIDE_SELL
    size:               int,
    stop_loss_ticks:    int,
    take_profit_ticks:  int,
    custom_tag:         str = "",
) -> int:
    """
    Submit a market order with an attached stop-loss + take-profit bracket.

    Returns the orderId on success.  Raises RuntimeError if the API rejects
    the order — caller must handle and decide whether to flatten.
    """
    client.ensure_authenticated()

    payload: dict = {
        "accountId":  account_id,
        "contractId": contract_id,
        "type":       ORDER_MARKET,
        "side":       side,
        "size":       size,
        "stopLossBracket":   {"ticks": stop_loss_ticks,   "type": ORDER_STOP},
        "takeProfitBracket": {"ticks": take_profit_ticks, "type": ORDER_LIMIT},
    }
    if custom_tag:
        payload["customTag"] = custom_tag

    direction = "BUY" if side == SIDE_BUY else "SELL"
    logger.info(
        f"ORDER  {direction} {size}x {contract_id}  "
        f"SL={stop_loss_ticks}t  TP={take_profit_ticks}t  tag={custom_tag!r}"
    )

    data = client._post("/api/Order/place", payload)
    order_id: int = data.get("orderId") or data.get("order", {}).get("id", 0)
    logger.info(f"ORDER  confirmed  orderId={order_id}")
    return order_id


def cancel_order(
    client:     ProjectXClient,
    account_id: int,
    order_id:   int,
) -> None:
    client.ensure_authenticated()
    client._post("/api/Order/cancel", {"accountId": account_id, "orderId": order_id})
    logger.info(f"CANCEL orderId={order_id}")


def cancel_all_orders(
    client:      ProjectXClient,
    account_id:  int,
    contract_id: str,
) -> None:
    """Cancel all open orders for a contract — use before flattening."""
    client.ensure_authenticated()
    data = client._post("/api/Order/search", {
        "accountId":  account_id,
        "contractId": contract_id,
        "status":     [1, 2],   # 1=Pending, 2=Working
    })
    orders = data.get("orders", []) or []
    for o in orders:
        try:
            cancel_order(client, account_id, o["id"])
        except Exception as exc:
            logger.warning(f"CANCEL failed orderId={o['id']}: {exc}")


def flatten_position(
    client:      ProjectXClient,
    account_id:  int,
    contract_id: str,
) -> None:
    """
    Emergency flatten — cancels all open orders then closes the position.
    Called automatically if bracket attachment fails after an entry fill.
    """
    logger.warning(f"FLATTEN {contract_id} account={account_id}")
    try:
        cancel_all_orders(client, account_id, contract_id)
    except Exception as exc:
        logger.error(f"FLATTEN  cancel_all_orders failed: {exc}")
    try:
        client.ensure_authenticated()
        client._post("/api/Position/closeContract", {
            "accountId":  account_id,
            "contractId": contract_id,
        })
        logger.warning(f"FLATTEN  position closed for {contract_id}")
    except Exception as exc:
        logger.error(f"FLATTEN  closeContract failed: {exc} — MANUAL INTERVENTION REQUIRED")


def get_open_positions(
    client:     ProjectXClient,
    account_id: int,
) -> list[dict]:
    client.ensure_authenticated()
    data = client._post("/api/Position/search", {"accountId": account_id})
    return data.get("positions", []) or []


def get_net_position(
    client:      ProjectXClient,
    account_id:  int,
    contract_id: str,
) -> int:
    """Return net size of open position for contract (positive=long, negative=short, 0=flat)."""
    positions = get_open_positions(client, account_id)
    for p in positions:
        if p.get("contractId") == contract_id:
            return int(p.get("netSize", 0))
    return 0
