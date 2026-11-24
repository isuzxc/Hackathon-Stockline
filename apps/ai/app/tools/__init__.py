"""
The tools the model can call, grouped by domain. Each one wraps ERP API endpoints.

    registry.py  @tool decorator, Tool, ToolContext (start here to add a tool)
    erp.py       the HTTP client for the ERP API
    catalog.py   search, get_client, get_product_stock, find_alternatives, get_price
    context.py   search_context, get_context, save_context
    history.py   get_client_history, log_client_event, get_supplier_history, log_supplier_event
    orders.py    create_draft_order, add_order_line, update_order_line, remove_order_line, update_order,
                 cancel_order, confirm_order, get_order
    voice.py     identify_caller, find_product (composite tools for the phone: one call instead of 3–4)
    market.py    source_product, create_supplier_orders (« marché express » during a call: app/market/express.py)

A new file needs one more line: import it below so its @tool functions get registered.
"""

from . import catalog, context, history, market, negotiation, orders, voice  # noqa: F401  (importing registers the tools)
from .erp import ErpClient
from .registry import TOOLS, Tool, ToolContext, tool


def all_tools() -> list[Tool]:
    return list(TOOLS.values())


__all__ = ["TOOLS", "ErpClient", "Tool", "ToolContext", "all_tools", "tool"]
