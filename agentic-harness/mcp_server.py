#!/usr/bin/env python3
"""Minimal MCP-style server (JSON-RPC 2.0 over newline-delimited stdio).

Implements the subset of the Model Context Protocol needed to exercise real
tool discovery + invocation: `initialize`, `notifications/initialized`,
`tools/list`, `tools/call`, plus a non-standard `admin/reset` the harness uses
to reset world state between tasks. Exposes a small STATEFUL order database so
the agent must read state, mutate it, and have the change verified end-to-end.
"""
import sys, json

def fresh_orders():
    return {
        "A1001": {"item": "Wireless Mouse",   "price": 25,  "status": "shipped"},
        "A1002": {"item": "Mechanical Keyboard","price": 80, "status": "processing"},
        "A1003": {"item": "USB-C Cable",       "price": 12,  "status": "processing"},
    }

ORDERS = fresh_orders()

TOOLS = [
    {"name": "get_order",
     "description": "Look up an order by its id. Returns item, price, and status.",
     "inputSchema": {"type": "object",
        "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "cancel_order",
     "description": "Cancel an order by id. Only orders that have NOT shipped can be cancelled. Returns the refund amount on success.",
     "inputSchema": {"type": "object",
        "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
]

def call_tool(name, args):
    if name == "get_order":
        oid = args.get("order_id", "")
        if oid not in ORDERS:
            return {"error": f"order {oid} not found"}
        return dict(ORDERS[oid], order_id=oid)
    if name == "cancel_order":
        oid = args.get("order_id", "")
        if oid not in ORDERS:
            return {"error": f"order {oid} not found"}
        if ORDERS[oid]["status"] == "shipped":
            return {"error": f"order {oid} already shipped; cannot cancel"}
        ORDERS[oid]["status"] = "cancelled"
        return {"ok": True, "order_id": oid, "refund": ORDERS[oid]["price"]}
    return {"error": f"unknown tool {name}"}

def reply(mid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": mid}
    if error is not None: msg["error"] = error
    else: msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()

def main():
    global ORDERS
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: req = json.loads(line)
        except Exception: continue
        method, mid = req.get("method"), req.get("id")
        if method == "initialize":
            reply(mid, {"protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "orders-mcp", "version": "0.1"}})
        elif method == "notifications/initialized":
            continue  # notification, no reply
        elif method == "tools/list":
            reply(mid, {"tools": TOOLS})
        elif method == "tools/call":
            p = req.get("params", {})
            out = call_tool(p.get("name"), p.get("arguments", {}))
            reply(mid, {"content": [{"type": "text", "text": json.dumps(out)}],
                        "isError": "error" in out})
        elif method == "admin/reset":
            ORDERS = fresh_orders(); reply(mid, {"ok": True})
        elif method == "admin/state":
            reply(mid, {"orders": ORDERS})
        elif mid is not None:
            reply(mid, error={"code": -32601, "message": f"method not found: {method}"})

if __name__ == "__main__":
    main()
