"""Quote-first Voix workflow. All external communications are non-binding until a human accepts.

The ERP supplies the retail baseline, actual inventory, client and product IDs. Supplier agents
supply procurement quotes; the buyer cannot invent either prices or availability.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from app.config import env
from app.agent.prompts import get_meta
from app.agent.speak_fr import say_date, say_price
from app.market import live_agents
from app.market.express import cached_profile, supply_blocked
from app.market.strategy import NeedLine, Offer, SupplierProfile
from app.tools.erp import enc
from app.tools.voice import next_delivery, say_line

STOCK = "__stock__"
QUOTE_TTL = 180
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def enabled():
    return env("VOIX_NEGOTIATION_FIRST", "1") == "1"


def customer_price(baseline, usual_cost, actual_cost, opening_cost, quantity):
    """Proposed commercial concession: half the savings, never below procurement cost
    or the existing nominal margin on usual cost. No unapproved increase over ERP price."""
    d = lambda x: Decimal(str(x))
    baseline, usual, actual, opening, quantity = map(d, (baseline, usual_cost, actual_cost, opening_cost, quantity))
    if quantity <= 0 or min(baseline, usual, actual, opening) < 0:
        raise ValueError("Prix ou quantité invalide.")
    floor = (actual / quantity + max(Decimal(0), baseline - usual)).quantize(Decimal(".01"), rounding=ROUND_CEILING)
    if floor > baseline:
        return None
    discount = max(Decimal(0), opening - actual) / (quantity * 2)
    target = (baseline - discount).quantize(Decimal(".01"), rounding=ROUND_FLOOR)
    return float(max(floor, target))


@dataclass
class LineQuote:
    product_id: str
    product: str
    unit: str
    qty: float
    price: float
    baseline: float
    allocations: list
    quotes: dict
    profiles: dict
    agents: int
    exchanges: int


@dataclass
class Basket:
    id: str
    client_id: str
    client_name: str
    delivery: str
    lines: list[LineQuote]
    mode: str
    expires: float = field(default_factory=lambda: time.monotonic() + QUOTE_TTL)
    say: str = ""
    state: str = "quoted"
    order: dict | None = None
    purchase_orders: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# One current offer per voice conversation. Superseded offers cannot be accepted.
BASKETS: dict[str, Basket] = {}


def pending(session_id):
    return BASKETS.get(session_id)


def approval_matches(messages, basket):
    """Only a short, explicit yes immediately after the exact system-generated recap."""
    if not basket or basket.state not in ("quoted", "accepted"):
        return False
    indices = [i for i, m in enumerate(messages) if m.role == "user"]
    if not indices:
        return False
    i = indices[-1]
    text = (messages[i].content or "").strip().lower().strip(" .!?")
    if not re.fullmatch(r"(oui|ouais|ok|okay|yes|d'accord|d’accord|je confirme|je valide|c'est bon|c’est bon|vas-y|allez-y|ça marche|go ahead)([, ]+(oui|merci|je confirme|je valide|please|s'il vous plaît))?", text):
        return False
    previous = messages[i-1] if i else None
    return bool(previous and previous.role == "assistant" and previous.content == basket.say)


async def quote_line(erp, client_id, request, today, delivery, market_mode):
    product_id, quantity = request["productId"], request["qty"]
    src, context, price = await asyncio.gather(
        erp.get("/api/purchase-orders/sources", {"productId": product_id}),
        erp.get("/api/context/for", {"entityType": "product", "id": product_id}),
        erp.get("/api/prices/resolve", {"clientId": client_id, "productId": product_id, "qty": quantity}),
    )
    unit = src["unit"]
    step = 0.5 if unit in ("kg", "liter") else 1
    if not math.isfinite(quantity) or quantity <= 0 or abs(quantity / step - round(quantity / step)) > 1e-7:
        raise ValueError(f"Quantité de {src['name']} à préciser dans l'unité {unit} (pas de {step}).")
    if blocked := supply_blocked(context):
        raise ValueError(f"{src['name']} : {blocked[0]}")
    # Suppliers must deliver before the customer's delivery day, allowing warehouse handling.
    deadline = delivery - timedelta(days=1)
    suppliers = src.get("suppliers") or []
    ids = [s["supplierId"] for s in suppliers]
    loaded = await asyncio.gather(*(cached_profile(erp, sid) for sid in ids), return_exceptions=True)
    profiles = {sid: p for sid, p in zip(ids, loaded, strict=True) if isinstance(p, SupplierProfile)}
    line = NeedLine(product_id, src["name"], unit, src["category"], float(src["cost"]), quantity,
                    {s["supplierId"]: s["price"] for s in suppliers if "price" in s})
    agents, quotes = {}, []
    if deadline >= today and profiles:
        agents, quotes, _ = await live_agents.run(
            line, profiles, today, deadline, f"voix|{product_id}|{today}",
            external=market_mode == "external",
            endpoints=live_agents.external_endpoints() if market_mode == "external" else None,
        )
    quotes = [q for q in quotes if q.arrives <= deadline and q.expires > datetime.now(timezone.utc)]
    # Existing stock is an alternative, not a fake competitor shared with shops.
    stock_qty = min(quantity, max(0.0, float(src.get("available") or 0)))
    if stock_qty >= step:
        profiles[STOCK] = SupplierProfile(STOCK, "Stock interne", reliability=3)
        quotes.append(live_agents.Quote(
            Offer(STOCK, product_id, line.cost, stock_qty, True, opening=line.cost),
            today, datetime.now(timezone.utc) + timedelta(minutes=3), "internal"))
    decision = live_agents.optimize(line, quotes, profiles, strict_min=market_mode == "external")
    if decision.covered + 1e-8 < quantity:
        raise ValueError(f"{src['name']} : seulement {decision.covered:g} {unit} disponibles à temps sur {quantity:g}.")
    baseline = float(price["price"])
    proposed = customer_price(baseline, line.cost, decision.total, decision.reference, quantity)
    if proposed is None:
        raise ValueError(f"{src['name']} : aucun tarif compatible avec le prix client et le coût vérifié.")
    return LineQuote(product_id, src["name"], unit, quantity, proposed, baseline, decision.allocations,
                     {q.offer.supplier_id: q for q in quotes}, profiles, len(agents),
                     sum(len(a.history) for a in agents.values()))


async def check_client(erp, client_id, delivery):
    client, routes = await asyncio.gather(erp.get(f"/api/clients/{enc(client_id)}"), erp.get("/api/routes"))
    if client.get("status") == "blocked":
        raise ValueError("Client bloqué : contacter la comptabilité.")
    code = WEEKDAYS[delivery.weekday()]
    if code not in (client.get("deliveryDays") or []):
        raise ValueError("Ce jour ne fait pas partie des jours de livraison du client.")
    if isinstance(routes, dict):
        routes = routes.get("items", [])
    if not any(r.get("zone") == client.get("zone") and code in (r.get("days") or []) for r in routes):
        raise ValueError("Aucune tournée vérifiée pour ce jour.")
    return client


async def quote_basket(ctx, client_id, requests, delivery_date):
    if not ctx.session_id:
        raise ValueError("Offre disponible uniquement dans une session Voix.")
    old = pending(ctx.session_id)
    if old and old.state == "processing":
        raise ValueError("Une acceptation est déjà en cours.")
    # Invalidate even when the replacement cannot be quoted.
    BASKETS.pop(ctx.session_id, None)
    for key, b in list(BASKETS.items()):
        if b.expires <= time.monotonic() and b.state != "processing":
            BASKETS.pop(key, None)
    if not 1 <= len(requests) <= 8 or len({r["productId"] for r in requests}) != len(requests):
        raise ValueError("Donnez de un à huit produits distincts.")
    today = date.fromisoformat((await get_meta())["today"])
    client = await ctx.erp.get(f"/api/clients/{enc(client_id)}")
    delivery = date.fromisoformat(delivery_date or next_delivery(today.isoformat(), client.get("deliveryDays") or []))
    if delivery <= today:
        raise ValueError("Choisissez un jour de livraison futur.")
    client = await check_client(ctx.erp, client_id, delivery)
    market_mode = live_agents.mode()
    if market_mode == "legacy":
        raise ValueError("Le mode négociation Voix nécessite MARKET_NEGOTIATION_MODE=demo ou external.")
    results = await asyncio.gather(*(quote_line(ctx.erp, client_id, r, today, delivery, market_mode)
                                     for r in requests), return_exceptions=True)
    problems = [str(r) for r in results if isinstance(r, Exception)]
    if problems:
        return {"found": False, "problems": problems, "mode": market_mode}
    basket = Basket(uuid.uuid4().hex, client_id, client["name"], delivery.isoformat(), results, market_mode)
    total = round(sum(l.qty * l.price for l in basket.lines), 2)
    savings = round(sum(l.qty * (l.baseline - l.price) for l in basket.lines), 2)
    credit = float(client.get("creditLimit") or 0)
    if credit and float(client.get("outstanding") or 0) + total > credit:
        return {"found": False, "problems": ["Plafond d'encours : validation comptable nécessaire."]}
    detail = " ; ".join(f"{say_line(l.qty, l.unit, l.product)} à {say_price(l.price)} l'unité" for l in basket.lines)
    saved = f", soit {say_price(savings)} d'économie" if savings > 0 else ""
    caveat = "Simulation : " if market_mode == "demo" else ""
    basket.say = (f"{caveat}{basket.client_name}, {detail}, total {say_price(total)} hors taxes{saved}, "
                  f"livraison prévue {say_date(basket.delivery, today.isoformat())}. "
                  "Acceptez-vous cette offre ?")
    BASKETS[ctx.session_id] = basket
    return {"found": True, "quoteId": basket.id, "mode": basket.mode, "deliveryDate": basket.delivery,
            "lines": [{"product": l.product, "qty": l.qty, "unit": l.unit, "unitPrice": l.price,
                       "total": round(l.qty * l.price, 2)} for l in basket.lines],
            "total": total, "savings": savings, "agents": sum(l.agents for l in basket.lines),
            "exchanges": sum(l.exchanges for l in basket.lines), "say": basket.say}


async def accept_basket(ctx, quote_id):
    basket = pending(ctx.session_id)
    if not basket or basket.id != quote_id:
        raise ValueError("Offre introuvable ou remplacée. Demandez une nouvelle offre.")
    async with basket.lock:
        if basket.state == "accepted":
            return {"accepted": True, "alreadyAccepted": True, "say": "Votre demande est déjà enregistrée."}
        if basket.state != "quoted" or basket.expires <= time.monotonic():
            raise ValueError("Offre expirée ou traitement à vérifier. Ne recréez pas la commande.")
        client = await check_client(ctx.erp, basket.client_id, date.fromisoformat(basket.delivery))
        total = round(sum(l.qty * l.price for l in basket.lines), 2)
        credit = float(client.get("creditLimit") or 0)
        if credit and float(client.get("outstanding") or 0) + total > credit:
            raise ValueError("Plafond d'encours : validation comptable nécessaire.")
        grouped = {}
        for line in basket.lines:
            current = await ctx.erp.get("/api/purchase-orders/sources", {"productId": line.product_id})
            for alloc in line.allocations:
                if alloc.supplier_id == STOCK:
                    if float(current.get("available") or 0) + 1e-8 < alloc.qty:
                        raise ValueError("Le stock a changé. Demandez une nouvelle offre.")
                    continue
                q = line.quotes[alloc.supplier_id]
                if q.expires <= datetime.now(timezone.utc):
                    raise ValueError("Devis fournisseur expiré. Demandez une nouvelle offre.")
                group = grouped.setdefault(alloc.supplier_id, {"lines": [], "arrivals": [], "quotes": []})
                group["lines"].append({"productId": line.product_id, "qty": alloc.qty, "unitPrice": alloc.price})
                group["arrivals"].append(q.arrives.isoformat())
                group["quotes"].append(q.quote_id)
        # On an ambiguous ERP/network failure, never retry automatically: a write may already exist.
        basket.state = "processing"
        try:
            basket.order = await ctx.erp.post("/api/orders", {
                "clientId": basket.client_id, "deliveryDate": basket.delivery, "source": "voice_agent",
                "note": f"Voix offre acceptée {basket.id} ({basket.mode})",
                "lines": [{"productId": l.product_id, "qty": l.qty, "unitPrice": l.price} for l in basket.lines],
            })
            for sid, group in grouped.items():
                po = await ctx.erp.post("/api/purchase-orders", {
                    "supplierId": sid, "expectedDate": max(group["arrivals"]), "lines": group["lines"],
                    "note": f"Voix {basket.id} ; devis {', '.join(group['quotes'])} ; à confirmer au fournisseur",
                })
                basket.purchase_orders.append(po)
        except BaseException:
            basket.state = "review"
            raise
        basket.state = "accepted"
        return {"accepted": True, "orderNumber": basket.order.get("number"),
                "supplierDrafts": len(basket.purchase_orders),
                "say": "Votre offre est acceptée et enregistrée. La commande reste en attente de validation logistique."}
