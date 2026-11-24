"""
Composite tools for the phone (GUIDE-VOIX 4.6): one tool call instead of three or four model round-trips.

    identify_caller  search + client card + usual basket + credit + route / next delivery + open orders
                     + how to talk to them (context table)                  replaces search → get_client → get_context
    find_product     search + stock + this client's price (+ usual qty, + substitutes when short)
                                                                            replaces search → get_product_stock → get_price
                                                                            → find_alternatives

Every model round-trip re-reads the whole prompt (≈ 0.5–1.5 s each on a real model), so on a phone call the
number of steps matters more than the size of one result. Values the agent will say out loud come with a `say`
field already in spoken French (agent/speak_fr.py): the model copies it instead of formatting numbers itself.
"""

import asyncio
import time

from pydantic import Field

from .erp import enc, pick
from .registry import ToolArgs, ToolContext, tool

# Search scores are 0–100 (apps/backend/app/services/search.py). A match is "sure" when it is high and clearly ahead.
SURE_SCORE = 85
CLEAR_LEAD = 10
BIG_LEAD = 15  # far ahead of the next one is sure too, even with a low score (« Jean-Marc du Panier Savoyard du Grand-Bornand »)

WEEKDAYS_FR = {"mon": "lundi", "tue": "mardi", "wed": "mercredi", "thu": "jeudi", "fri": "vendredi", "sat": "samedi", "sun": "dimanche"}
TALK_KINDS = ("relationship", "procedure", "risk", "preference")


def _speak():
    # imported lazily: app.agent imports app.tools (the agent loop), so a module-level import would be circular
    from app.agent import speak_fr

    return speak_fr


async def _today() -> str | None:
    from app.agent.prompts import get_meta

    return (await get_meta()).get("today")


_routes_cache: tuple[float, list[dict]] | None = None


async def _routes(ctx: ToolContext) -> list[dict]:
    """Delivery routes change rarely: cached 5 minutes."""
    global _routes_cache
    if _routes_cache and time.monotonic() - _routes_cache[0] < 300:
        return _routes_cache[1]
    routes = await ctx.erp.get("/api/routes")
    _routes_cache = (time.monotonic(), routes)
    return routes


def _ambiguous(results: list[dict]) -> bool:
    if not results:
        return False
    best = results[0]["score"]
    second = results[1]["score"] if len(results) > 1 else 0
    if best - second >= BIG_LEAD:
        return False
    return best < SURE_SCORE or second > best - CLEAR_LEAD


def _norm(s: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


async def _by_contact_name(ctx: ToolContext, who: str, candidates: list[dict]) -> str | None:
    """« c'est Jean-Marc du Panier Savoyard »: three Panier Savoyard, one has a contact named Jean-Marc. Among the
    candidates whose contact name was said, the best match wins: the longest name said (« Jean-Marc » beats « Marc »
    in another shop), then the search score. None when no candidate, or a tie."""
    said = f" {_norm(who).replace('-', ' ')} "
    clients = await asyncio.gather(*(ctx.erp.get(f"/api/clients/{enc(c['id'])}") for c in candidates), return_exceptions=True)
    hits: list[tuple[int, float, str]] = []  # (length of the name said, search score, client id)
    for cand, c in zip(candidates, clients, strict=True):
        if isinstance(c, BaseException):
            continue
        best = 0
        for contact in c.get("contacts") or []:
            words = _norm(contact.get("name") or "").split()
            if not words or words[0] == "service":
                continue
            for name in (" ".join(words), words[0]):  # full name, or the first name (« Jean-Marc »)
                name = name.replace("-", " ")
                if f" {name} " in said:
                    best = max(best, len(name))
        if best:
            hits.append((best, cand["score"], cand["id"]))
    hits.sort(reverse=True)
    if not hits or (len(hits) > 1 and hits[0][:2] == hits[1][:2]):
        return None
    return hits[0][2]


def short_name(name: str) -> str:
    """'Reblochon de Savoie AOP laitier (450 g)' → 'Reblochon de Savoie AOP laitier' (what a salesperson says)."""
    return name.split(" (")[0].strip()


def say_line(qty: float, unit: str, name: str) -> str:
    """(4, 'piece', 'Reblochon … (450 g)') → 'quatre Reblochon de Savoie AOP laitier';
    (2.5, 'kg', 'Abondance AOP') → "deux kilos et demi d'Abondance AOP"."""
    sp = _speak()
    n = short_name(name)
    if unit == "piece":
        return f"{sp.say_qty(qty, 'piece').rsplit(' ', 1)[0]} {n}" if float(qty).is_integer() else f"{sp.say_qty(qty, 'piece')} de {n}"
    of = "d'" if n[:1].lower() in "aeiouyéèêâîôûœh" else "de "
    return f"{sp.say_qty(qty, unit)} {of}{n}"


def say_unit_price(price: float, unit: str) -> str:
    per = {"kg": "kilo", "piece": "pièce", "box": "carton", "liter": "litre", "dozen": "douzaine"}.get(unit)
    feminine = per in ("pièce", "douzaine")
    return f"{_speak().say_price(price)} {'la' if feminine else 'le'} {per}" if per else _speak().say_price(price)


class ProductAsk(ToolArgs):
    query: str = Field(description="Le produit tel que dit, ex. « reblo fermier », « beaufort d'été »")
    qty: float | None = Field(None, description="Quantité dite, dans l'unité du produit")


# ── identify_caller ──────────────────────────────────────────────────────────────────────────────────────────────


@tool
async def identify_caller(
    ctx: ToolContext,
    who: str = Field(description="Le nom de l'établissement ou de la personne tel qu'entendu, ex. « le chalet de la balme », « Mélanie de la Balme »"),
    products: list[ProductAsk] | None = Field(
        None, description="Les produits que l'appelant a DÉJÀ cités dans la même phrase : cherchés en même temps (stock, SON prix)"
    ),
) -> dict:
    """
    À appeler EN PREMIER dès que l'appelant dit qui il est (un seul appel, au lieu de search + get_client + get_context).
    Renvoie la fiche utile au téléphone : statut (bloqué ?), encours et plafond, panier habituel, jours de livraison et prochaine
    livraison, commandes en cours (pour une modification), réclamations ouvertes, et comment lui parler (tutoiement, tics, ce qui l'agace).
    Si l'appelant cite déjà des produits (« c'est Jean-Marc, il vous reste du beaufort ? »), passe-les dans `products` :
    ils reviennent dans `products` comme find_product, sans un appel de plus.
    Si plusieurs clients sont plausibles, renvoie `candidates` : demande lequel, sans rien supposer.
    Identify the caller and load everything needed for the call in one step (+ the products already mentioned).
    """
    results = [r for r in await ctx.erp.get("/api/search", {"q": who, "types": "client"}) if r["type"] == "client"]
    if not results:
        return {"found": False, "hint": "Aucun client à ce nom : redemande le nom de l'établissement et la ville, ou traite-le comme un prospect."}
    if _ambiguous(results) and (picked := await _by_contact_name(ctx, who, results[:4])):
        results = [r for r in results if r["id"] == picked]
    if _ambiguous(results):
        return {
            "found": False,
            "candidates": [{"clientId": r["id"], "name": r["label"], "where": r.get("sublabel")} for r in results[:4]],
            "hint": "Plusieurs clients possibles : demande lequel (nom + ville), une seule question.",
        }
    client_id = results[0]["id"]
    if not products:
        return await caller_card(ctx, client_id)
    card, *found = await asyncio.gather(
        caller_card(ctx, client_id), *(lookup_product(ctx, p.query, client_id, p.qty) for p in products)
    )
    return {**card, "products": [{"asked": p.query, **f} for p, f in zip(products, found, strict=True)]}


async def caller_card(ctx: ToolContext, client_id: str) -> dict:
    client, context, routes, today = await asyncio.gather(
        ctx.erp.get(f"/api/clients/{enc(client_id)}"),
        ctx.erp.get("/api/context/for", {"entityType": "client", "id": client_id}),
        _routes(ctx),
        _today(),
    )
    sp = _speak()
    route = next((r for r in routes if r.get("zone") == client.get("zone")), None)
    days = client.get("deliveryDays") or []
    next_day = next_delivery(today, days) if today else None
    limit, outstanding = client.get("creditLimit") or 0, client.get("outstanding") or 0
    open_orders = [o for o in client.get("recentOrders") or [] if o.get("status") in ("draft", "confirmed")]
    open_events = [e for e in client.get("recentEvents") or [] if e.get("status") == "open"]
    talk = [e for e in context.get("entries") or [] if e.get("kind") in TALK_KINDS]
    about = next((e.get("body") or "" for e in context.get("entries") or [] if e.get("title") == "Qui ils sont"), "")
    usual = client.get("usual") or []
    last = None if usual else await _last_order(ctx, client, today)
    return {
        "found": True,
        "client": {
            **pick(client, "id", "name", "type", "city", "status", "preferences"),
            "contacts": [pick(c, "name", "role") for c in client.get("contacts") or []],
        },
        **({"blocked": "Client BLOQUÉ : ne confirme aucune commande, explique poliment et propose de voir avec la comptabilité."} if client.get("status") == "blocked" else {}),
        "credit": {
            "outstanding": outstanding,
            "creditLimit": limit,
            "overdue": client.get("overdue") or 0,
            **({"warning": "Encours au-delà de 80 % du plafond : à mentionner avec tact."} if limit and outstanding > 0.8 * limit else {}),
        },
        "delivery": {
            "days": [WEEKDAYS_FR.get(d, d) for d in days],
            **({"route": route["name"], "routeDays": [WEEKDAYS_FR.get(d, d) for d in route.get("days") or []], "departure": route.get("departureTime")} if route else {}),
            **({"next": next_day, "nextSay": sp.say_date(next_day, today)} if next_day else {}),
        },
        "usual": [
            {**pick(u, "productId", "productName", "unit", "avgQty", "lastQty"), "say": say_line(round(u["avgQty"]), u["unit"], u["productName"]) if round(u["avgQty"]) > 0 else None}
            for u in usual[:8]
        ],
        **({"lastOrder": last} if last else {}),
        **({"about": clip(about, 320)} if about else {}),
        **({"closedNow": closed} if (closed := closed_now(about, today)) else {}),
        "openOrders": [pick(o, "id", "number", "status", "deliveryDate", "totalHt") for o in open_orders[:3]],
        **({"openIssues": [pick(e, "id", "type", "title", "at") for e in open_events[:3]]} if open_events else {}),
        "notes": [n["text"] for n in (client.get("notes") or [])[-3:]],
        "howToTalk": [{"title": e.get("title"), "body": clip(e.get("body") or "", 450)} for e in talk[:5]],
    }


_MONTHS = {m: i + 1 for i, m in enumerate(["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                                           "septembre", "octobre", "novembre", "décembre"])}


def closed_now(about: str, today: str | None) -> str | None:
    """« Ouvert de mi-décembre à mi-avril et en juillet-août. » on 24 November → « Fermé en ce moment (ouvert de
    mi-décembre à mi-avril et en juillet-août). » (s05: the model never worked it out from the text). None if open or
    unknown."""
    import re

    m = re.search(r"\bOuvert ([^.]*)\.", about or "")
    if not m or not today:
        return None
    open_months: set[int] = set()
    for a, b in re.findall(r"(?:mi-)?(\w+)\s*(?:à|-|au)\s*(?:mi-)?(\w+)", m.group(1)):
        if a.lower() in _MONTHS and b.lower() in _MONTHS:
            first, last = _MONTHS[a.lower()], _MONTHS[b.lower()]
            open_months |= {(first - 1 + k) % 12 + 1 for k in range((last - first) % 12 + 1)}
    if not open_months or int(today[5:7]) in open_months:
        return None
    return f"Fermé en ce moment (ouvert {m.group(1).strip()})."


async def _last_order(ctx: ToolContext, client: dict, today: str | None) -> dict | None:
    """No usual basket (seasonal client, last order months ago): « même chose que la dernière fois » = this order."""
    past = [o for o in client.get("recentOrders") or [] if o.get("status") not in ("cancelled", "draft")]
    if not past:
        return None
    o = await ctx.erp.get(f"/api/orders/{enc(past[0]['id'])}")
    lines = o.get("lines") or []
    return {
        "deliveryDate": o.get("deliveryDate"),
        "dateSay": _speak().say_date(o["deliveryDate"], today) if o.get("deliveryDate") else None,
        "lines": [{**pick(line, "productId", "productName", "unit", "qty"), "say": say_line(line["qty"], line.get("unit") or "", line.get("productName") or "")} for line in lines[:10]],
    }


def next_delivery(today: str, days: list[str]) -> str:
    """Same rule as the ERP's default delivery date (apps/backend/app/services/clients.py next_delivery_date)."""
    from datetime import date, timedelta

    t = date.fromisoformat(today[:10])
    codes = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    for i in range(1, 8):
        d = t + timedelta(days=i)
        if days and codes[d.weekday()] in days:
            return d.isoformat()
    return (t + timedelta(days=1)).isoformat()


def clip(s: str, n: int) -> str:
    return f"{s[:n]}…" if len(s) > n else s


# ── find_product ─────────────────────────────────────────────────────────────────────────────────────────────────


@tool
async def find_product(
    ctx: ToolContext,
    query: str = Field(description="Le produit tel que dit par l'appelant, ex. « reblo fermier », « de la raclette au poivre », « beurre en plaquettes »"),
    client_id: str | None = Field(None, description="Le client au téléphone (cli_…) : donne SON prix et ce qu'il prend d'habitude"),
    qty: float | None = Field(None, description="Quantité demandée, dans l'unité du produit (vérifie le stock et les prix au volume)"),
    compare: bool = Field(False, description="Comparer les variantes et leurs prix sans choisir à la place du client"),
    delivery_date: str | None = Field(
        None,
        description="Date de livraison YYYY-MM-DD pour une demande de commande complète et non ambiguë"
    ),
) -> dict:
    """
    Trouve un produit à partir de ce qu'a dit l'appelant ET donne, en un seul appel : stock disponible, prix pour ce client (prix négocié,
    règles), s'il est dans son panier habituel, et des remplacements si le stock ne suffit pas. Les champs `say` sont déjà en français parlé.
    Si plusieurs produits sont plausibles (ex. reblochon fermier / laitier), renvoie `candidates` : ceux marqués `inUsual` sont ceux que
    le client prend d'habitude ; sinon demande lequel. Find a product with stock + client price + substitutes in one call.
    """
    card = await lookup_product(
        ctx, query, client_id, qty, compare
    )

    if (
        not ctx.session_id
        and compare
        and client_id
        and delivery_date
        and qty is not None
        and qty > 0
        and card.get("found")
        and card.get("product")
        and not card.get("seasonBlocked")
        and not card.get("candidates")
        and not card.get("pickedBecause")
        and not card.get("stock", {}).get("enough", False)
    ):
        from app.market.express import source

        try:
            sourcing = await source(
                ctx.erp,
                card["product"]["id"],
                qty,
                client_id,
                delivery_date,
            )
        except Exception:
            card["hint"] = (
                "Réassort non vérifié. Ne promets pas "
                "la quantité totale ni une date non vérifiée."
            )
        else:
            card["sourcing"] = sourcing
            card["hint"] = (
                "Le réassort a déjà été recherché. "
                "Ne rappelle pas source_product pour cette même demande. "
                "Vérifie la couverture totale et la livraison client. "
                "Annonce uniquement le prix de vente client."
            )

    return card


async def lookup_product(ctx: ToolContext, query: str, client_id: str | None, qty: float | None, compare: bool = False) -> dict:
    results = [r for r in await ctx.erp.get("/api/search", {"q": query, "types": "product"}) if r["type"] == "product"]
    if not results:
        return {"found": False, "hint": "Aucun produit ne correspond : demande de préciser (famille, affinage, format)."}
    usual = await ctx.erp.get(f"/api/clients/{enc(client_id)}/usual") if client_id else []
    usual_by_id = {u["productId"]: u for u in usual}

    if _ambiguous(results):
        close = [r for r in results if r["score"] >= results[0]["score"] - CLEAR_LEAD][:4]
        in_usual = [r for r in close if r["id"] in usual_by_id]
        if len(in_usual) == 1 and not compare:  # the caller's usual product wins: "le reblo" = the one they always take
            return {**await product_card(ctx, in_usual[0]["id"], client_id, qty, usual_by_id), "pickedBecause": "c'est celui que le client prend d'habitude"}
        if compare:
            cards = await asyncio.gather(
                *(product_card(
                    ctx, r["id"], client_id, qty, usual_by_id
                ) for r in close[:3]),
                return_exceptions=True,
            )
            candidates = [
                {
                    "productId": card["product"]["id"],
                    "name": card["product"]["say"],
                    "unit": card["product"]["unit"],
                    "price": card["price"],
                    "stock": card["stock"],
                }
                for card in cards
                if isinstance(card, dict)
                and card.get("product")
                and card.get("price")
            ]
            return {
                "found": False,
                "candidates": candidates,
                "hint": (
                    "Présente les options pertinentes avec leur prix "
                    "et leur unité. Vérifie la quantité et le délai. "
                    "Attends le choix explicite du client."
                ),
            }
        return {
            "found": False,
            "candidates": [
                {"productId": r["id"], "name": short_name(r["label"]), **({"inUsual": True} if r["id"] in usual_by_id else {})}
                for r in close
            ],
            "hint": "Plusieurs produits possibles : demande lequel en citant au plus deux ou trois noms courts.",
        }
    return await product_card(ctx, results[0]["id"], client_id, qty, usual_by_id)


async def product_card(ctx: ToolContext, product_id: str, client_id: str | None, qty: float | None, usual_by_id: dict) -> dict:
    query = {"clientId": client_id, "productId": product_id, "qty": _fmt(qty) if qty else None}
    product, price = await asyncio.gather(
        ctx.erp.get(f"/api/products/{enc(product_id)}"),
        ctx.erp.get("/api/prices/resolve", query) if client_id else asyncio.sleep(0, result=None),
    )
    unit, available = product["unit"], product.get("available") or 0
    stock = product.get("stock") or {}
    short = qty is not None and available < qty or available <= 0
    card: dict = {
        "found": True,
        "product": {**pick(product, "id", "name", "unit", "caliber", "origin"), "say": short_name(product["name"])},
        "stock": {
            "available": available,
            **({"incoming": stock["incoming"]} if stock.get("incoming") else {}),
            **pick(product, "nextExpiry"),
            "enough": not short,
        },
    }
    unit_price = price["price"] if price else product["price"]
    card["price"] = {
        "unitPrice": unit_price,
        "say": say_unit_price(unit_price, unit),
        "source": price["source"] if price else "list",
        **({"listPrice": price["listPrice"], "explanation": price["explanation"]} if price and price["price"] != price["listPrice"] else {}),
    }
    if price:
        volume = next((c for c in price.get("candidates") or [] if c.get("source") == "rule" and (c.get("reason") or "").startswith("à partir de") and c["price"] < price["price"]), None)
        if volume:
            card["price"]["betterWithVolume"] = f"{say_unit_price(volume['price'], unit)} {volume['reason']}"
    if product_id in usual_by_id:
        u = usual_by_id[product_id]
        card["usual"] = {"avgQty": u["avgQty"], "lastQty": u["lastQty"], "lastOrderedAt": u.get("lastOrderedAt")}
    if qty:
        card["lineSay"] = say_line(qty, unit, product["name"])
    if short:
        alternatives = await ctx.erp.get(f"/api/products/{enc(product_id)}/alternatives")
        card["alternatives"] = [
            {**pick(p, "id", "unit", "available"), "name": short_name(p["name"]), "say": say_unit_price(p["price"], p["unit"])}
            for p in alternatives[:3]
            if (p.get("available") or 0) > 0
        ]
        card["hint"] = (
            "Le stock interne ne suffit pas, mais cela ne signifie "
            "pas que la commande est impossible. "
            "Pour le mode vocal, appelle source_product avec "
            "la quantité TOTALE demandée, le client et la date "
            "de livraison souhaitée AVANT de proposer une quantité "
            "réduite. Attends le résultat. Si toute la quantité "
            "peut être livrée à temps, propose-la directement "
            "au prix client vérifié. Sinon, propose les options "
            "réellement disponibles et laisse le client choisir. "
            "Ne remplace jamais un produit sans son accord."
        )
    return card


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)
