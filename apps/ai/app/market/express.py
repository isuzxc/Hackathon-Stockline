"""
« Marché express »: the morning market, run DURING a customer call, for one product the caller wants and we lack.

    source()           what the voice agent calls (tool source_product). Everything is decided in code, at once:
                       stock → shortfall → suppliers who can supply it → their terms → the negotiation (same rules as
                       the morning market: opening price, counters against the best rival, floors, ≤ 2 rounds) →
                       allocate(). Typical cost: a few local ERP reads (~0.1 s, supplier profiles cached), no model.
    ExpressRound       the same negotiation, then replayed for humans on the « Marché du matin » page (the supplier
                       voices are phrased by a fast model in the background): the numbers are the script's, never
                       re-decided, so what the caller was told is exactly what the page shows.
    place_orders()     what the voice agent calls after the caller's « oui » (tool create_supplier_orders): one draft
                       purchase order per supplier kept, idempotent per sourcing id.

Nothing here blocks the phone turn: the replay runs as a background task.
"""

import asyncio
import math
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any

from app.agent.prompts import get_meta
from app.agent.speak_fr import say_date, say_price
from app.tools import ErpClient

from . import voices, live_agents
from .round import MAX_COUNTERS, ROUNDS, MarketRound, arrival, fetch_profile
from . import levers
from .levers import LABELS, LeverMemory, available, buyer_line, sensitivity
from .strategy import Decision, NeedLine, Offer, SupplierProfile, allocate, counter, counter_lever, supplier_terms
from .voices import eur, of, qty, short, the

PROFILE_TTL_S = 300
EXPRESS_NOTE = "Marché express"
SAY_UNIT = {"kg": "kilo", "piece": "pièce", "box": "carton", "liter": "litre", "dozen": "douzaine"}

_profiles: dict[str, tuple[float, SupplierProfile]] = {}


async def cached_profile(erp: ErpClient, supplier_id: str) -> SupplierProfile:
    hit = _profiles.get(supplier_id)
    if hit and time.monotonic() - hit[0] < PROFILE_TTL_S:
        return hit[1]
    p = await fetch_profile(erp, supplier_id)
    _profiles[supplier_id] = (time.monotonic(), p)
    return p


# ── the negotiation, in code ──


@dataclass
class Step:
    """One exchange of one call, decided before anyone speaks."""

    kind: str  # "quote" | "counter"
    offers: list[Offer]
    target: str | None = None  # product pushed on (counter)
    rival: Offer | None = None  # the competitor's price used against this supplier
    before: float | None = None  # target price before the counter
    after: float | None = None
    lever: str | None = None  # the argument the buyer used (market/levers.py): « concurrence », « volume »…


@dataclass
class Script:
    steps: dict[str, list[Step]]  # supplier id → its exchanges, in order
    final: list[Offer]
    decision: Decision


def negotiate(lines: list[NeedLine], profiles: dict[str, SupplierProfile], seed: str, urgent: bool = False,
              offered: dict[str, set[str]] | None = None, synchronized: bool = False,
              memory: "LeverMemory | None" = None) -> Script:
    """The morning market's negotiation without the waiting: every supplier quotes, then (≤ MAX_COUNTERS rounds) each
    one is pushed on its biggest line with the best rival price, and answers with strategy.counter().
    urgent: a customer's shortfall (a few kilos, not Christmas volumes): any supplier of that family can cover it
    alone, so the best offer takes it all (one delivery, one purchase order) instead of three small ones.
    offered: supplier id → the product ids it is asked for (the morning market: only its families); default all.
    synchronized: every round, all suppliers answer from the SAME snapshot of the board (fair, order-independent);
    default: one after the other, each seeing the counters made just before it (the original behaviour).
    Early stop: an offer already at its floor (fixed price, or floor reached) is final; a supplier whose offers are all
    final is not pushed again.
    memory (market/levers.py): each counter uses the lever this supplier responds to best so far (UCB: each one tried
    once, then mostly the best), never the same lever twice in one call; the caller then calls memory.learn(script).
    Without memory: the competitor's price only, as before."""
    terms = {(sid, l.product_id): supplier_terms(p, l, seed) for sid, p in profiles.items() for l in lines}
    if urgent:
        for (sid, pid), t in terms.items():
            line = next(l for l in lines if l.product_id == pid)
            terms[sid, pid] = replace(t, capacity=line.need)
    board: dict[tuple[str, str], Offer] = {}
    steps: dict[str, list[Step]] = {sid: [] for sid in profiles}
    for sid in profiles:
        mine_lines = [l for l in lines if offered is None or l.product_id in offered.get(sid, set())]
        quote = [Offer(sid, l.product_id, terms[sid, l.product_id].opening, terms[sid, l.product_id].capacity,
                       final=terms[sid, l.product_id].opening <= terms[sid, l.product_id].floor + 0.005,
                       opening=terms[sid, l.product_id].opening) for l in mine_lines]  # fmt: skip
        quote = [o for o in quote if o.qty > 0]
        for o in quote:
            board[sid, o.product_id] = o
        steps[sid].append(Step("quote", quote))

    def best_other(sid: str, pid: str, seen: dict) -> Offer | None:
        others = [o for (s, p), o in seen.items() if p == pid and s != sid and o.qty > 0]
        return min(others, key=lambda o: o.price, default=None)

    sens = {sid: sensitivity(sid, p.persona, p.fixed_price) for sid, p in profiles.items()} if memory else {}
    floors: dict[tuple[str, str], float] = {}  # floors lowered by a lever that worked (they stay low for this call)
    used: dict[str, set[str]] = {sid: set() for sid in profiles}
    done: set[str] = {sid for sid in profiles if all(o.final for o in (o for (s, _p), o in board.items() if s == sid))}
    for _ in range(MAX_COUNTERS):
        snapshot = dict(board)  # synchronized: everyone answers the same market
        updates: dict[tuple[str, str], Offer] = {}
        for sid in sorted(profiles):
            seen = snapshot if synchronized else board
            mine = [o for (s, _p), o in board.items() if s == sid]
            if sid in done or not mine:
                continue
            target = max(mine, key=lambda o: o.qty * o.price)
            rival = best_other(sid, target.product_id, seen)
            rival = rival if rival and rival.price < target.price else None
            lever = None
            if memory is not None:
                lever = memory.choose(sid, tuple(lv for lv in available(urgent, rival is not None) if lv not in used[sid]))
                used[sid].add(lever) if lever else None
            new = []
            for o in mine:
                key = (sid, o.product_id)
                t = replace(terms[key], floor=floors.get(key, terms[key].floor))
                on_target = o.product_id == target.product_id
                if lever and on_target:
                    price, final, floors[key] = counter_lever(t, o.price, sens[sid][lever], rival.price if lever == "concurrence" and rival else None)
                else:
                    comp = best_other(sid, o.product_id, seen) if on_target and memory is None else None
                    price, final = counter(t, o.price, comp.price if comp else None)
                new.append(Offer(sid, o.product_id, price, o.qty, final, opening=o.opening))
            t_new = next(o for o in new if o.product_id == target.product_id)
            for o in new:
                (updates if synchronized else board)[sid, o.product_id] = o
            shown_rival = rival if (lever == "concurrence" or memory is None) else None
            steps[sid].append(Step("counter", new, target.product_id, shown_rival, target.price, t_new.price,
                                   lever=lever or ("concurrence" if rival else None)))
            if all(o.final for o in new) or (memory is None and t_new.price >= target.price):
                done.add(sid)
        board.update(updates)
        if len(done) == len(profiles):
            break
    final = list(board.values())
    return Script(steps, final, allocate(lines, final, profiles))


# ── what the voice agent calls ──


@dataclass
class Sourcing:
    id: str
    client_id: str | None
    line: NeedLine
    profiles: dict[str, SupplierProfile]
    script: Script
    arrivals: dict[str, str]  # supplier id → ISO date the goods reach us
    round_id: str | None = None
    orders: list[dict] = field(default_factory=list)  # purchase orders placed (after the caller's yes)
    available: float = 0.0  # our own stock when the caller asked: the producers are needed beyond this
    mode: str = "legacy"
    quotes: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    placed_suppliers: set[str] = field(default_factory=set)


SOURCINGS: dict[str, Sourcing] = {}


def _say_qty(n: float, unit: str) -> str:
    from app.agent.speak_fr import say_qty

    return say_qty(n, {"piece": "piece", "box": "carton", "liter": "litre", "dozen": "douzaine"}.get(unit, unit))


# A product's « Rupture en cours » context entry that says the shortage is the trade's, not just ours: no producer has
# any (« le millésime 2026 ne sort pas avant janvier », « chèvres taries : plus de chevrotin jusqu'à mi-mars »).
SEASONAL = re.compile(r"pas avant|jusqu'(?:à|au|en)|ne sort|taries|fin des meules|fin de saison|hors saison|épuisé", re.I)
SUBSTITUTE = re.compile(r"Substitut possible[^:]*:\s*([^.\n]+)", re.I)


def supply_blocked(ctx: dict) -> tuple[str, str | None] | None:
    """(reason as said on the phone, substitute or None) when the product's context says nobody can supply it now."""
    for e in ctx.get("entries") or []:
        body = e.get("body") or ""
        if "rupture" not in (e.get("title") or "").lower() or not SEASONAL.search(body):
            continue
        reason = re.sub(r"^À zéro depuis [^:]*:\s*", "", body.split("Substitut")[0]).strip().rstrip(".")
        m = SUBSTITUTE.search(body)
        return reason, (m.group(1).strip() if m else None)
    return None


def _say_about(n: float, unit: str) -> str:
    """Stock as said on the phone: 9,6 kg → « à peu près neuf kilos et demi » (nearest half), exact counts unchanged."""
    if unit not in ("kg", "liter"):
        return _say_qty(n, unit)
    half = round(n * 2) / 2 or 0.5
    return _say_qty(half, unit) if abs(half - n) < 1e-9 else f"à peu près {_say_qty(half, unit)}"


async def source(erp: ErpClient, product_id: str, wanted: float, client_id: str | None = None,
                 delivery_date: str | None = None, replay: bool = True) -> dict[str, Any]:  # fmt: skip
    """The in-call buying round. Returns what the voice agent needs to answer in one sentence."""
    meta = await get_meta()
    today = date.fromisoformat(meta.get("today") or date.today().isoformat())
    deliver_by = date.fromisoformat(delivery_date) if delivery_date else today + timedelta(days=1)
    src, ctx = await asyncio.gather(
        erp.get("/api/purchase-orders/sources", {"productId": product_id}),
        erp.get("/api/context/for", {"entityType": "product", "id": product_id}),
    )
    available = float(src.get("available") or 0)
    shortfall = round(wanted - max(available, 0), 3)
    unit = src["unit"]
    if shortfall <= 0:
        return {"needed": False, "available": available, "say": f"Il y en a assez en stock : {_say_qty(available, unit)}."}
    if blocked := supply_blocked(ctx):  # the whole trade is out (season, end of a vintage): nobody to call
        reason, substitute = blocked
        out = {"found": False, "available": available, "shortfall": shortfall, "reason": reason,
               "say": f"Aucun producteur ne peut en fournir en ce moment : {reason}."}  # fmt: skip
        if substitute:
            out["substitute"] = substitute
            out["next"] = "Dis-le au client et propose le remplacement seulement s'il l'accepte (ses préférences)."
        return out
    # buy whole units, or half-kilos / half-litres (a producer does not cut 10,4 kg): always round UP
    shortfall = math.ceil(shortfall * 2) / 2 if unit in ("kg", "liter") else float(math.ceil(shortfall - 1e-9))
    suppliers = src.get("suppliers") or []
    if not suppliers:
        return {"found": False, "available": available, "shortfall": shortfall,
                "say": f"Aucun producteur ne propose {the(src['name'])} : je ne peux pas le faire venir."}  # fmt: skip
    profiles_all = dict(zip([s["supplierId"] for s in suppliers],
                            await asyncio.gather(*(cached_profile(erp, s["supplierId"]) for s in suppliers)), strict=True))  # fmt: skip
    in_time = {sid: p for sid, p in profiles_all.items() if arrival(today, p) <= deliver_by}
    late = not in_time  # nobody in time: negotiate with everyone, the caller decides on the later date
    negotiation_mode = live_agents.mode()
    profiles = profiles_all if negotiation_mode == "external" else (in_time or profiles_all)
    line = NeedLine(product_id, src["name"], unit, src["category"], src["cost"], shortfall,
                    {s["supplierId"]: s["price"] for s in suppliers if "price" in s and s["supplierId"] in profiles})  # fmt: skip
    sid_ = uuid.uuid4().hex[:8]
    seed = f"express|{product_id}|{today}"
    quotes = {}
    if negotiation_mode == "legacy":
        script = negotiate([line], profiles, seed=seed, urgent=True, memory=levers.MEMORY)
        levers.MEMORY.learn(script)
        arrivals = {sid: arrival(today, p).isoformat() for sid, p in profiles.items()}
    else:
        agents, usable, decision = await live_agents.run(
            line, profiles, today, deliver_by, seed, external=negotiation_mode == "external",
            endpoints=live_agents.external_endpoints() if negotiation_mode == "external" else None)
        profiles = {sid: profiles[sid] for sid in agents}
        quotes = {q.offer.supplier_id: q for q in usable}
        arrivals = {sid: q.arrives.isoformat() for sid, q in quotes.items()}
        steps = {}
        for sid, agent in agents.items():
            steps[sid] = []
            previous = None
            for exchange in agent.history:
                if exchange.quote is None:
                    steps[sid].append(Step("quote", []))
                    continue
                o = exchange.quote.offer
                if previous is None:
                    steps[sid].append(Step("quote", [o]))
                else:
                    steps[sid].append(Step("counter", [o], product_id, exchange.rival, previous.price, o.price,
                                           exchange.tactic if exchange.tactic in LABELS else None))
                previous = o
        script = Script(steps, [q.offer for q in usable], decision)
        late = any(q.arrives > deliver_by for q in usable)
        if negotiation_mode == "demo":
            levers.MEMORY.learn(script)
    d = script.decision
    sourcing = Sourcing(sid_, client_id, line, profiles, script, arrivals, available=max(available, 0),
                        mode=negotiation_mode, quotes=quotes)
    SOURCINGS[sid_] = sourcing
    for old in list(SOURCINGS)[:-20]:
        SOURCINGS.pop(old, None)

    kept = []
    for a in d.allocations:
        p = profiles[a.supplier_id]
        kept.append({
            "supplier": p.name, "qty": a.qty, "unitPrice": a.price, "openingPrice": a.opening,
            "arrives": arrivals[a.supplier_id], "arrivesSay": say_date(arrivals[a.supplier_id], today.isoformat()),
            "say": f"{p.name} : {_say_qty(a.qty, unit)} à {say_price(a.price, per=SAY_UNIT.get(unit, unit))}, arrivée {say_date(arrivals[a.supplier_id], today.isoformat())}",  # fmt: skip
        })
    covered = round(sum(a.qty for a in d.allocations), 3)
    latest = max((arrivals[a.supplier_id] for a in d.allocations), default=None)
    have = f"Il me reste {_say_about(available, unit)}" if available > 0 else "Je n'en ai plus en stock"
    if kept:
        who = kept[0]["supplier"] if len(kept) == 1 else f"{len(kept)} producteurs"
        say = (f"{have} ; {who} peu{'t' if len(kept) == 1 else 'vent'} m'en livrer {_say_qty(covered, unit)} "
               f"{say_date(latest, today.isoformat())}")  # fmt: skip
        if late:
            say += f", donc pour une livraison chez vous {say_date((date.fromisoformat(latest) + timedelta(days=1)).isoformat(), today.isoformat())} au plus tôt"  # fmt: skip
        if covered + 1e-9 < shortfall:
            say += f", mais pas plus : il manquerait encore {_say_qty(round(shortfall - covered, 3), unit)}"
    else:
        say = f"{have}, et aucun producteur ne peut m'en fournir pour l'instant"
    result = {
        "sourcingId": sid_, "product": short(src["name"]), "unit": unit, "wanted": wanted, "available": available,
        "shortfall": shortfall, "covered": covered, "uncovered": max(0.0, round(shortfall - covered, 3)),
        "inTime": not late, "deliverBy": deliver_by.isoformat(), "offers": kept,
        "negotiated": {"saved": d.savings, "suppliersCalled": len(profiles), "mode": negotiation_mode},
        "say": say + ".",
        "next": "Propose-le au client en une phrase. Après son oui : commande la quantité totale au client, puis create_supplier_orders(sourcingId).",  # fmt: skip
    }
    if replay and profiles and negotiation_mode != "external":
        r = ExpressRound(sourcing, erp=erp, title=f"Marché express : {short(src['name'])}", need_by=latest or deliver_by.isoformat())
        ROUNDS[r.id] = r
        for old in list(ROUNDS)[:-5]:
            ROUNDS.pop(old, None)
        sourcing.round_id = r.id
        asyncio.get_running_loop().create_task(r.run())
        result["roundId"] = r.id
    return result


async def place_orders(erp: ErpClient, sourcing_id: str) -> dict[str, Any]:
    """After the caller's yes: one draft purchase order per supplier kept (negotiated prices, expected = arrival)."""
    s = SOURCINGS.get(sourcing_id)
    if not s:
        raise ValueError("Marché express introuvable (trop ancien ou service redémarré) : relance source_product.")
    async with s.lock:
        return await _place_orders(erp, s)


async def _place_orders(erp: ErpClient, s: Sourcing) -> dict[str, Any]:
    if s.orders and len(s.placed_suppliers) == len({a.supplier_id for a in s.script.decision.allocations}):
        return {"orders": s.orders, "alreadyPlaced": True}
    if not s.script.decision.allocations:
        raise ValueError("Ce marché express n'a trouvé aucun producteur : rien à commander.")
    by_supplier: dict[str, list] = {}
    for a in s.script.decision.allocations:
        by_supplier.setdefault(a.supplier_id, []).append(a)
    for sid, allocs in by_supplier.items():
        if sid in s.placed_suppliers:
            continue
        if s.mode == "external":
            from datetime import datetime, timezone
            if s.quotes[sid].expires <= datetime.now(timezone.utc):
                raise ValueError("Offre fournisseur expirée : relance source_product.")
        body = {"supplierId": sid, "expectedDate": s.arrivals[sid],
                "note": f"{EXPRESS_NOTE} ({s.client_id or 'appel client'}) : prix négociés par l'agent achats",
                "lines": [{"productId": a.product_id, "qty": a.qty, "unitPrice": a.price} for a in allocs]}  # fmt: skip
        if s.mode == "external":
            body["note"] += f" (devis externe {s.quotes[sid].quote_id}; à confirmer auprès du fournisseur)"
        po = await erp.post("/api/purchase-orders", body)
        s.placed_suppliers.add(sid)
        s.orders.append({"number": po.get("number"), "supplier": s.profiles[sid].name, "totalHt": po.get("totalHt")})
        r = ROUNDS.get(s.round_id or "")
        if r:
            await r.emit("po_created", call=sid, poId=po["id"], number=po.get("number"), total=po.get("totalHt"))
    return {"orders": s.orders, "say": "C'est commandé chez " + " et ".join(o["supplier"] for o in s.orders) + "."}


async def place_for_order(erp: ErpClient, order: dict) -> list[dict]:
    """The client order was just written with a product sourced in this call, for more than our stock: the caller's
    yes to « je vous réserve le tout ? » is this order, so the producers' orders are placed now, in code (mar 03:43,
    s14: the model wrote the client order and forgot create_supplier_orders). Idempotent; [] when nothing to place."""
    if order.get("status") != "draft":
        return []
    qty = {line.get("productId"): float(line.get("qty") or 0) for line in order.get("lines") or []}
    placed, seen = [], set()
    for s in reversed(list(SOURCINGS.values())):  # latest sourcing per product
        pid = s.line.product_id
        if pid in seen or pid not in qty or (s.client_id and s.client_id != order.get("clientId")):
            continue
        seen.add(pid)
        if s.orders or not s.script.decision.allocations or qty[pid] <= s.available + 1e-9:
            continue
        if s.mode == "external" and (s.client_id != order.get("clientId") or abs(qty[pid] - s.available - s.line.need) > 0.501 or s.script.decision.uncovered):
            continue
        placed += (await place_orders(erp, s.id))["orders"]
    return placed


# ── the replay for the « Marché du matin » page ──


class ExpressRound(MarketRound):
    """The script of one express sourcing, told as the morning market tells it (same events, same page)."""

    def __init__(self, sourcing: Sourcing, erp: ErpClient, title: str, need_by: str, pace: float = 1.0):
        super().__init__(erp=erp, pace=pace, seed=sourcing.id)
        self.sourcing, self.title, self.need_by = sourcing, title, need_by

    async def _run(self) -> None:
        s, line = self.sourcing, self.sourcing.line
        await self.emit(
            "round_started", title=self.title,
            reference=f"appel client {s.client_id or ''} : manque {qty(line.need, line.unit)} {of(line.name)}".strip(),
            needBy=self.need_by,
            needs=[{"productId": line.product_id, "name": short(line.name), "unit": line.unit, "category": line.category, "need": line.need, "listPrice": line.reference_price, "suppliers": sorted(s.profiles)}],  # fmt: skip
            suppliers=[{"id": p.id, "name": p.name, "called": True, "contact": p.contact, "reliability": p.reliability, "flexibility": p.flexibility} for p in s.profiles.values()],  # fmt: skip
        )
        waiting = asyncio.Event()
        reached = 0

        async def call(p: SupplierProfile) -> None:
            nonlocal reached
            await self.tell(p, s.script.steps[p.id])
            reached += 1
            if reached == len(s.profiles):
                waiting.set()
            await self.decided.wait()
            await self.close(p)

        async def decide() -> None:
            await waiting.wait()
            d = s.script.decision
            self.decision = d
            await self.emit("decision", allocations=[{"supplierId": a.supplier_id, "productId": a.product_id, "qty": a.qty, "price": a.price} for a in d.allocations],  # fmt: skip
                            uncovered=d.uncovered, droppedMinOrder=d.dropped_min_order, total=d.total, reference=d.reference,
                            savings=d.savings, needed=d.needed, covered=d.covered)  # fmt: skip
            self.decided.set()

        await asyncio.gather(*(call(p) for p in s.profiles.values()), decide())
        d = s.script.decision
        pct = round(100 * d.savings / d.reference, 1) if d.reference else 0
        coverage = int(100 * d.covered / d.needed) if d.needed else 0
        await self.emit("summary", text=f"Marché express bouclé pendant l'appel : {len({a.supplier_id for a in d.allocations})} producteur(s) retenu(s) sur {len(s.profiles)}, {eur(d.total)}, {eur(d.savings)} gagnés ({str(pct).replace('.', ',')} %). Bons de commande passés au oui du client.",  # fmt: skip
                        total=d.total, reference=d.reference, savings=d.savings, savingsPct=pct,
                        suppliers=len({a.supplier_id for a in d.allocations}), called=len(s.profiles), coverage=coverage, coverageRegular=0)  # fmt: skip

    async def tell(self, p: SupplierProfile, steps: list) -> None:
        line, cid = self.sourcing.line, p.id
        first, who = (p.contact.split() or [p.name])[0], f"{p.contact or 'le commercial'} de {p.name}"
        await self.emit("call_ringing", call=cid)
        await self.pause(base=0.2 + self.rng.uniform(0, 0.8))
        await self.emit("call_connected", call=cid)
        await self.say(cid, "supplier", await voices.phrase(who, p.persona, "Tu décroches : salue l'agent achats de la Crèmerie des Aravis, en une phrase courte.", [], f"{p.name}, {first} à l'appareil !"))  # fmt: skip
        await self.say(cid, "buyer", f"Bonjour {first}, c'est urgent : un client me demande {qty(line.need, line.unit)} {of(line.name)}. Vous me les livrez quand, et à quel prix ?")  # fmt: skip
        for st in steps:
            if st.kind == "quote":
                if not st.offers:
                    await self.say(cid, "supplier", "Désolé, je n'en ai plus pour l'instant.")
                    continue
                o = st.offers[0]
                q = f"{qty(o.qty, line.unit)} à {eur(o.price)}"
                await self.say(cid, "supplier", await voices.phrase(who, p.persona, f"Tu donnes ton offre : {q}, livré en J+{p.lead_time_days}.", [eur(o.price)], f"Je peux te faire {q}, livré en J+{p.lead_time_days}."))  # fmt: skip
                await self.emit("offer", call=cid, offers=[self._express_offer(o, p)], final=False)
                continue
            product = the(line.name)
            if st.rival:  # the rival's price is an argument, its name stays confidential
                await self.emit("intel", call=cid, fromCall=st.rival.supplier_id, productId=st.target, price=st.rival.price)
            if st.lever:
                await self.emit("lever", call=cid, lever=st.lever, label=LABELS[st.lever], productId=st.target)
            first_counter = st is next(x for x in steps if x.kind == "counter")
            await self.say(cid, "buyer", buyer_line(st.lever, product, eur(st.rival.price) if st.rival else None, first_counter, urgent=True))
            if st.after < st.before:
                why = f", en échange de ce que propose l'acheteur ({LABELS[st.lever]})" if st.lever and st.lever != "concurrence" else ""
                text = await voices.phrase(who, p.persona, f"Tu ACCEPTES de baisser {product} à {eur(st.after)}{why}", [eur(st.after)], f"Allez, {eur(st.after)} pour toi.", forbid=voices.REFUSAL)  # fmt: skip
            else:
                text = await voices.phrase(who, p.persona, f"Tu refuses de baisser : {eur(st.after)}, c'est ton dernier prix.", [eur(st.after)], f"Non, {eur(st.after)}, c'est mon dernier prix.")  # fmt: skip
            await self.say(cid, "supplier", text)
            final = st is steps[-1]
            await self.emit("offer", call=cid, offers=[self._express_offer(o, p) for o in st.offers], final=final)
        await self.emit("call_waiting", call=cid)

    async def close(self, p: SupplierProfile) -> None:
        cid, line = p.id, self.sourcing.line
        won = [a for a in self.sourcing.script.decision.allocations if a.supplier_id == cid]
        if won:
            what = " ; ".join(f"{qty(a.qty, line.unit)} à {eur(a.price)}" for a in won)
            await self.say(cid, "buyer", f"C'est bon : {what}. Je confirme dès que le client dit oui.")
        else:
            await self.say(cid, "buyer", "Merci, cette fois on part ailleurs.")
        await self.emit("call_ended", call=cid, won=bool(won), total=round(sum(a.qty * a.price for a in won), 2))

    def _express_offer(self, o: Offer, p: SupplierProfile) -> dict:
        return {"productId": o.product_id, "price": o.price, "qty": o.qty, "final": o.final, "regular": p.id in self.sourcing.line.list_prices}  # fmt: skip
