"""
Fast paths for the phone: answer with ONE model call, or none.

A model call costs ~0.65 s to the first token (Claude Haiku, measured with the team key; the prompt size does not
matter once cached). A turn that needs a tool costs two calls in a row. So, in code, before the model:

  fast_confirm()  « oui c'est bon » right after an order recap → confirm_order in code + a templated reply, NO model
                  call. Same guard as the model path (agent/guardrails.py may_confirm), and only for a clean draft (no
                  ERP warning, client not blocked): anything else goes to the model.
  preload()       the caller and the products named in the sentence are looked up in code (local ERP, ~0.1 s) and
                  added to the turn as if the model had called identify_caller / find_product: the model answers in
                  ONE call instead of two. Strict: a client is injected only if its name was really said (a colleague
                  « Julien au dépôt » is not the client whose contact is called Julien).

Both return what to append to the conversation (assistant tool calls + tool results [+ reply]) and never raise:
on any doubt or error they return nothing and the normal agent runs.
"""

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.agent.guardrails import may_confirm
from app.agent.speak_fr import say_date
from app.schemas import ChatMessage, ToolCall
from app.tools import TOOLS, ErpClient, ToolContext
from app.agent.prompts import get_meta
from app.tools.voice import _norm, short_name

MARKET_BUDGET_S = 0.6  # the in-call market (all short products at once) measured ~0.15 s on the local ERP
StatusFn = Callable[[str], Awaitable[None]]  # progress for the caller's screen (« Recherche des offres… »)

ORDER_TOOLS = ("create_draft_order", "add_order_line", "update_order_line", "remove_order_line", "update_order", "get_order")
MAX_CONFIRM_WORDS = 8  # « ouais vas-y c'est bon » yes; a long sentence probably says more than yes: the model reads it
STOP = {"le", "la", "les", "l", "de", "du", "des", "d", "chez", "au", "aux", "et", "a", "en", "sur", "the"}
INTRO = re.compile(r"\b(?:c'est|c est|ici|de la part de|je suis|moi c'est|je m'appelle)\s+(.+?)(?=[.!?,;:]|$)", re.I)


@dataclass
class Injected:
    """What a fast path adds to the turn: messages for the history, events for the UI, and maybe the final reply."""

    messages: list[ChatMessage]
    calls: list[tuple[ToolCall, object, int]]  # (call, result, ms) for the UI
    reply: str | None = None


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=f"fast_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _results(history: list[ChatMessage]):
    """(tool name, parsed result) of every tool message, oldest first."""
    for m in history:
        if m.role == "tool" and m.name:
            try:
                yield m.name, json.loads(m.content)
            except (ValueError, TypeError):
                continue


# ── « oui c'est bon » → confirm in code ──────────────────────────────────────────────────────────────────────────


def pending_draft(history: list[ChatMessage]) -> dict | None:
    """The draft order the conversation is about: the latest order result, if it is still a draft."""
    latest = None
    for name, r in _results(history):
        if name in (*ORDER_TOOLS, "confirm_order", "cancel_order") and isinstance(r, dict) and r.get("id"):
            latest = r
    return latest if latest and latest.get("status") == "draft" else None


def draft_is_last_recap(working: list[ChatMessage]) -> bool:
    """The yes answers the draft as written: the order was written in the turn just before this yes. If the caller
    asked a change since and the model only talked about it (mar 03:58, s01: « je passe les diots à vingt kilos,
    c'est bien ça ? » without update_order_line), confirming the stored draft would confirm the OLD quantities."""
    last_write = max((i for i, m in enumerate(working) if m.role == "tool" and m.name in ORDER_TOOLS), default=-1)
    return last_write >= 0 and not any(m.role == "user" for m in working[last_write + 1 : -1])


def caller_blocked(history: list[ChatMessage]) -> bool:
    return any(name == "identify_caller" and isinstance(r, dict) and r.get("blocked") for name, r in _results(history))


async def fast_confirm(working: list[ChatMessage], erp: ErpClient, today: str | None) -> Injected | None:
    heard = working[-1].content
    if len(heard.split()) > MAX_CONFIRM_WORDS or not may_confirm(working, "voice"):
        return None
    draft = pending_draft(working[:-1])
    if not draft or draft.get("warnings") or not draft.get("lines") or caller_blocked(working) or not draft_is_last_recap(working):
        return None
    call = _call("confirm_order", {"orderId": draft["id"]})
    t0 = time.perf_counter()
    try:
        result = await TOOLS["confirm_order"].run(call.arguments, ToolContext(erp=erp, actor="voice_agent"))
    except Exception as e:
        print(f"[fast] confirm_order failed ({e}): the model handles the turn", flush=True)
        return None
    ms = int((time.perf_counter() - t0) * 1000)
    when = f", livraison {say_date(result['deliveryDate'], today)}" if result.get("deliveryDate") else ""
    reply = f"Parfait, c'est validé{when}."
    if warnings := [w.get("message") for w in result.get("warnings") or [] if w.get("message")]:
        reply += f" Attention : {' ; '.join(warnings)}."
    reply += " Merci, bonne journée !"
    messages = [
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id=call.id, name=call.name, content=_json(result)),
        ChatMessage(role="assistant", content=reply),
    ]
    return Injected(messages=messages, calls=[(call, result, ms)], reply=reply)


# ── preload the caller and the products named ────────────────────────────────────────────────────────────────────


def _words(text: str) -> list[str]:
    """Normalised words, crude singular (« reblochons laitiers » → « reblochon laitier »)."""
    out = []
    for w in re.split(r"[^a-z0-9]+", _norm(text).replace("'", " ")):
        if w:
            out.append(w[:-1] if len(w) > 3 and w[-1] in "sx" else w)
    return out


_index: tuple[float, list[tuple[tuple[str, ...], str]]] = (0.0, [])  # (expires, [(words of a phrase, phrase)])


async def product_index(erp: ErpClient) -> list[tuple[tuple[str, ...], str]]:
    """Every alias and short name of the catalogue, as word tuples, longest first. Cached 5 minutes."""
    global _index
    if time.monotonic() < _index[0]:
        return _index[1]
    products = await erp.get("/api/products", {"limit": 500})
    phrases = {p for prod in products for p in [*(prod.get("aliases") or []), short_name(prod.get("name") or "")] if p}
    index = sorted(((tuple(_words(p)), p) for p in phrases if _words(p) and not all(
        w in {
            "meule", "piece", "pce", "kg", "kilo",
            "gramme", "g", "litre", "l", "carton",
            "plateau", "sac", "seau", "barquette",
            "lot", "boite", "paquet", "portion"
        } for w in _words(p)
    )), key=lambda x: (-len(x[0]), x[1]))
    _index = (time.monotonic() + 300, index)
    return index


def _raw_words(text: str) -> list[str]:
    """Same split as _words, without the singular (« trois » must stay « trois »)."""
    return [w for w in re.split(r"[^a-z0-9]+", _norm(text).replace("'", " ")) if w]


NUM_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
             "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16, "vingt": 20,
             "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60, "cent": 100, "cents": 100,
             "mille": 1000, "demi": 0.5, "et": 0}  # fmt: skip
QTY_SKIP = {"de", "d", "du", "des", "kg", "kilo", "kilos", "g", "l", "litre", "litres", "piece", "pieces", "carton",
            "cartons", "meule", "meules", "plateau", "plateaux", "sac", "sacs", "seau", "seaux", "barquette", "barquettes"}  # fmt: skip


def parse_fr_number(tokens: list[str]) -> float | None:
    """« vingt cinq » → 25, « quatre vingt » → 80, « deux cents » → 200, « 12 » → 12, « 2 5 » (2,5) → 2.5."""
    if not tokens:
        return None
    if all(t.isdigit() for t in tokens):
        return float(tokens[0]) if len(tokens) == 1 else float(f"{tokens[0]}.{tokens[1]}")
    total = current = 0.0
    for t in tokens:
        v = NUM_WORDS.get(t)
        if v is None:
            return None
        if v == 100:
            current = (current or 1) * 100
        elif v == 1000:
            total += (current or 1) * 1000
            current = 0
        elif v == 20 and current and current % 10 == 4:  # quatre vingt
            current *= 20
        else:
            current += v
    return total + current or None


def quantities_said(text: str, index: list[tuple[tuple[str, ...], str]], limit: int = 4) -> list[tuple[str, float | None]]:
    """Each catalogue phrase said, with the quantity said just before it if any: « vingt reblochons laitiers et
    3 kilos de tomme » → [("reblochon laitier", 20), ("tomme", 3)]; « du beaufort » → [("beaufort", None)]."""
    raw = _raw_words(text)
    found = _matches(_words(text), index)
    out = []
    for i, phrase in found[:limit]:
        j = i - 1
        while j >= 0 and raw[j] in QTY_SKIP:
            j -= 1
        k = j
        while k >= 0:
            t = raw[k]
            if t.isdigit() or (t in NUM_WORDS and t != "et"):
                k -= 1
            elif t == "et" and k < j and k > 0 and raw[k - 1] in NUM_WORDS and not raw[k + 1].isdigit():  # vingt et un
                k -= 1
            else:
                break
        out.append((phrase, parse_fr_number(raw[k + 1 : j + 1])))
    return out


def units_said(text: str, index: list[tuple[tuple[str, ...], str]], limit: int = 4) -> dict[str, set[str]]:
    """The unit / packaging words said between the quantity and each product: « cinquante meules de raclette » →
    {"raclette": {"meules", "de"}}. Same walk as quantities_said."""
    raw = _raw_words(text)
    out: dict[str, set[str]] = {}
    for i, phrase in _matches(_words(text), index)[:limit]:
        j = i - 1
        while j >= 0 and raw[j] in QTY_SKIP:
            j -= 1
        out[phrase] = set(raw[j + 1 : i])
    return out


UNIT_WORDS = {"kg": {"kg", "kilo", "kilos"}, "liter": {"l", "litre", "litres"}, "piece": {"piece", "pieces"}}
LINK_WORDS = {"de", "d", "du", "des"}


PACKAGING = {"meule", "demi", "plateau", "sac", "seau", "barquette", "carton", "boite", "lot", "caisse", "palette", "colis"}


def unit_ok(words: set[str], product_result: dict, phrase: str = "") -> bool:
    """The quantity was said in the product's own unit (or with no unit), and no packaging word is in play: « cinquante
    meules de raclette » is 50 wheels, not 50 kg — the agent asks (a meule is a packaging, Abel 05:05)."""
    unit = (product_result.get("product") or {}).get("unit") or ""
    if PACKAGING & set(_words(phrase)) or PACKAGING & {_words(w)[0] for w in words if _words(w)}:
        return False
    return (words - LINK_WORDS) <= UNIT_WORDS.get(unit, set())


def _matches(words: list[str], index: list[tuple[tuple[str, ...], str]]) -> list[tuple[int, str]]:
    used = [False] * len(words)
    found: list[tuple[int, str]] = []
    for phrase_words, phrase in index:
        n = len(phrase_words)
        for i in range(len(words) - n + 1):
            if tuple(words[i : i + n]) == phrase_words and not any(used[i : i + n]):
                used[i : i + n] = [True] * n
                found.append((i, phrase))
                break
    return sorted(found)


def products_said(text: str, index: list[tuple[tuple[str, ...], str]], limit: int = 4) -> list[str]:
    """The catalogue phrases said in `text`, longest match first, no overlap: « vingt reblochons laitiers et de la
    tomme » → ["reblochon laitier", "tomme"]."""
    return [p for p, _ in quantities_said(text, index, limit)]


def known_client(history: list[ChatMessage]) -> str | None:
    """The client already identified in this conversation (identify_caller result or a tool call's clientId)."""
    client = None
    for m in history:
        if m.role == "assistant" and m.tool_calls:
            for c in m.tool_calls:
                if isinstance(c.arguments, dict) and isinstance(c.arguments.get("clientId"), str):
                    client = c.arguments["clientId"]
        if m.role == "tool" and m.name == "identify_caller":
            try:
                r = json.loads(m.content)
            except ValueError:
                continue
            if r.get("found"):
                client = r["client"]["id"]
    return client


def name_was_said(card: dict, text: str) -> bool:
    """Most of the client's name is in the sentence (all of it, or half of it + a contact's first name)."""
    said = set(_words(text))
    name = [w for w in _words(card["client"]["name"]) if w not in STOP]
    if not name:
        return False
    share = sum(w in said for w in name) / len(name)
    first_names = {_words(c["name"])[0] for c in card["client"].get("contacts") or [] if _words(c.get("name") or "")}
    return share == 1 or (share >= 0.5 and bool(first_names & said))


def who_segments(text: str) -> list[str]:
    """Where the caller's name probably is: after « c'est / ici / de la part de », else the first clause."""
    out = [m.group(1).strip() for m in INTRO.finditer(text)]
    first = re.split(r"[.!?]", text, maxsplit=1)[0].strip()
    if first and first not in out and len(first.split()) <= 8:
        out.append(first)
    return out


# Seen in real runs (mar 03:58, s04, s11): everything known, the model said « Je valide ? » without writing the draft,
# so the caller's yes needed a 2-second model call instead of the 40 ms confirmation in code.
READY_HINT = ("Commande complète (client, produits, quantités, stock) : dans la MÊME réponse, le récap court puis "
              "« Je valide ? », ET create_draft_order avec toutes les lignes.")


DRAFT_HINT = ("Brouillon écrit par le système avec ce que l'appelant vient de dire : récap court avec `say`, puis "
              "« Je valide ? ». Ne le recrée pas ; un changement = update_order_line / remove_order_line / update_order.")
USUAL = re.compile(r"comme d'(?:habitude|hab)|comme toujours|(?:mon|le|son) habituel|la même chose que d'habitude", re.I)
WEEKDAYS = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4, "samedi": 5, "dimanche": 6}


def date_said(text: str, today: str | None) -> str | None:
    """« pour samedi » → next Saturday after today (ISO); « demain », « après-demain », « le week-end » (= samedi).
    None when no day was said: the ERP then takes the client's next delivery day."""
    from datetime import date, timedelta

    if not today:
        return None
    t, d = _norm(text), date.fromisoformat(today[:10])
    if re.search(r"\bapres[- ]demain\b", t):
        return (d + timedelta(days=2)).isoformat()
    if re.search(r"\bdemain\b", t):
        return (d + timedelta(days=1)).isoformat()
    day = next((n for w, n in WEEKDAYS.items() if re.search(rf"\b{w}\b", t)), None)
    if day is None and re.search(r"\bweek[- ]?end\b", t):
        day = 5
    if day is None:
        return None
    return (d + timedelta(days=(day - d.weekday() - 1) % 7 + 1)).isoformat()


async def preload(working: list[ChatMessage], erp: ErpClient, on_status: StatusFn | None = None) -> Injected | None:
    """The side agents, in code and in parallel, before the model answers: who is calling (only if their name was
    said), each product named with its quantity (stock, the caller's price, volume discount, substitutes), and, for a
    product short for the quantity asked, the market negotiation with the suppliers (market/express.source: every
    supplier at once, replayed on the « Marché du matin » page). All products are evaluated concurrently."""
    heard = working[-1].content
    ctx = ToolContext(erp=erp, actor="voice_agent")

    async def status(label: str) -> None:
        if on_status:
            await on_status(label)

    try:
        said = quantities_said(heard, await product_index(erp))
        client_id = known_client(working[:-1])
        t0 = time.perf_counter()
        if said:
            await status("Recherche des offres…")
        asks = [{"query": p, **({"qty": q} if q else {})} for p, q in said]
        injected: list[tuple[ToolCall, object, int]] = []
        products: list[tuple[dict, dict]] = []  # (ask, find_product-like result)
        delivery: str | None = None
        if client_id is None:
            for who in who_segments(heard):
                args = {"who": who, **({"products": asks} if asks else {})}
                card = await TOOLS["identify_caller"].run(args, ctx)
                if card.get("found") and name_was_said(card, heard):
                    injected.append((_call("identify_caller", args), card, int((time.perf_counter() - t0) * 1000)))
                    client_id = card["client"]["id"]
                    delivery = (card.get("delivery") or {}).get("next")
                    products = list(zip(asks, card.get("products") or [], strict=False))
                    break
        if said and not products:
            calls = [_call("find_product", {**a, **({"clientId": client_id} if client_id else {})}) for a in asks]
            results = await asyncio.gather(*(TOOLS["find_product"].run(c.arguments, ctx) for c in calls))
            ms = int((time.perf_counter() - t0) * 1000)
            injected += [(c, r, ms) for c, r in zip(calls, results, strict=True)]
            products = list(zip(asks, results, strict=True))
        if not injected:
            return None
        card = next((r for c, r, _ in injected if c.name == "identify_caller"), {})
        ready = client_id and products and not card.get("blocked") and all(
            a.get("qty") and r.get("found") and r.get("product") and not r.get("candidates") and not r.get("pickedBecause")
            and (r.get("stock") or {}).get("enough", True)
            for a, r in products
        )
        # No speculative order (Abel, 05:05): the preload writes a draft ONLY in the safe case — every product named by
        # its own name (unit words are not products), one reference (no variant to choose, no usual guessed), the
        # quantity said in the product's own unit (« cinquante meules » of a kg product → the agent asks), enough stock
        # (else the market and the customer's choice come first), the delivery day said, client known and not
        # blocked; or « comme d'habitude » asked explicitly. Everything else: the agent resolves it with the caller.
        when = date_said(heard, (await get_meta()).get("today"))
        units = units_said(heard, await product_index(erp))
        lines = None
        if ready and when and not pending_draft(working[:-1]) and all(unit_ok(units.get(a["query"], set()), r, a["query"]) for a, r in products):
            lines = [{"productId": r["product"]["id"], "qty": a["qty"]} for a, r in products]
        elif card.get("found") and not card.get("blocked") and not said and USUAL.search(heard) and not pending_draft(working[:-1]):
            lines = [{"productId": u["productId"], "qty": float(round(u["avgQty"]))} for u in card.get("usual") or [] if round(u["avgQty"]) > 0]
        if lines:
            args = {"clientId": client_id, "lines": lines, **({"deliveryDate": when} if when else {})}
            t2 = time.perf_counter()
            draft = await TOOLS["create_draft_order"].run(args, ctx)
            if isinstance(draft, dict) and draft.get("id"):
                draft["next"] = DRAFT_HINT
                injected.append((_call("create_draft_order", args), draft, int((time.perf_counter() - t2) * 1000)))

        # short for the quantity asked → the market, for all of them at once
        short = [(a, r) for a, r in products if r.get("found") and a.get("qty") and not (r.get("stock") or {}).get("enough", True)]  # fmt: skip
        if short:
            delivery = date_said(heard, (await get_meta()).get("today")) or delivery  # « pour samedi »: the market sources for Saturday
            await status("Vérification du stock…")
            await status("Comparaison des prix…")
            from app.market.express import source  # the market imports the tools: imported when needed
            from app.market.live_agents import mode

            t1 = time.perf_counter()
            calls = [_call("source_product", {"productId": r["product"]["id"], "qty": a["qty"], **({"clientId": client_id} if client_id else {}),
                                              **({"deliveryDate": delivery} if delivery else {})}) for a, r in short]  # fmt: skip
            try:  # bounded: past MARKET_BUDGET_S the model answers with the stock and calls source_product itself
                sourced = await asyncio.wait_for(asyncio.gather(*(source(erp, c.arguments["productId"], c.arguments["qty"], client_id, delivery)
                                                                  for c in calls), return_exceptions=True), MARKET_BUDGET_S if mode() == 'legacy' else 7.0)  # fmt: skip
            except TimeoutError:
                sourced = [TimeoutError()] * len(calls)
            ms = int((time.perf_counter() - t1) * 1000)
            injected += [(c, r, ms) for c, r in zip(calls, sourced, strict=True) if not isinstance(r, BaseException)]
        return _as_injected(injected)
    except Exception as e:
        print(f"[fast] preload skipped ({type(e).__name__}: {e})", flush=True)
        return None


def _as_injected(calls: list[tuple[ToolCall, object, int]]) -> Injected:
    messages = [ChatMessage(role="assistant", content="", tool_calls=[c for c, _, _ in calls])]
    messages += [ChatMessage(role="tool", tool_call_id=c.id, name=c.name, content=_json(r)) for c, r, _ in calls]
    return Injected(messages=messages, calls=calls)
