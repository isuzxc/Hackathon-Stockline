"""The team's voice upgrade: spoken French before TTS, composite phone tools, order edits, filler, confirmation guard."""

import asyncio
import json

import httpx
import pytest
from test_history_pairing import FakeWS, ScriptedLLM

from app.agent.guardrails import is_mutation, may_confirm
from app.agent.speak_fr import to_speech
from app.clients import LLMText, LLMToolCall
from app.clients.mock_speech import MockTTS
from app.schemas import ChatMessage, ToolCall
from app.services import voice_session
from app.tools import TOOLS, ErpClient, ToolContext
from app.tools import voice as voice_tools

TODAY = "2026-11-24"  # a Tuesday, like the data pack's meta.today


# ── spoken French ────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "written,spoken",
    [
        ("Le **reblochon** est à 9.40 €/pièce.", "Le reblochon est à neuf euros quarante la pièce."),
        ("Ça fait 12 kg à 18,50 € le kg.", "Ça fait douze kilos à dix-huit euros cinquante le kilo."),
        ("Livraison 2026-11-25.", "Livraison demain."),
        ("Livraison le 2026-11-27.", "Livraison le vendredi vingt-sept novembre."),
        ("La tournée part à 5h45.", "La tournée part à cinq heures quarante-cinq."),
        ("Commande CMD-2026-006831 créée.", "Commande numéro huit cent trente et un créée."),
        ("Le client cli_0027 (prd_0001) est servi.", "Le client est servi."),
        ("- 4 reblochons\n- 2 tommes", "4 reblochons, 2 tommes"),
        ("Remise de 10 %, total 245,80 € HT.", "Remise de dix pour cent, total deux cent quarante-cinq euros quatre-vingts hors taxes."),
        ("Il reste 1,5 kg.", "Il reste un kilo et demi."),
        ("Reblochon AOP fermier", "Reblochon A O P fermier"),
    ],
)
def test_to_speech(written, spoken):
    assert to_speech(written, TODAY) == spoken


# ── confirmation guard ───────────────────────────────────────────────────────────────────────────────────────────

RECAP = ChatMessage(role="assistant", content="Quatre reblochons, livraison demain, total trente euros. Je valide ?")


@pytest.mark.parametrize(
    "answer,ok",
    [
        ("Ouais vas-y", True),
        ("ça marche", True),
        ("nickel", True),
        ("c'est tout bon", True),
        ("non attends", False),
        ("Oui mais enlève le beurre", False),  # a correction, not an approval
        ("oui, rajoute deux tommes", False),
        ("euh je sais pas", False),
    ],
)
def test_colloquial_confirmation(answer, ok):
    history = [ChatMessage(role="user", content="c'est le Panier Savoyard"), RECAP, ChatMessage(role="user", content=answer)]
    assert may_confirm(history, "voice") is ok


def test_command_mode_keeps_one_shot_orders():
    msg = ChatMessage(role="user", content="ajoute deux reblochons et confirme la commande du Panier Savoyard")
    assert may_confirm([msg], "command")


def test_order_edits_are_serialised_writes():
    assert all(is_mutation(n) for n in ("update_order_line", "remove_order_line", "update_order", "cancel_order"))
    assert not any(is_mutation(n) for n in ("identify_caller", "find_product"))


# ── composite tools over a fake ERP ──────────────────────────────────────────────────────────────────────────────

PRODUCTS = {
    "prd_0001": {"id": "prd_0001", "name": "Reblochon de Savoie AOP fermier (450 g)", "unit": "piece", "price": 9.4, "available": 851.0, "stock": {"incoming": 0}},
    "prd_0002": {"id": "prd_0002", "name": "Reblochon de Savoie AOP laitier (450 g)", "unit": "piece", "price": 7.2, "available": 1309.0, "stock": {}},
    "prd_0018": {"id": "prd_0018", "name": "Raclette au poivre (demi-meule 3 kg)", "unit": "kg", "price": 16.9, "available": 9.6, "stock": {"incoming": 9.0}},
}
CLIENT = {
    "id": "cli_0027", "name": "Le Panier Savoyard du Grand-Bornand", "type": "supermarket", "city": "Le Grand-Bornand",
    "status": "active", "zone": "Aravis", "deliveryDays": ["wed", "sat"], "creditLimit": 50000, "outstanding": 45000, "overdue": 0,
    "preferences": "Accepte les substitutions équivalentes si on prévient.",
    "contacts": [{"name": "Jean-Marc Dupraz", "role": "Gérant", "phone": "04"}],
    "usual": [{"productId": "prd_0001", "productName": "Reblochon de Savoie AOP fermier (450 g)", "unit": "piece", "avgQty": 51.8, "lastQty": 46, "frequency": 0.9, "lastOrderedAt": "2026-11-21"}],
    "recentOrders": [{"id": "ord_1", "number": "CMD-2026-000001", "status": "draft", "deliveryDate": "2026-11-25", "totalHt": 10}],
    "recentEvents": [], "notes": [{"at": "2026-11-01T00:00:00Z", "author": "x", "text": "Livrer par l'arrière."}],
}
ORDER = {
    "id": "ord_9", "number": "CMD-2026-000009", "status": "draft", "client": {"name": CLIENT["name"]}, "deliveryDate": "2026-11-25",
    "totalHt": 232.5, "warnings": [],
    "lines": [{"id": "lin_1", "productId": "prd_0004", "productName": "Tomme de Savoie IGP (meule d'environ 1,8 kg)", "qty": 15, "unit": "kg", "unitPrice": 15.5, "available": 300}],
}


def fake_erp(requests: list):
    def handler(req: httpx.Request) -> httpx.Response:
        requests.append((req.method, req.url.path, dict(req.url.params), req.content and json.loads(req.content)))
        p, q = req.url.path, req.url.params
        if p == "/api/search":
            if q["types"] == "client":
                hits = {"panier": [("cli_0027", CLIENT["name"], 95), ("cli_0060", "Le Panier Savoyard de Saint-Jean-de-Sixt", 90)],
                        "panier savoyard du grand-bornand": [("cli_0027", CLIENT["name"], 100), ("cli_0060", "Le Panier Savoyard de Saint-Jean", 60)]}[q["q"]]
                return httpx.Response(200, json=[{"type": "client", "id": i, "label": n, "sublabel": "", "score": s} for i, n, s in hits])
            hits = {"reblo": [("prd_0001", 95), ("prd_0002", 95)], "raclette au poivre": [("prd_0018", 100)]}[q["q"]]
            return httpx.Response(200, json=[{"type": "product", "id": i, "label": PRODUCTS[i]["name"], "score": s} for i, s in hits])
        if p == "/api/clients/cli_0027":
            return httpx.Response(200, json=CLIENT)
        if p == "/api/clients/cli_0027/usual":
            return httpx.Response(200, json=CLIENT["usual"])
        if p == "/api/context/for":
            return httpx.Response(200, json={"label": CLIENT["name"], "entries": [{"id": "c1", "kind": "relationship", "title": "Les personnes", "body": "Tutoie."}, {"id": "c2", "kind": "pricing", "title": "Prix", "body": "x"}], "related": []})
        if p == "/api/routes":
            return httpx.Response(200, json=[{"id": "rte_2", "name": "Tournée Aravis", "zone": "Aravis", "days": ["mon", "wed", "sat"], "departureTime": "05:45"}])
        if p.startswith("/api/products/") and p.endswith("/alternatives"):
            return httpx.Response(200, json=[{"id": "prd_0014", "name": "Raclette de Savoie IGP (demi-meule 3 kg)", "unit": "kg", "price": 16.4, "available": 146.9}])
        if p.startswith("/api/products/"):
            return httpx.Response(200, json=PRODUCTS[p.rsplit("/", 1)[1]])
        if p == "/api/prices/resolve":
            return httpx.Response(200, json={"price": 8.9, "listPrice": 9.4, "source": "client_price", "explanation": "Prix négocié", "candidates": []})
        if p.startswith("/api/orders/"):
            return httpx.Response(200, json=ORDER)
        return httpx.Response(404, json={"error": {"message": f"no route {p}"}})

    return handler


@pytest.fixture
def ctx(monkeypatch):
    requests: list = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake_erp(requests)))
    monkeypatch.setattr(voice_tools, "_routes_cache", None)

    async def today():
        return TODAY

    monkeypatch.setattr(voice_tools, "_today", today)
    c = ToolContext(erp=ErpClient("voice_agent", base_url="http://erp", http=http), actor="voice_agent")
    c.requests = requests  # type: ignore[attr-defined]
    return c


async def test_identify_caller_ambiguous_asks(ctx):
    r = await TOOLS["identify_caller"].run({"who": "panier"}, ctx)
    assert r["found"] is False and [c["clientId"] for c in r["candidates"]] == ["cli_0027", "cli_0060"]


async def test_identify_caller_loads_everything_in_one_call(ctx):
    r = await TOOLS["identify_caller"].run({"who": "panier savoyard du grand-bornand"}, ctx)
    assert r["client"]["id"] == "cli_0027"
    assert r["delivery"] == {"days": ["mercredi", "samedi"], "route": "Tournée Aravis", "routeDays": ["lundi", "mercredi", "samedi"], "departure": "05:45", "next": "2026-11-25", "nextSay": "demain"}
    assert r["usual"][0]["say"] == "cinquante-deux Reblochon de Savoie AOP fermier"
    assert "warning" in r["credit"]  # 45 000 of 50 000
    assert r["openOrders"][0]["id"] == "ord_1"  # the caller may be calling to change it
    assert [t["title"] for t in r["howToTalk"]] == ["Les personnes"]  # pricing entries are not "how to talk"
    assert "phone" not in r["client"]["contacts"][0]


async def test_find_product_prefers_the_callers_usual_product(ctx):
    r = await TOOLS["find_product"].run({"query": "reblo", "clientId": "cli_0027", "qty": 4}, ctx)
    assert r["product"]["id"] == "prd_0001" and "pickedBecause" in r
    assert r["price"]["say"] == "huit euros quatre-vingt-dix la pièce" and r["price"]["listPrice"] == 9.4
    assert r["lineSay"] == "quatre Reblochon de Savoie AOP fermier" and r["stock"]["enough"]


async def test_find_product_without_client_asks_which_one(ctx):
    r = await TOOLS["find_product"].run({"query": "reblo"}, ctx)
    assert r["found"] is False and len(r["candidates"]) == 2


async def test_find_product_short_stock_offers_alternatives(ctx):
    r = await TOOLS["find_product"].run({"query": "raclette au poivre", "clientId": "cli_0027", "qty": 20}, ctx)
    assert r["stock"] == {"available": 9.6, "incoming": 9.0, "enough": False}
    assert r["alternatives"][0]["name"] == "Raclette de Savoie IGP" and "hint" in r


async def test_update_order_line_replaces_quantity_and_reads_back(ctx):
    r = await TOOLS["update_order_line"].run({"orderId": "ord_9", "lineId": "lin_1", "qty": 15}, ctx)
    method, path, _, body = ctx.requests[-1]
    assert (method, path, body) == ("PATCH", "/api/orders/ord_9/lines/lin_1", {"qty": 15})
    assert r["say"] == "quinze kilos de Tomme de Savoie IGP, livraison demain, total deux cent trente-deux euros cinquante hors taxes"


async def test_remove_line_and_move_delivery(ctx):
    await TOOLS["remove_order_line"].run({"orderId": "ord_9", "lineId": "lin_1"}, ctx)
    assert ctx.requests[-1][:2] == ("DELETE", "/api/orders/ord_9/lines/lin_1")
    await TOOLS["update_order"].run({"orderId": "ord_9", "deliveryDate": "2026-11-28"}, ctx)
    assert ctx.requests[-1][::3] == ("PATCH", {"deliveryDate": "2026-11-28"})
    with pytest.raises(ValueError):
        await TOOLS["update_order"].run({"orderId": "ord_9"}, ctx)


# ── voice session: filler + spoken French + first-clause cut ────────────────────────────────────────────────────


class RecordingTTS(MockTTS):
    name = "recording"

    def __init__(self):
        self.texts: list[str] = []

    async def synthesize(self, text, voice=None):
        self.texts.append(text)
        return await super().synthesize(text, voice)


def session_with(monkeypatch, steps, tts):
    # These tests target the original streaming/filler fast path.
    monkeypatch.setattr(voice_session, "negotiation_enabled", lambda: False)
    async def no_fast_path(*args, **kwargs):  # these tests are about the model path (fast paths: test_fast_path.py)
        return None

    monkeypatch.setattr(voice_session, "fast_confirm", no_fast_path)
    monkeypatch.setattr(voice_session, "preload", no_fast_path)
    async def lookup(ctx):
        return {"ok": True}

    from app.tools.registry import Tool, ToolArgs

    tools = [Tool(name="lookup", description="x", args_model=ToolArgs, fn=lookup)]
    monkeypatch.setattr(voice_session, "get_llm_client", lambda: ScriptedLLM(steps))
    monkeypatch.setattr(voice_session, "all_tools", lambda: tools)
    monkeypatch.setattr(voice_session, "get_tts_client", lambda: tts)
    return voice_session.VoiceConnection(FakeWS()), voice_session.Conversation(id="t")


def tool_call(id="a"):
    return LLMToolCall(ToolCall(id=id, name="lookup", arguments={}))


async def test_filler_on_first_tool_call_uses_prerendered_audio(monkeypatch):
    monkeypatch.setattr(voice_session, "USE_FILLERS", True)  # fillers are opt-in (VOICE_FILLERS=1)
    tts = RecordingTTS()
    monkeypatch.setattr(voice_session, "_filler_audio", {})
    await voice_session.prerender_fillers(tts)
    rendered = len(tts.texts)
    assert rendered == len(voice_session.FILLERS) + len(voice_session.ACKS) + 2  # + FIRST_ACK, YES_ACK
    session, conv = session_with(monkeypatch, [[tool_call()], [LLMText("Il reste 12 kg de tomme.")]], tts)
    await session.run_turn(conv, text="il te reste de la tomme ?")
    done = next(m for m in session.ws.sent if m["type"] == "agent_done")
    assert done["text"] == "Je regarde. Il reste 12 kg de tomme."
    assert tts.texts[rendered:] == ["Il reste douze kilos de tomme."]  # filler: no TTS call; answer: spoken French
    assert conv.history[-1].content == "Il reste 12 kg de tomme."  # the model's own words, not the filler
    await session.run_turn(conv, text="et du beaufort ?")  # next turn: another filler
    assert next(m for m in reversed(session.ws.sent) if m["type"] == "agent_done")["text"].startswith("Un instant, je vérifie.")


async def test_no_filler_when_the_agent_already_spoke(monkeypatch):
    tts = RecordingTTS()
    session, conv = session_with(monkeypatch, [[LLMText("Bonjour Jean-Marc ! "), tool_call()], [LLMText("C'est noté.")]], tts)
    await session.run_turn(conv, text="salut")
    assert tts.texts == ["Bonjour Jean-Marc !", "C'est noté."]


def test_first_clause_is_cut_early_then_whole_sentences():
    spoken: list[str] = []
    speaker = voice_session.Speaker(MockTTS(), None, None)  # type: ignore[arg-type]
    speaker._queue = lambda item: (spoken.append(item[0]), setattr(speaker, "queued_any", True))  # type: ignore[method-assign]
    for delta in ["Alors pour demain, je te mets ", "quatre reblochons, deux tommes ", "et la raclette. Je valide ?"]:
        speaker.push(delta)
    assert spoken == ["Alors pour demain, je te mets quatre reblochons,", "deux tommes et la raclette."]


def test_heard_part():
    assert voice_session.heard_part("Bonjour. Je prépare tout. Ensuite", "Bonjour. Je prépare tout.") == "Bonjour. Je prépare tout."
    assert voice_session.heard_part("Bonjour. Ensuite", "") == ""


async def test_prerender_never_raises(monkeypatch):
    class Broken(MockTTS):
        name = "broken"

        async def synthesize(self, text, voice=None):
            raise RuntimeError("401")

    monkeypatch.setattr(voice_session, "_filler_audio", {})
    await asyncio.wait_for(voice_session.prerender_fillers(Broken()), 1)
    assert voice_session._filler_audio == {}


async def test_nothing_heard_is_said_out_loud_not_left_silent(monkeypatch):
    tts = RecordingTTS()
    session, conv = session_with(monkeypatch, [], tts)
    await session.run_turn(conv, text="   ")
    kinds = [m["type"] for m in session.ws.sent]
    assert "error" not in kinds and kinds[-1] == "agent_done"
    assert tts.texts == [voice_session.REPROMPTS[0]]
    assert conv.history == []  # nothing was heard: nothing to remember


def test_key_error_names_the_right_key():
    from app.clients.http import provider_error

    assert "LLM_API_KEY" in str(provider_error("api.anthropic.com", 401, "invalid x-api-key"))
    assert "STT_API_KEY" in str(provider_error("soniox stt", 401, "bad"))
    assert "TTS_API_KEY" in str(provider_error("elevenlabs tts", 401, "bad"))


class DelayedLLM(ScriptedLLM):
    """Each step waits `delay` seconds before its first event (a slow model / first token)."""

    def __init__(self, steps, delay):
        super().__init__(steps)
        self.delay = delay

    async def stream(self, request):
        await asyncio.sleep(self.delay)
        async for ev in super().stream(request):
            yield ev


def slow_session(monkeypatch, steps, delay, tts):
    session, conv = session_with(monkeypatch, [], tts)
    monkeypatch.setattr(voice_session, "get_llm_client", lambda: DelayedLLM(steps, delay))
    monkeypatch.setattr(voice_session, "ACK_AFTER_S", 0.05)
    return session, conv


async def test_slow_answer_gets_an_acknowledgement_first(monkeypatch):
    monkeypatch.setattr(voice_session, "USE_FILLERS", True)  # fillers are opt-in (VOICE_FILLERS=1)
    tts = RecordingTTS()
    monkeypatch.setattr(voice_session, "_filler_audio", {})
    session, conv = slow_session(monkeypatch, [[LLMText("Bonjour Jean-Marc, je vous écoute.")]], 0.2, tts)
    await session.run_turn(conv, text="Bonjour, c'est Jean-Marc")
    done = next(m for m in session.ws.sent if m["type"] == "agent_done")
    assert done["text"] == "Oui, un instant. Bonjour Jean-Marc, je vous écoute."  # first turn: FIRST_ACK
    assert tts.texts == ["Oui, un instant.", "Bonjour Jean-Marc, je vous écoute."]
    assert conv.history[-1].content == "Bonjour Jean-Marc, je vous écoute."  # the model's words only


async def test_fast_answer_gets_no_acknowledgement(monkeypatch):
    tts = RecordingTTS()
    session, conv = slow_session(monkeypatch, [[LLMText("C'est noté.")]], 0.0, tts)
    await session.run_turn(conv, text="cinq kilos de tomme")
    assert tts.texts == ["C'est noté."]


async def test_one_filler_per_turn_ack_then_tool(monkeypatch):
    monkeypatch.setattr(voice_session, "USE_FILLERS", True)  # fillers are opt-in (VOICE_FILLERS=1)
    tts = RecordingTTS()
    session, conv = slow_session(monkeypatch, [[tool_call()], [LLMText("Il en reste.")]], 0.2, tts)
    conv.history = [ChatMessage(role="user", content="salut"), ChatMessage(role="assistant", content="Bonjour !")]
    await session.run_turn(conv, text="il reste du beaufort ?")
    assert tts.texts == ["D'accord.", "Il en reste."]  # no « Je regarde. » after the acknowledgement


async def test_identify_caller_with_products_in_one_call(ctx):
    r = await TOOLS["identify_caller"].run(
        {"who": "panier savoyard du grand-bornand", "products": [{"query": "reblo", "qty": 4}, {"query": "raclette au poivre", "qty": 20}]}, ctx
    )
    assert r["client"]["id"] == "cli_0027"
    reblo, raclette = r["products"]
    assert reblo["asked"] == "reblo" and reblo["product"]["id"] == "prd_0001"  # the caller's usual one
    assert raclette["stock"]["enough"] is False and raclette["alternatives"]


async def test_contact_first_name_picks_the_client(ctx):
    from app.tools.voice import _by_contact_name

    candidates = [{"id": "cli_0060", "score": 90}, {"id": "cli_0027", "score": 90}]  # cli_0060: the fake ERP 404s
    assert await _by_contact_name(ctx, "c'est Jean-Marc du Panier Savoyard", candidates) == "cli_0027"
    assert await _by_contact_name(ctx, "c'est Paul du Panier Savoyard", candidates) is None
