"""
WS /voice — push-to-talk voice sessions (messages: app/schemas/voice.py).

One turn:
    audio_start … binary frames … audio_end   (or a typed `text` message)
      → STT, live: each frame goes to the provider as it arrives (services/utterance.py);
        "transcript" final:false while the caller speaks, then "transcript" final:true soon after audio_end
      → agent (voice mode) streaming "agent_text" / "agent_tool"
      → each finished sentence goes to TTS right away (sentence by sentence = first audio sooner)
      → "agent_audio_start" {mime}, binary audio chunks…, "agent_audio_end"   (one audio segment per turn)
      → "agent_done" {text, timings}

Barge-in: a new audio_start / text while the agent is still busy cancels the running turn (asyncio
cancellation); what was heard stays in the history, and of the agent's reply only what was actually voiced.

Team additions (GUIDE-VOIX 4.7, 4.8, 4.9, 5.2, 5.3):
  - filler: on the first tool call of a turn, if the agent has said nothing yet, a short pre-synthesised phrase
    (« Je regarde. ») plays at once (no TTS call: the audio was rendered when the connection opened);
  - warm-up: those renders also open the TTS connection before the caller first speaks;
  - spoken French: every sentence goes through speak_fr.to_speech() before TTS (prices, units, dates, ids,
    markdown), so the voice never says « dix-huit point cinq zéro euro slash kg »;
  - earlier first sound: the very first chunk of a reply may be cut at a comma once it is long enough;
  - honest history on barge-in: the model only remembers the sentences whose audio actually started.
"""

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.agent import build_system_prompt, run_agent, sanitize_history
from app.agent.loop import ms_since
from app.agent.prompts import get_meta
from app.agent.guardrails import may_confirm
from app.agent.speak_fr import say_date, to_speech
from app.clients import LLMRequest, STTUpdate, TTSClient, get_llm_client, get_stt_client, get_tts_client
from app.clients.http import http_client
from app.config import env, settings
from app.schemas import (
    AgentAudioEndMessage,
    AgentAudioMarkMessage,
    AgentAudioStartMessage,
    AgentDoneMessage,
    AgentOfferMessage,
    AgentStatusMessage,
    AgentTextMessage,
    AgentToolMessage,
    AudioEndMessage,
    AudioStartMessage,
    ChatMessage,
    DoneEvent,
    ErrorEvent,
    ReadyMessage,
    ResetMessage,
    StartMessage,
    TextEvent,
    TextMessage,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    ToolTiming,
    TranscriptMessage,
    TurnTimings,
    VoiceClientMessage,
    VoiceErrorMessage,
)
from app.schemas.base import WireModel
from app.tools import ErpClient, all_tools

from .model_router import fast_client, route
from .fast_path import MAX_CONFIRM_WORDS, Injected, caller_blocked, fast_confirm, preload
from .offer import OfferBoard
from .negotiation import enabled as negotiation_enabled
from .utterance import Utterance
from .vocabulary import stt_hint

client_message = TypeAdapter(VoiceClientMessage)


@dataclass
class Conversation:
    id: str
    actor_name: str | None = None
    history: list[ChatMessage] = field(default_factory=list)
    fillers_used: int = 0  # rotates the filler phrases so the caller does not hear the same one every turn
    credit_said: bool = False  # the near-limit encours was mentioned in this call (said once, by the model or the code)


# Said while the first tool runs, when the agent has said nothing yet. Short, varied, no promise about the result.
FILLERS = ["Je regarde.", "Un instant, je vérifie.", "Je regarde ça.", "Deux secondes.", "Je vérifie tout de suite."]

# First sound within a deadline (presentation: « le temps avant le premier mot »). If nothing has been said
# ACK_AFTER_S after the transcript is ready, a pre-rendered short acknowledgement plays, like a person on the phone,
# while the model and the tools work. At most ONE filler per turn: this one, or « Je regarde. » on the first tool call,
# whichever comes first. Counted as first sound in the timings: say so in DECISIONS.md.
ACK_AFTER_S = 0.35
# Off by default (team decision, mar.): « Je regarde », « Un instant » sound like a call centre and add nothing; the
# answer itself comes fast enough. VOICE_FILLERS=1 brings back the acknowledgement and the first-tool filler.
USE_FILLERS = env("VOICE_FILLERS", "0").lower() in ("1", "true", "on", "yes")
PRELOAD_TIMEOUT_S = 1.0  # identification + offers + (if stock is short) the express market, all in parallel
PROMPT_CACHE_REFRESH_S = 240  # fast_path.preload: past this, the model looks the caller / products up itself
ACKS = ["D'accord.", "Oui.", "Très bien."]
# « Oui » to a recap whose draft the code cannot confirm alone (the model wrote no draft yet, or the caller changed
# something since): said at once, pre-rendered, while the model writes + confirms (~2 s in the 04:04 run: s04, s06,
# s11, s14). The model's own text is then voiced only if it reports a problem or asks something new.
YES_ACK = "Parfait, c'est noté."
FIRST_ACK = "Oui, un instant."  # first turn of a call: the caller usually just said who they are

# The status line while the model's own tools run (the preload sends its own).
STATUS_ON_CALL = {"identify_caller": "Recherche des offres…", "find_product": "Recherche des offres…",
                  "check_stock": "Vérification du stock…", "source_product": "Comparaison des prix…",
                  "negotiate_basket": "Les agents négocient avec les fournisseurs…", "accept_negotiated_offer": "Enregistrement de l’offre…"}  # fmt: skip

# Writes whose result the model is told to read back BEFORE calling them (VOICE_RULES: « écris d'abord ta phrase …
# PUIS appelle create_draft_order … Après le résultat de l'outil, n'écris plus rien, sauf erreur ou alerte »).
WRITE_TOOLS = {"create_draft_order", "add_order_line", "update_order_line", "remove_order_line", "update_order", "log_client_event"}

# The model narrating its own reasoning (seen in real runs with thinking disabled, mar 01:41): it talks about the
# caller in the 3rd person or announces what it is doing. Never voiced (the screen keeps the model's text as written).
META_TALK = re.compile(
    # ARAVIS: additional internal narration
    r"^\s*je n[\x27’]ai pas encore identifié[^.!?]*|"
    r"^\s*je le fais maintenant[^.!?]*|"
    r"^\s*je vais identifier[^.!?]*|"
    r"^\s*je dois (?:identifier|vérifier|consulter)[^.!?]*|"
    r"^\s*je lance (?:la recherche|une recherche)[^.!?]*|"
    r"^\s*je consulte (?:la fiche|le système|l[\x27’]ERP)[^.!?]*|"
    r"(?<!\w)(?:du client|le client (?:a|est|veut|prend|demande|dit)|son habituel|sa commande habituelle|"
    r"je fais répéter|je préfère (?:te|vous) redemander|je vérifie son contenu|je consulte d'abord|"
    r"je modifie la ligne(?: existante)?\s*\.|je retire simplement|je poursuis|le nom est clair|je crée le brouillon|"
    r"la dernière valeur dite|ne me dit pas si|je consigne|le client sera|je l'ajoute au dossier|pas d'alerte|je (?:te|vous) donne le récap|je vais enregistrer|je note (?:la|votre) (?:réclamation|demande)|il a confirmé|elle a confirmé|l'appelant|le sourcing|"
    r"\b(?:il|elle) a (?:dit|confirmé|accepté|répondu)\b|"
    r"^\s*je (?:vérifie|crée|cherche|consulte|lance|contrôle|passe les commandes|m'occupe)\b[^?]*$|"
    r"je regarde(?: ça)?\s*[.!]|un instant[^?]*[.!]|je vais vérifier[^?]*[.!]|je vérifie(?: ça)?(?: tout de suite)?\s*[.!])(?!\w)",
    re.IGNORECASE,
)


# The model sometimes writes its plan in English before answering (mar 03:43, s01: « Tomorrow is Wednesday, a delivery
# day. Usual basket is large; I'll draft it now. »). A sentence made of English function words, in a French call,
# is never voiced.
_EN = re.compile(r"\b(?:the|is|it|it's|i'll|i'm|i|now|and|of|to|will|let|we|this|that|they|he|she|basket|tomorrow|today|draft|check|usual|delivery|day|so|should|with|for|need|needs|order|customer|caller|quantities|large|wants|asked|already)\b", re.IGNORECASE)
_FR = re.compile(r"\b(?:le|la|les|un|une|des|du|de|est|et|vous|tu|je|pour|demain|kilos?|euros?|ça|c'est|oui|non|à)\b", re.IGNORECASE)


def english_aside(sentence: str) -> bool:
    return len(_EN.findall(sentence)) >= 2 and len(_FR.findall(sentence)) <= 1


def speaks_english(heard: str) -> bool:
    return english_aside(heard) or len(_EN.findall(heard)) > 2 * len(_FR.findall(heard)) + 1


# « Marché express » is our internal name; the caller hears what it means (mar 03:43, s14: said 3 times).
_EXPRESS = [
    (re.compile(r"\b(?:avec )?le complément du marché express", re.I), "avec le complément des producteurs"),
    (re.compile(r"\bdu marché express", re.I), "des producteurs"),
    (re.compile(r"\bau marché express", re.I), "aux producteurs"),
    (re.compile(r"\ble marché express", re.I), "la demande aux producteurs"),
    (re.compile(r"\bmarché express", re.I), "demande aux producteurs"),
]


def customer_words(text: str) -> str:
    for pattern, repl in _EXPRESS:
        text = pattern.sub(lambda m, r=repl: r[0].upper() + r[1:] if m.group(0)[:1].isupper() else r, text)
    return text


# An encours near the credit limit must be said once in the call (BRIEF; s04 failed 3 runs in a row with the rule only
# in the prompt). If the model wrote an order without mentioning it, the code adds this sentence.
CREDIT_WORDS = re.compile(r"encours|plafond|facture|règlement|paiement", re.IGNORECASE)
CREDIT_SAY = "Petit point : votre encours approche du plafond, pensez au règlement en attente."
ORDER_WRITES = {"create_draft_order", "add_order_line", "update_order_line"}


_STOP = {"pour", "avec", "dans", "vous", "votre", "nous", "c'est", "sont", "mais", "plus", "tout", "toute", "bien",
         "est", "une", "des", "les", "le", "la", "du", "de", "en", "et", "je", "tu", "il", "on", "ça", "pas"}


def content_words(sentence: str) -> set[str]:
    return {w for w in re.findall(r"[\w']+", sentence.lower()) if len(w) > 3 and w not in _STOP}


def products_named(history: list) -> bool:
    """This turn's caller card came back with the products already named (« vingt kilos de tomme »): an order."""
    for m in reversed(history):
        if getattr(m, "role", None) == "user":
            return False
        if getattr(m, "role", None) == "tool" and getattr(m, "name", None) == "identify_caller" and '"products"' in (m.content or ""):
            return True
    return False


def credit_warning(history: list) -> bool:
    for m in history:
        if getattr(m, "role", None) == "tool" and getattr(m, "name", None) == "identify_caller":
            try:
                r = json.loads(m.content) if isinstance(m.content, str) else m.content
            except (TypeError, ValueError):
                continue
            if isinstance(r, dict) and (r.get("credit") or {}).get("warning"):
                return True
    return False


VALIDATION_ASK = re.compile(r"\bje (?:te |vous )?(?:la |le |les )?valide\b[^?]*\?\s*$", re.IGNORECASE)


def last_reply_asks_validation(history: list) -> bool:
    """The last thing the agent said ends with « Je valide ? » (an order recap), not an offer or a choice."""
    last = next((m for m in reversed(history) if getattr(m, "role", None) == "assistant" and (m.content or "").strip()), None)
    return bool(last and VALIDATION_ASK.search(last.content.strip()))


def _quiet_result(name: str, ok: bool, result) -> bool:
    """A write that went through with nothing to report: the recap said before the call already covers it."""
    return _write_ok(name, ok, result) and not (result.get("warnings") or result.get("alerts"))


def _write_ok(name: str, ok: bool, result) -> bool:
    """A write that went through (maybe with warnings: lot expiring before delivery, stock short…)."""
    return name in WRITE_TOOLS and ok and isinstance(result, dict) and not result.get("error")


# After a write the caller already heard the recap + « Je valide ? ». If the ERP warned, the model re-reads the whole
# recap around the warning (s11, mar 03:27: 451 chars). Only these sentences are voiced: the warning, or a question
# that is not the same « je valide ? » again.
ALERT = re.compile(
    r"attention|expir|périm|date limite|DLC|rupture|manque|insuffisant|ne suffi|plus que|reste que|seulement|plafond|encours|bloqu|"
    r"erreur|impossible|pas pu|ne peux|pas en stock|retard",
    re.IGNORECASE,
)
SAME_QUESTION = re.compile(r"\b(?:valide|confirme|c'est bon|on y va|ça vous va|ça te va)\b", re.IGNORECASE)


def worth_saying_after_write(text: str) -> list[str]:
    """The sentences of the model's text after a warned write that the caller has not heard yet."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?…])\s+", text) if x.strip()]
    return [x for x in sentences if ALERT.search(x) or (x.endswith("?") and not SAME_QUESTION.search(x))]


# Said when the STT returned nothing usable (noise, a mumble, Space released too early).
REPROMPTS = ["Pardon, je n'ai pas bien entendu, vous pouvez répéter ?", "Désolé, ça a coupé. Vous pouvez répéter ?"]

# (tts name, voice, text) → (mime, audio bytes): rendered once per process, at the first connection.
_filler_audio: dict[tuple[str, str | None, str], tuple[str, bytes]] = {}


async def warm_prompt(actor_name: str | None = None) -> None:
    """A 1-token request with the exact voice prefix (tools + system prompt): the provider caches it, so the
    caller's first turn reads it from the cache. Costs one cache write per 5 min, a cache read otherwise. Never raises."""
    for llm in {get_llm_client(), *([fast] if (fast := fast_client()) else [])}:  # each model has its own cache
        if llm.provider != "anthropic":  # the others cache automatically or not at all
            continue
        with suppress(Exception):
            system = await build_system_prompt("voice", actor_name)
            request = LLMRequest(system=system, messages=[ChatMessage(role="user", content="(préchauffage)")], tools=[t.info for t in all_tools()], max_tokens=1)  # fmt: skip
            async for _ in llm.stream(request):
                pass


async def warm_llm() -> None:
    """Open the TLS connection to the LLM provider before the first turn (kept alive in the shared pool,
    clients/http.py): the first model call then skips the handshake. No tokens spent. Never raises."""
    llm = get_llm_client()
    base = getattr(llm, "base_url", None)
    if not base or llm.provider == "mock":
        return
    with suppress(Exception):
        await http_client().get(base, timeout=5)


async def prerender_fillers(tts: TTSClient) -> None:
    """Synthesise the fillers in the background. Doubles as the TTS warm-up (GUIDE-VOIX 4.9: first TTS call ≈ 1.9 s
    cold, then 170–370 ms). Never raises: without a render, the filler is synthesised on the fly."""
    voice = settings.tts.voice or None
    for text in [FIRST_ACK, *ACKS, *FILLERS, YES_ACK]:
        key = (tts.name, voice, text)
        if key in _filler_audio:
            continue
        try:
            result = await tts.synthesize(text, voice)
            async with aclosing(result.chunks) as chunks:  # type: ignore[type-var]
                audio = b"".join([chunk async for chunk in chunks])
            if audio:
                _filler_audio[key] = (result.mime, audio)
        except Exception as e:
            print(f"[voice] filler pre-render failed ({type(e).__name__}: {e})", flush=True)
            return


# In memory: a reconnect with the same sessionId continues the conversation.
conversations: dict[str, Conversation] = {}


class VoiceConnection:
    """One WebSocket connection. Reads client messages; runs each turn as a background task."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.conversation: Conversation | None = None
        self.recording: Utterance | None = None  # while the key is down
        self.turn: asyncio.Task | None = None
        self.warm_up: asyncio.Task | None = None
        self.keep_warm: asyncio.Task | None = None  # refreshes the LLM prompt cache while the page is open
        self._send_lock = asyncio.Lock()

    # ── sending ──

    async def send(self, msg: WireModel) -> None:
        await self._send(text=msg.to_json())

    async def send_bytes(self, data: bytes) -> None:
        await self._send(bytes=data)

    async def _send(self, **message) -> None:
        async with self._send_lock:
            with suppress(WebSocketDisconnect, RuntimeError):  # the client is gone: nothing to do
                if message.get("text") is not None:
                    await self.ws.send_text(message["text"])
                else:
                    await self.ws.send_bytes(message["bytes"])

    # ── receiving ──

    async def serve(self) -> None:
        await self.ws.accept()
        try:
            while True:
                frame = await self.ws.receive()
                if frame["type"] == "websocket.disconnect":
                    break
                if frame.get("bytes") is not None:
                    if self.recording:
                        self.recording.feed(frame["bytes"])
                    continue
                await self.handle_text(frame.get("text") or "")
        except WebSocketDisconnect:
            pass
        finally:
            if self.keep_warm:
                self.keep_warm.cancel()
            await self.interrupt()

    async def handle_text(self, raw: str) -> None:
        try:
            msg = client_message.validate_python(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            await self.send(VoiceErrorMessage(message="invalid JSON message"))
            return
        if not self.conversation and not isinstance(msg, StartMessage):
            await self.open()  # no "start" yet: open a fresh conversation

        match msg:
            case StartMessage(session_id=session_id, actor_name=actor_name):
                await self.interrupt()
                await self.open(session_id, actor_name)
            case AudioStartMessage(mime=mime, hint=hint):
                await self.interrupt()  # barge-in
                self.recording = Utterance(get_stt_client(), mime, hint, on_partial=self.send_partial)
            case AudioEndMessage():
                if self.recording:
                    utterance, self.recording = self.recording, None
                    self.start_turn(utterance=utterance)
            case TextMessage(text=text):
                await self.interrupt()
                self.start_turn(text=text)
            case ResetMessage():
                await self.interrupt()
                self.conversation.history = []
                await self.open(self.conversation.id)

    async def send_partial(self, update: STTUpdate) -> None:
        await self.send(TranscriptMessage(text=update.text, final=False, stable=update.stable))

    async def open(self, session_id: str | None = None, actor_name: str | None = None) -> None:
        conv = conversations.get(session_id or "") or Conversation(id=session_id or str(uuid.uuid4()))
        if actor_name:
            conv.actor_name = actor_name
        conversations[conv.id] = self.conversation = conv
        llm = get_llm_client()
        await self.send(
            ReadyMessage(session_id=conv.id, stt=get_stt_client().name, tts=get_tts_client().name, llm=f"{llm.provider}:{llm.model}")
        )
        # Fetch the company vocabulary for the STT now, not when the caller first speaks; warm the TTS with the fillers.
        self.warm_up = self.warm_up or asyncio.create_task(self._warm_up())
        self.keep_warm = self.keep_warm or asyncio.create_task(self._keep_prompt_cache_warm())

    async def _keep_prompt_cache_warm(self) -> None:
        """The first model call of a cold prompt cache took 2.3 s instead of ~0.7 s (26k tokens to write). Write it
        when the call opens, then refresh it every 4 min while the page is open (Anthropic's cache lives 5 min)."""
        while True:
            await warm_prompt(self.conversation.actor_name if self.conversation else None)
            await asyncio.sleep(PROMPT_CACHE_REFRESH_S)

    async def _warm_up(self) -> None:
        await asyncio.gather(stt_hint(get_stt_client().name, None), prerender_fillers(get_tts_client()), get_meta(), warm_llm())

    def start_turn(self, **turn_input) -> None:
        self.turn = asyncio.create_task(self.run_turn(self.conversation, **turn_input))

    async def interrupt(self) -> None:
        """Barge-in: drop a recording in progress, stop the running turn and wait until it has cleaned up."""
        if self.recording:
            self.recording.cancel()
            self.recording = None
        if self.turn and not self.turn.done():
            self.turn.cancel()
            with suppress(asyncio.CancelledError):
                await self.turn
        self.turn = None

    # ── one turn ──

    async def run_turn(
        self,
        conv: Conversation,
        text: str | None = None,
        utterance: Utterance | None = None,  # spoken: already being transcribed since audio_start
    ) -> None:
        t0 = time.perf_counter()  # end of caller speech
        timings = TurnTimings()
        heard, spoken, hushed = "", "", ""  # hushed: model text after a clean write, not voiced (recap once)
        step_text = ""  # what the model said since the last tool result (not yet in `working`)
        working: list[ChatMessage] | None = None
        committed = False
        speaker: Speaker | None = None
        ack: asyncio.Task | None = None
        said_first: list[str] = []  # the deadline acknowledgement, if it played (spoken before the model's text)
        try:
            # 1. Speech → text: with a streaming STT only the last words are left to finalize
            if utterance is not None:
                text = await utterance.text()
                timings.stt_ms = ms_since(t0)  # end of speech → final transcript
                stt_timings(timings, t0, utterance)
            heard = (text or "").strip()
            if not heard:
                # Nothing usable (noise, a mumble, Space released too early): say so out loud instead of leaving the
                # caller in silence. Not added to the history: nothing was heard.
                reprompt = REPROMPTS[conv.fillers_used % len(REPROMPTS)]
                conv.fillers_used += 1
                await self.send(TranscriptMessage(text="(inaudible)", final=True))
                await self.send(AgentTextMessage(delta=reprompt))
                timings.useful_text_ms = ms_since(t0)
                speaker = Speaker(get_tts_client(), self.send, self.send_bytes)
                speaker.speak(reprompt)
                await speaker.finish()
                audio_timings(timings, t0, speaker)
                await self.send(AgentDoneMessage(text=reprompt, timings=timings))
                return
            await self.send(TranscriptMessage(text=heard, final=True))

            # 2. Agent, on a working copy of the history (committed at the end)
            working = [*conv.history, ChatMessage(role="user", content=heard)]
            today = (await get_meta()).get("today")
            speaker = Speaker(get_tts_client(), self.send, self.send_bytes, today=today)
            speaker.french = not speaks_english(heard)
            ack = asyncio.create_task(self._ack_after(speaker, conv, said_first)) if USE_FILLERS else None
            board = OfferBoard(today=today)
            llm_start = time.perf_counter()

            # 2a. Fast paths (services/fast_path.py): « oui c'est bon » after a recap = confirmed in code, no model call;
            #     otherwise the caller / products named are looked up in code so the model answers in one call.
            erp = ErpClient("voice_agent", conv.actor_name)
            negotiation_first = negotiation_enabled()
            if not negotiation_first and (fast := await fast_confirm(working, erp, today)):
                working += fast.messages
                await self._show_injected(fast)
                add_injected_timings(timings, t0, fast)
                spoken = fast.reply or ""
                await self.send(AgentTextMessage(delta=spoken))
                timings.useful_text_ms = ms_since(t0)
                speaker.push(spoken)
                timings.llm_total_ms = ms_since(llm_start)
                conv.history = trim(working)
                committed = True
                await speaker.finish()
                audio_timings(timings, t0, speaker)
                await self.send(AgentDoneMessage(text=" ".join([*said_first, spoken]).strip(), timings=timings))
                return
            # Instant « Parfait, c'est noté. » only after a real order recap (« … Je valide ? »): never after an alternative,
            # a market offer or any other question — a yes to those is not a confirmation (Abel, 05:05).
            optimistic = not negotiation_first and len(heard.split()) <= MAX_CONFIRM_WORDS and last_reply_asks_validation(conv.history) \
                and may_confirm(working, "voice") and not caller_blocked(working)
            if optimistic:
                spoken = f"{YES_ACK} "
                await self.send(AgentTextMessage(delta=spoken))
                timings.useful_text_ms = ms_since(t0)
                speaker.filler(YES_ACK)  # pre-rendered: plays at once
            confirmed: dict | None = None
            try:  # the ERP is local (~0.1 s); never let a slow ERP hold the turn: then the model looks things up itself
                pre = None if negotiation_first else await asyncio.wait_for(preload(working, erp, self._status), PRELOAD_TIMEOUT_S)
            except TimeoutError:
                pre = None
            if pre:
                working += pre.messages
                await self._show_injected(pre)
                add_injected_timings(timings, t0, pre)
                await self._offer(board, pre.calls)
            tool_started: dict[str, int] = {}
            # Recap once: the model says the recap then calls the write; after a clean result it must stay silent. If it
            # talks anyway (« Pour récapituler : … Je valide ? » a second time), that text is not voiced nor shown.
            hush, step_asked, step_quiet, in_tools, after_results = False, False, True, False, False
            step_written, hush_warned = True, False  # every call of the step was a write that went through
            step_said, step_logs = False, False  # the step's text came before its calls; all its calls were log_client_event
            llm, why = (get_llm_client(), "negotiation-first") if negotiation_first else route(get_llm_client(), heard, conv.history, bool(pre), optimistic, caller_blocked(working))
            timings.llm_model = f"{getattr(llm, 'model', llm.provider)} ({why})"
            print(f"[route] {timings.llm_model}", flush=True)
            tools = all_tools()
            if negotiation_first:
                allowed = {"identify_caller", "find_product", "get_client", "get_context", "search_context",
                           "get_order", "get_client_history", "negotiate_basket", "accept_negotiated_offer"}
                tools = [t for t in tools if t.name in allowed]
            async for ev in run_agent(working, llm=llm, tools=tools, mode="voice", actor_name=conv.actor_name,
                                      session_id=conv.id if negotiation_first else None):
                match ev:
                    case TextEvent(delta=delta) if hush or optimistic:
                        hushed += delta
                    case TextEvent(delta=delta):
                        if in_tools and after_results:  # first text of the next model step: were the writes quiet?
                            hush, in_tools, after_results = step_written and (step_asked or (step_said and step_logs)), False, False
                            hush_warned = hush and not step_quiet
                            if hush:
                                hushed += delta
                                continue
                        if timings.llm_first_token_ms is None:
                            timings.llm_first_token_ms = ms_since(llm_start)
                        if timings.useful_text_ms is None and delta.strip():
                            timings.useful_text_ms = ms_since(t0)
                        spoken += delta
                        step_text += delta
                        await self.send(AgentTextMessage(delta=delta))
                        speaker.push(delta)
                    case ToolCallEvent(call=call):
                        if not in_tools or after_results:  # first call of a step: did its text ask the caller something?
                            step_asked, step_quiet, in_tools, after_results, hush = "?" in step_text, True, True, False, False
                            step_written, step_said, step_logs = True, bool(step_text.strip()), True
                        step_logs = step_logs and call.name == "log_client_event"
                        if call.name in STATUS_ON_CALL:
                            await self._status(STATUS_ON_CALL[call.name])
                        if USE_FILLERS and not spoken.strip() and not speaker.queued_any:  # silence while tools run: a short filler now
                            filler = FILLERS[conv.fillers_used % len(FILLERS)]
                            conv.fillers_used += 1
                            spoken += f"{filler} "
                            await self.send(AgentTextMessage(delta=f"{filler} "))
                            speaker.filler(filler)
                            if timings.llm_first_token_ms is None:
                                timings.llm_first_token_ms = ms_since(llm_start)
                        tool_started[call.id] = ms_since(t0)
                        await self.send(AgentToolMessage(name=call.name, args=call.arguments))
                    case ToolResultEvent(call_id=call_id, name=name, ok=ok, ms=ms, result=result):
                        step_quiet, after_results = step_quiet and _quiet_result(name, ok, result), True
                        step_written = step_written and _write_ok(name, ok, result)
                        if name == "confirm_order" and ok and isinstance(result, dict) and not result.get("error"):
                            confirmed = result
                        timings.tools = [*(timings.tools or []), ToolTiming(name=name, start_ms=tool_started.get(call_id, max(0, ms_since(t0) - ms)), ms=ms)]  # fmt: skip
                        step_text = ""
                        await self.send(AgentToolMessage(name=name, ok=ok, ms=ms))
                        if ok:
                            args = next((c.arguments for m in working[::-1] for c in (m.tool_calls or []) if c.id == call_id), None)
                            await self._offer(board, [(ToolCall(id=call_id, name=name, arguments=args or {}), result, ms)])
                    case DoneEvent(usage=usage) if usage:
                        timings.llm_usage = usage
                    case ErrorEvent(message=message):
                        await self.send(VoiceErrorMessage(message=message))
            if optimistic:  # « Parfait, c'est noté. » is said: add the date once confirmed, or what the model needs to say
                if confirmed:
                    extra = " ".join(worth_saying_after_write(hushed))
                    when = f"Livraison {say_date(confirmed['deliveryDate'], today)}. " if confirmed.get("deliveryDate") else ""
                    extra = f"{extra} {when}Merci, bonne journée !".strip()
                else:  # not confirmed (a question, a change): the model's answer as written
                    extra = re.sub(r"^\s*(?:parfait|c'est noté|très bien)[ ,.!]*", "", hushed.strip(), flags=re.I)
                if extra:
                    spoken += f" {extra}"
                    await self.send(AgentTextMessage(delta=f" {extra}"))
                    speaker.push(f" {extra}")
            elif hush_warned and (extra := " ".join(worth_saying_after_write(hushed))):
                spoken += f" {extra}"
                await self.send(AgentTextMessage(delta=f" {extra}"))
                speaker.push(f" {extra}")
            conv.credit_said = conv.credit_said or bool(CREDIT_WORDS.search(spoken))
            ordering = any(t.name in ORDER_WRITES or t.name in ("find_product", "source_product") for t in timings.tools or []) or products_named(working)
            if not conv.credit_said and ordering and speaker.french and credit_warning(working) and not caller_blocked(working):
                conv.credit_said = True
                spoken += f" {CREDIT_SAY}"
                await self.send(AgentTextMessage(delta=f" {CREDIT_SAY}"))
                speaker.push(f" {CREDIT_SAY}")
            timings.llm_total_ms = ms_since(llm_start)
            conv.history = trim(working)
            committed = True

            # 3. Speak the last sentence and wait until all the audio is sent
            await speaker.finish()
            audio_timings(timings, t0, speaker)
            heard_text = spoken
            for sentence in speaker.dropped:  # reasoning out loud: written on screen by the model, never voiced
                heard_text = heard_text.replace(sentence, "")
            heard_text = customer_words(re.sub(r"\s+", " ", heard_text))
            await self.send(AgentDoneMessage(text=" ".join([*said_first, heard_text.strip()]).strip(), timings=timings))

        except asyncio.CancelledError:
            # Interrupted: keep what was heard, the tool rounds already done (an order the agent created must stay
            # known) and what the agent had started to say. Not when the turn was already committed (barge-in during
            # the last sentence's audio): that would duplicate it. trim() closes a tool call cut mid-run.
            if speaker:
                await speaker.stop()
            voiced = speaker.voiced_text() if speaker else ""
            if heard and not committed:
                kept = working if working is not None else [*conv.history, ChatMessage(role="user", content=heard)]
                last = kept[-1]
                said = heard_part(step_text, voiced)
                if said and not (last.role == "assistant" and last.tool_calls):
                    kept = [*kept, ChatMessage(role="assistant", content=f"{said} …")]
                conv.history = trim(kept)
            elif committed and speaker and not speaker.done and conv.history and conv.history[-1].role == "assistant":
                # Cut while the final answer was being spoken: remember only what the caller actually heard.
                final = conv.history[-1]
                said = heard_part(final.content, voiced)
                if said != final.content.strip():
                    conv.history[-1] = final.model_copy(update={"content": f"{said} … (interrompu)" if said else "(interrompu avant de répondre)"})
            raise
        except Exception as e:
            await self.send(VoiceErrorMessage(message=str(e) or type(e).__name__))
        finally:
            if ack and not ack.done():
                ack.cancel()
            if utterance:
                utterance.cancel()  # interrupted while transcribing: close the provider stream

    async def _show_injected(self, injected: Injected) -> None:
        """The tool calls a fast path made, on the Voix page like the model's own."""
        for call, _, ms in injected.calls:
            await self.send(AgentToolMessage(name=call.name, args=call.arguments))
            await self.send(AgentToolMessage(name=call.name, ok=True, ms=ms))

    async def _status(self, label: str) -> None:
        await self.send(AgentStatusMessage(label=label))

    async def _offer(self, board: OfferBoard, calls) -> None:
        """Customer-facing offer card from tool results (never supplier names, costs or ids)."""
        changed = False
        for call, result, _ in calls:
            if negotiation_enabled() and call.name != "negotiate_basket":
                continue  # never present a catalog lookup as the negotiated customer price
            changed |= board.add(call.name, call.arguments, result)
        if changed:
            await self.send(AgentOfferMessage(lines=board.snapshot()))
            if any(line.unit_price is not None for line in board.lines.values()):
                await self._status("Meilleure offre trouvée.")

    async def _ack_after(self, speaker: "Speaker", conv: Conversation, said: list[str]) -> None:
        """After ACK_AFTER_S, if the agent has said nothing yet (no text, nothing queued), play an acknowledgement."""
        await asyncio.sleep(ACK_AFTER_S)
        if speaker.queued_any or speaker.text_seen:
            return
        phrase = FIRST_ACK if not conv.history else ACKS[conv.fillers_used % len(ACKS)]
        conv.fillers_used += 1
        said.append(phrase)
        speaker.filler(phrase)
        await self.send(AgentTextMessage(delta=f"{phrase} "))


class Speaker:
    """Buffers the streamed text, cuts it into sentences and speaks them in order, as one audio segment."""

    # a sentence ends with . ! ? … followed by a space ("18.50" is not a boundary)
    SENTENCE = re.compile(r"^([\s\S]*?[.!?…])\s+")
    # the FIRST chunk of a reply may end at a comma / colon once it is long enough: first sound sooner
    FIRST_CLAUSE = re.compile(r"^([\s\S]{28,}?[,;:])\s+")

    def __init__(
        self,
        tts: TTSClient,
        send: Callable[[WireModel], Awaitable],
        send_bytes: Callable[[bytes], Awaitable],
        today: str | None = None,  # the ERP's business date, for « demain » in spoken dates
    ):
        self.tts, self.send, self.send_bytes, self.today = tts, send, send_bytes, today
        self.buffer = ""
        # queued items: (text as written, audio source): a str to synthesise, or pre-rendered (mime, bytes)
        self.sentences: asyncio.Queue[tuple[str, str | tuple[str, bytes], bool] | None] = asyncio.Queue()
        self.worker: asyncio.Task | None = None  # speaks the queued sentences one after the other
        self.started = False
        self.done = False  # all the audio was sent
        self.queued_any = False
        self.text_seen = False  # the model started its answer (the deadline acknowledgement must not cut in)
        self.voiced: list[tuple[str, bool]] = []  # (text, is_filler) whose audio started playing
        self.first_byte_at: float | None = None
        self.tts_first_byte_ms: int | None = None
        self.first_ack_at: float | None = None  # first audio byte of an acknowledgement / filler sent
        self.first_useful_at: float | None = None  # first audio byte of the real answer sent
        self.dropped: list[str] = []  # sentences not voiced: the model narrating its reasoning (META_TALK)
        self.french = True  # the caller speaks French: an English sentence is the model thinking aloud
        self.drop_open = False  # the last clause dropped did not end its sentence
        self.said_words: list[set[str]] = []  # content words of each sentence said this turn (no saying it twice)

    def push(self, delta: str) -> None:
        self.text_seen = self.text_seen or bool(delta.strip())
        self.buffer += delta
        if not self.queued_any and (m := self.FIRST_CLAUSE.match(self.buffer)) and not self.SENTENCE.match(self.buffer):
            self.speak(m[1])
            self.buffer = self.buffer[m.end() :]
        while m := self.SENTENCE.match(self.buffer):
            self.speak(m[1])
            self.buffer = self.buffer[m.end() :]

    def speak(self, sentence: str) -> None:
        if not sentence.strip():
            return
        if self.drop_open or META_TALK.search(sentence) or (self.french and english_aside(sentence)) or self._repeats(sentence):
            # a clause cut at a comma was dropped: the rest of that sentence goes too, or the caller hears a fragment
            # (03:58 « large quantities. », 04:04 « et ses jours de livraison sont lundi… »)
            self.drop_open = not re.search(r"[.!?…]\s*$", sentence)
            self.dropped.append(sentence.strip())
            return
        sentence = customer_words(sentence)
        if self.french:  # « Reblochon fermier or laitier ? » (04:04, s13)
            sentence = re.sub(r"(?<=\w) or (?=\w)", " ou ", sentence)
        spoken = to_speech(sentence, self.today) or sentence
        self._queue((sentence.strip(), spoken, False))

    def _repeats(self, sentence: str) -> bool:
        """Said already in this turn, in other words (s02 mar 03:52: « … en rupture jusqu'en janvier. » then « Désolé,
        le beaufort d'été est épuisé jusqu'en janvier. »): 75 % of its content words were in one earlier sentence."""
        words = content_words(sentence)
        if len(words) >= 3:
            if any(len(words & seen) >= 0.75 * min(len(words), len(seen)) for seen in self.said_words):
                return True
            self.said_words.append(words)
        return False

    def filler(self, text: str) -> None:
        """A waiting phrase: the pre-rendered audio if there is one, else synthesised like any sentence."""
        audio = _filler_audio.get((self.tts.name, settings.tts.voice or None, text))
        self._queue((text, audio or text, True))

    def _queue(self, item: tuple[str, str | tuple[str, bytes], bool]) -> None:
        self.queued_any = True
        if self.worker is None:
            self.worker = asyncio.create_task(self._speak_queued())
        self.sentences.put_nowait(item)

    def voiced_text(self, fillers: bool = False) -> str:
        return " ".join(text for text, is_filler in self.voiced if fillers or not is_filler)

    async def finish(self) -> None:
        """Speak what is left in the buffer, wait for all the audio, close the segment."""
        self.speak(self.buffer)
        self.buffer = ""
        if self.worker:
            self.sentences.put_nowait(None)
            await self.worker
        self.done = True
        if self.started:
            await self.send(AgentAudioEndMessage())

    async def stop(self) -> None:
        """Interrupted: drop the queued sentences, close the segment if audio was sent."""
        if self.worker:
            self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker
        if self.started:
            await self.send(AgentAudioEndMessage())

    async def _speak_queued(self) -> None:
        while (item := await self.sentences.get()) is not None:
            text, source, is_filler = item
            tts_start = time.perf_counter()
            try:
                if isinstance(source, tuple):  # pre-rendered filler: no TTS round-trip at all
                    mime, audio = source
                    await self._audio_start(mime, tts_start)
                    await self._mark(is_filler)
                    self.voiced.append((text, is_filler))
                    await self.send_bytes(audio)
                    continue
                result = await self.tts.synthesize(source, settings.tts.voice or None)
                # aclosing: on barge-in (cancellation) the provider's HTTP stream is closed right away
                first = True
                async with aclosing(result.chunks) as chunks:  # type: ignore[type-var]
                    async for chunk in chunks:
                        if first:
                            first = False
                            await self._audio_start(result.mime, tts_start)
                            await self._mark(is_filler)
                            self.voiced.append((text, is_filler))
                        await self.send_bytes(chunk)
            except Exception as e:
                await self.send(VoiceErrorMessage(message=f"TTS : {e}"))

    async def _mark(self, is_filler: bool) -> None:
        """Before the first bytes of a sentence: note when the first acknowledgement / useful audio left, and tell the
        browser where the useful answer starts in the audio segment (agent_audio_mark)."""
        now = time.perf_counter()
        if is_filler:
            self.first_ack_at = self.first_ack_at or now
        elif self.first_useful_at is None:
            self.first_useful_at = now
            await self.send(AgentAudioMarkMessage())

    async def _audio_start(self, mime: str, tts_start: float) -> None:
        if not self.started:
            self.started = True
            await self.send(AgentAudioStartMessage(mime=mime))
            self.first_byte_at = time.perf_counter()
            self.tts_first_byte_ms = ms_since(tts_start)


# ── timings (latency diagnosis) ──


def _ms(at: float | None, t0: float) -> int | None:
    return None if at is None else int((at - t0) * 1000 + 0.5)


def stt_timings(timings: TurnTimings, t0: float, utterance: Utterance) -> None:
    """Where the speech-to-text time went: stream opening, audio still queued at the end, finalize, fallback."""
    if utterance.stream_ready_at is not None:
        timings.stt_connect_ms = _ms(utterance.stream_ready_at, utterance.started_at)
    timings.stt_backlog_frames = utterance.backlog_frames
    timings.stt_finalize_sent_ms = _ms(utterance.finalize_sent_at, t0)
    timings.stt_fallback = utterance.fallback


def add_injected_timings(timings: TurnTimings, t0: float, injected: Injected) -> None:
    now = ms_since(t0)
    timings.tools = [*(timings.tools or []), *(ToolTiming(name=c.name, start_ms=max(0, now - ms), ms=ms) for c, _, ms in injected.calls)]


def audio_timings(timings: TurnTimings, t0: float, speaker: "Speaker") -> None:
    """First audio byte (acknowledgement or answer), and the acknowledgement / useful answer separately. Logged."""
    timings.tts_first_byte_ms = speaker.tts_first_byte_ms
    timings.total_ms = _ms(speaker.first_byte_at, t0)
    timings.ack_audio_ms = _ms(speaker.first_ack_at, t0)
    timings.useful_audio_ms = _ms(speaker.first_useful_at, t0)
    print(f"[timing] {timings.to_json()}", flush=True)


def heard_part(text: str, voiced: str) -> str:
    """The beginning of `text` the caller actually heard: its sentences found in what was voiced, up to the first
    one that was not (the rest was generated but cut by the barge-in)."""
    out = []
    for sentence in re.split(r"(?<=[.!?…,;:])\s+", (text or "").strip()):
        if sentence and sentence in voiced:
            out.append(sentence)
        else:
            break
    return " ".join(out)


def trim(history: list[ChatMessage], max_messages: int = 40) -> list[ChatMessage]:
    """Keep the last ~40 messages, starting on a user message (never split a tool call from its result)."""
    history = sanitize_history(history)  # no dangling tool call (cancelled mid-tool), no orphan result
    if len(history) <= max_messages:
        return history
    start = next((i for i, m in enumerate(history) if i >= len(history) - max_messages and m.role == "user"), -1)
    return history[start:] if start > 0 else history

