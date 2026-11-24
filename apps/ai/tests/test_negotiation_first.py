from types import SimpleNamespace
from app.services.negotiation import Basket, approval_matches, customer_price
from app.schemas import ChatMessage


def test_customer_gets_half_savings_without_losing_margin():
    assert customer_price(12, 8, 70, 80, 10) == 11.5
    assert customer_price(12, 8, 90, 100, 10) is None
    assert customer_price(12, 8, 80, 80, 10) == 12


def test_approval_is_for_exact_latest_recap():
    b = Basket("id", "client", "Client", "2026-11-27", [], "demo")
    b.say = "Total 100 euros. Acceptez-vous cette offre ?"
    history = [ChatMessage(role="assistant", content=b.say), ChatMessage(role="user", content="oui")]
    assert approval_matches(history, b)
    history[-1] = ChatMessage(role="user", content="oui mais ajoute 2 kilos")
    assert not approval_matches(history, b)
    history[-1] = ChatMessage(role="user", content="oui")
    history[0] = ChatMessage(role="assistant", content="Vous voulez le fermier ?")
    assert not approval_matches(history, b)


async def test_basket_negotiates_even_with_stock_and_only_writes_after_acceptance(monkeypatch):
    from datetime import date, datetime, timedelta, timezone
    from app.services import negotiation as n
    from app.market.strategy import SupplierProfile, Offer

    class ERP:
        def __init__(self):
            self.posts = []

        async def get(self, path, query=None):
            if path == "/api/clients/client":
                return {"id": "client", "name": "Client", "status": "active", "zone": "z",
                        "deliveryDays": ["fri"], "creditLimit": 5000, "outstanding": 0}
            if path == "/api/routes":
                return [{"zone": "z", "days": ["fri"]}]
            if path == "/api/purchase-orders/sources":
                return {"name": "Tomme", "unit": "kg", "category": "cheese", "cost": 8,
                        "available": 20, "suppliers": [{"supplierId": "a", "price": 8},
                                                       {"supplierId": "b", "price": 9}]}
            if path == "/api/context/for":
                return {"entries": []}
            if path == "/api/prices/resolve":
                return {"price": 12}
            raise AssertionError(path)

        async def post(self, path, body):
            self.posts.append((path, body))
            return {"id": str(len(self.posts)), "number": str(len(self.posts))}

    async def meta():
        return {"today": "2026-11-24"}

    async def profile(erp, sid):
        return SupplierProfile(sid, sid)

    requested = []

    async def run(line, profiles, today, deadline, seed, **kwargs):
        requested.append((line.need, set(profiles)))
        q = n.live_agents.Quote(Offer("a", line.product_id, 7, 10, True, 8),
                               date(2026, 11, 26), datetime.now(timezone.utc) + timedelta(minutes=2), "q")
        return {sid: SimpleNamespace(history=[]) for sid in profiles}, [q], None

    monkeypatch.setattr(n, "get_meta", meta)
    monkeypatch.setattr(n, "cached_profile", profile)
    monkeypatch.setattr(n.live_agents, "run", run)
    monkeypatch.setattr(n.live_agents, "mode", lambda: "demo")
    monkeypatch.setattr(n, "BASKETS", {})
    monkeypatch.setattr(n, "ROUNDS", {})
    erp = ERP()
    ctx = SimpleNamespace(erp=erp, session_id="session")
    out = await n.quote_basket(ctx, "client", [{"productId": "p", "qty": 10}], "2026-11-27")
    assert out["found"] and out["total"] == 115 and out["agents"] == 2
    assert requested == [(10, {"a", "b"})] and not erp.posts
    assert not n.approval_matches([ChatMessage(role="user", content="oui")], n.pending("session"))
    assert n.approval_matches([ChatMessage(role="assistant", content=out["say"]),
                               ChatMessage(role="user", content="oui")], n.pending("session"))
    accepted = await n.accept_basket(ctx, out["quoteId"])
    assert accepted["accepted"]
    assert erp.posts[0][1]["lines"][0]["unitPrice"] == 11.5
    assert erp.posts[1][1]["lines"][0]["unitPrice"] == 7
    assert (await n.accept_basket(ctx, out["quoteId"]))["alreadyAccepted"]
    assert len(erp.posts) == 2
