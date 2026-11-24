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
