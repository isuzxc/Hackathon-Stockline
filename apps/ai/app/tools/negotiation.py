"""Tools for the negotiation-first voice conversation."""
from pydantic import Field
from .registry import ToolArgs, ToolContext, tool


class RequestedLine(ToolArgs):
    product_id: str
    qty: float = Field(gt=0)


@tool
async def negotiate_basket(ctx: ToolContext, client_id: str, lines: list[RequestedLine],
                           delivery_date: str | None = None) -> dict:
    """Négocie le PANIER COMPLET (produits identifiés, quantités dans les unités ERP, date souhaitée).
    Appelle les agents fournisseurs en parallèle même si du stock interne existe.
    Lecture/devis non engageants uniquement. Renvoie le tarif client et le récapitulatif exact, jamais le prix d'achat.
    """
    from app.services.negotiation import quote_basket
    return await quote_basket(ctx, client_id, [l.model_dump(by_alias=True) for l in lines], delivery_date)


@tool
async def accept_negotiated_offer(ctx: ToolContext, quote_id: str) -> dict:
    """Enregistre uniquement la dernière offre négociée après un oui explicite à son récapitulatif.
    Ne transmet pas de commande ferme au fournisseur : crée les brouillons ERP au prix accepté.
    """
    from app.services.negotiation import accept_basket
    return await accept_basket(ctx, quote_id)
