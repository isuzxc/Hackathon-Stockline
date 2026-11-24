"""
The offer card of the Voix page: what the side agents found, as the CUSTOMER may see it. Built only from tool results
(find_product, identify_caller, source_product): nothing is invented, a missing value stays empty. Never shown: supplier
names, purchase prices, margins, the negotiation, internal ids.
"""

from datetime import date

from app.schemas import OfferLine

_DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]  # fmt: skip
UNIT_SHORT = {"piece": "pièces", "box": "cartons", "liter": "L", "dozen": "douzaines", "kg": "kg"}


def day(iso: str | None, today: str | None = None) -> str | None:
    """'2026-11-25' → 'demain (mercredi 25 novembre)' / 'jeudi 26 novembre'."""
    if not iso:
        return None
    d = date.fromisoformat(str(iso)[:10])
    text = f"{_DAYS[d.weekday()]} {d.day} {_MONTHS[d.month - 1]}"
    if today and (d - date.fromisoformat(str(today)[:10])).days == 1:
        return f"demain ({text})"
    return text


def _num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") if float(x).is_integer() else f"{x:,.1f}".replace(",", " ").replace(".", ",")


def _short(name: str) -> str:
    return (name or "").split(" (")[0].strip()


class OfferBoard:
    """The lines of the current turn, keyed by product. A later result for the same product completes its line."""

    def __init__(self, today: str | None = None, delivery: str | None = None):
        self.today = today
        self.delivery = delivery  # the caller's next delivery date (identify_caller), when known
        self.lines: dict[str, OfferLine] = {}
        self._ids: dict[str, str] = {}  # product id → key, to attach source_product results

    def add(self, name: str, args: dict | None, result: object) -> bool:
        """Read one tool result. True if the card changed."""
        if not isinstance(result, dict):
            return False
        args = args or {}
        if name == "identify_caller":
            if result.get("found"):
                self.delivery = (result.get("delivery") or {}).get("next") or self.delivery
            asked = [p for p in (args.get("products") or []) if isinstance(p, dict)]
            changed = False
            for i, p in enumerate(result.get("products") or []):
                ask = asked[i] if i < len(asked) else {}
                changed |= self._product(ask.get("qty"), p)
            return changed
        if name == "find_product":
            return self._product(args.get("qty"), result)
        if name == "source_product":
            return self._sourced(args, result)
        if name == "negotiate_basket":
            self.lines.clear()
            if not result.get("found"):
                return True  # withdraw a previous offer if the replacement cannot be quoted
            for item in result.get("lines", []):
                label = item["product"]
                self.lines[label] = OfferLine(product=label, qty=item["qty"], unit=item["unit"],
                    unit_price=item["unitPrice"], total=item["total"],
                    availability="Offre négociée (simulation)" if result.get("mode") == "demo" else "Offre négociée",
                    delivery=day(result.get("deliveryDate"), self.today))
            return True
        return False

    def _product(self, qty: float | None, r: dict) -> bool:
        if r.get("found") and r.get("product"):
            p = r["product"]
            key = p.get("say") or _short(p.get("name", ""))
            price = (r.get("price") or {}).get("unitPrice")
            stock = r.get("stock") or {}
            if qty and stock.get("enough") is False:
                have = stock.get("available") or 0
                availability = f"{_num(have)} {UNIT_SHORT.get(p.get('unit'), p.get('unit') or '')} en stock" if have > 0 else "Rupture de stock"  # fmt: skip
            elif stock.get("available") is not None:
                availability = "En stock"
            else:
                availability = "À vérifier"
            self.lines[key] = OfferLine(
                product=key, qty=qty, unit=p.get("unit"), unit_price=price,
                total=round(qty * price, 2) if qty and price is not None else None,
                availability=availability, delivery=day(self.delivery, self.today) if qty else None,
            )  # fmt: skip
            self._ids[p.get("id", "")] = key
            return True
        options = [_short(c.get("name", "")) for c in (r.get("candidates") or []) if c.get("name")]
        options = [o for o in options if o.split(" ")[0] == options[0].split(" ")[0]] if options else []  # same product family
        if options and "productId" in (r.get("candidates") or [{}])[0]:
            key = " / ".join(options[:2])
            self.lines[key] = OfferLine(product=options[0].split(" ")[0] if options else "Produit", availability="À préciser", options=options[:3])  # fmt: skip
            return True
        return False

    def _sourced(self, args: dict, r: dict) -> bool:
        key = self._ids.get(args.get("productId", ""))
        line = self.lines.get(key) if key else None
        if line is None or r.get("needed") is False:
            return False
        unit = UNIT_SHORT.get(line.unit or "", line.unit or "")
        offers = r.get("offers") or []
        latest = max((o.get("arrives") for o in offers if o.get("arrives")), default=None)
        have = r.get("available") or 0
        if not offers:
            availability = f"{_num(have)} {unit} en stock, pas de réassort possible" if have > 0 else "Indisponible"
            if r.get("reason"):  # the whole trade is out (season, vintage): said as is, it is no secret
                availability += f" : {r['reason']}"
        else:
            rest = f"le reste arrive {day(latest, self.today)}" if latest else "réassort en cours"
            availability = f"{_num(have)} {unit} en stock, {rest}" if have > 0 else f"Réassort, arrivée {day(latest, self.today)}"
            if r.get("uncovered"):
                availability += f" (manque {_num(r['uncovered'])} {unit})"
        delivery = line.delivery
        if not r.get("inTime") and latest:  # nobody can deliver in time: the day after the last arrival at the earliest
            delivery = f"au plus tôt le lendemain du {day(latest)}"
        self.lines[key] = line.model_copy(update={"availability": availability, "delivery": delivery})
        return True

    def snapshot(self) -> list[OfferLine]:
        return list(self.lines.values())
