# Voix — negotiation-first

Voice session uses a restricted quote-first tool set by default. Existing chat and ERP voice-command are unchanged.
Set VOIX_NEGOTIATION_FIRST=0 in .env to restore the original voice flow.

- identify_caller and find_product resolve identities and references, without automatic procurement.
- negotiate_basket compares stock with parallel supplier-agent quotes for all requested products. No ERP writes.
- Each supplier has an independent buyer agent. Agents request quotes, share verified comparable offers and
  choose counteroffer tactics in rounds. Supplier replies determine prices, not the customer-facing LLM.
- A deterministic optimizer selects a feasible allocation with lowest procurement cost. Customer price uses the
  ERP baseline, shares half the savings on selected allocations, and protects the baseline nominal margin.
  If no feasible offer exists, the customer must change date/quantity rather than receive a fabricated price.
- The service reads the exact recap and displays the customer-facing quote. A short affirmative response to that
  exact recap enables accept_negotiated_offer. Changes require a new quote.
- Acceptance writes a customer draft and supplier drafts with negotiated amounts; no real supplier commitment.
  The quote is session-scoped and expires after three minutes. Ambiguous write failures require manual review.

## Run

Default MARKET_NEGOTIATION_MODE=demo uses fictional shop actors. The spoken quote says Simulation.
For real shop agents use MARKET_NEGOTIATION_MODE=external and MARKET_SUPPLIER_ENDPOINTS as a JSON
supplier-ID-to-HTTPS-URL map. Only configured endpoints are contacted. There is no automatic fallback to fake shops.
See live_agents.py for the non-binding HTTP request and structured quote response schema.
Use a real configured LLM for Voix; the old mock LLM is not a natural language negotiation orchestrator.

Restart the AI service after editing .env. Test with a known customer, exact product, quantity and valid route day.
Check that the ERP remains unchanged during negotiation and that a short "oui" creates drafts only at the accepted price.
Run cd apps/ai && uv run pytest. The old voice path is available for regression comparison.

The existing market dashboard now streams real request/quote events per product, including withdrawn quotes and the final procurement allocation. For a multi-product basket the dashboard follows the latest product round; the voice offer card displays the entire basket. The trace represents quote selection, not supplier acceptance.

An exact short approval triggers acceptance deterministically before calling the LLM again. Preliminary catalog prices are no longer displayed as final offers in negotiation-first mode.
