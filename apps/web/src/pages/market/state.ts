/**
 * The trading floor's state, rebuilt from the round's events (GET /ai/market/rounds/{id}/events). Pure: a page opened
 * mid-round replays the events and shows the same thing; unit-tested in tests/market-state.test.ts.
 */

export interface MarketEvent { type: string; t: number; call?: string | null; data: Record<string, any> }

export type CallStatus = "idle" | "skipped" | "ringing" | "talking" | "waiting" | "won" | "lost";

export interface Line { t: number; speaker: "buyer" | "supplier"; text: string }
export interface OfferView { productId: string; price: number; qty: number; final: boolean; regular: boolean; opening: number }

export interface CallView {
  id: string;
  name: string;
  city?: string;
  contact?: string;
  reliability?: number;
  flexibility?: number;
  status: CallStatus;
  reason?: string; // skipped
  lines: Line[];
  offers: Record<string, OfferView>; // product id → latest offer
  intel: { t: number; fromCall: string; productId: string; price: number }[]; // competitor prices used against this supplier
  levers: { t: number; lever: string; label: string }[]; // arguments the buyer used on this call (learned per supplier)
  start?: number;
  end?: number;
  wonTotal?: number;
  po?: { id: string; number?: string; total?: number };
}

export interface NeedView { productId: string; name: string; unit: string; category: string; need: number; listPrice: number; suppliers: string[] }

export interface Allocation { supplierId: string; productId: string; qty: number; price: number }

export interface RoundView {
  id?: string;
  title?: string;
  reference?: string;
  needBy?: string;
  needs: NeedView[];
  calls: CallView[]; // suppliers in ERP order (called and skipped)
  decision?: { allocations: Allocation[]; uncovered: Record<string, number>; total: number; reference: number; savings: number; needed: number; covered: number };
  summary?: {
    text: string; total: number; savings: number; savingsPct: number; suppliers: number; called: number; coverage: number; coverageRegular: number;
    metrics?: { timeToDecisionMs?: number; timeToOrdersCreatedMs?: number; timeToAllDialoguesFinishedMs?: number; exchanges?: number;
      voices?: { llmCalls: number; cacheHits: number; templates: number; fallbacks: number; medianLatencyMs?: number | null } };
  };
  error?: string;
  t: number; // time of the last event (ms since the start)
  done: boolean;
}

export const emptyRound = (): RoundView => ({ needs: [], calls: [], t: 0, done: false });

export function applyEvent(r: RoundView, e: MarketEvent): RoundView {
  const t = Math.max(r.t, e.t);
  const call = (id: string | null | undefined, f: (c: CallView) => CallView): CallView[] =>
    r.calls.map((c) => (c.id === id ? f(c) : c));
  const d = e.data ?? {};
  switch (e.type) {
    case "round_started":
      return {
        ...r, t, title: d.title, reference: d.reference, needBy: d.needBy, needs: d.needs ?? [],
        calls: (d.suppliers ?? []).map((s: any) => ({
          id: s.id, name: s.name, city: s.city, contact: s.contact, reliability: s.reliability, flexibility: s.flexibility,
          status: "idle" as CallStatus, lines: [], offers: {}, intel: [], levers: [],
        })),
      };
    case "call_skipped":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, status: "skipped", reason: d.reason })) };
    case "call_ringing":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, status: "ringing", start: e.t })) };
    case "call_connected":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, status: "talking" })) };
    case "line":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, lines: [...c.lines, { t: e.t, speaker: d.speaker, text: d.text }] })) };
    case "offer":
      return {
        ...r, t, calls: call(e.call, (c) => {
          const offers = { ...c.offers };
          if (d.withdrawnProductId) delete offers[d.withdrawnProductId];
          for (const o of d.offers ?? []) offers[o.productId] = { ...o, opening: c.offers[o.productId]?.opening ?? o.price };
          return { ...c, offers };
        }),
      };
    case "lever":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, levers: [...c.levers, { t: e.t, lever: d.lever, label: d.label }] })) };
    case "intel":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, intel: [...c.intel, { t: e.t, fromCall: d.fromCall, productId: d.productId, price: d.price }] })) };
    case "call_waiting":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, status: "waiting" })) };
    case "decision":
      return { ...r, t, decision: d as RoundView["decision"] };
    case "call_ended":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, status: d.won ? "won" : "lost", end: e.t, wonTotal: d.total })) };
    case "po_created":
      return { ...r, t, calls: call(e.call, (c) => ({ ...c, po: { id: d.poId, number: d.number, total: d.total } })) };
    case "summary":
      return { ...r, t, summary: d as RoundView["summary"], done: true };
    case "error":
      return { ...r, t, error: d.message };
    default:
      return { ...r, t };
  }
}

export const replay = (events: MarketEvent[]): RoundView => events.reduce(applyEvent, emptyRound());

/** Calls on the phone right now (ringing, talking or waiting for the decision). */
export const liveCalls = (r: RoundView) => r.calls.filter((c) => c.status === "ringing" || c.status === "talking" || c.status === "waiting").length;

/** Offers per product, best first, with what the decision allocated (the « tableau des offres »). */
export function board(r: RoundView) {
  return r.needs.map((n) => {
    const rows = r.calls
      .filter((c) => c.offers[n.productId])
      .map((c) => {
        const o = c.offers[n.productId];
        const alloc = r.decision?.allocations.find((a) => a.supplierId === c.id && a.productId === n.productId);
        return { callId: c.id, name: c.name, price: o.price, opening: o.opening, qty: o.qty, final: o.final, regular: o.regular, allocated: alloc?.qty };
      })
      .sort((a, b) => a.price - b.price);
    return { need: n, rows, uncovered: r.decision?.uncovered?.[n.productId] };
  });
}

/** Savings so far: what the suppliers first asked − their current price, on their whole offered quantity (live), then
 *  the decision's exact figure once it is known. */
export function savingsSoFar(r: RoundView): number {
  if (r.decision) return r.decision.savings;
  return r.calls.reduce((sum, c) => sum + Object.values(c.offers).reduce((s, o) => s + (o.opening - o.price) * o.qty, 0), 0);
}
