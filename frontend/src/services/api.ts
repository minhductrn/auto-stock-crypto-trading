import type { Candle, Dashboard, OrderRequest, OrderResult } from "../types";
import { mockCandles, mockDashboard } from "./mock";

/** When VITE_API_URL is unset the UI runs on deterministic mock data. */
const BASE_URL = import.meta.env.VITE_API_URL;
const observationMode = new URLSearchParams(window.location.search).get("mode") === "observation";
let observation: Promise<{ dashboard: Dashboard; candles: Record<string, Candle[]> }> | undefined;

function loadObservation() {
  return (observation ??= getJson("/observation.json"));
}

/** True when a live backend is configured (order execution is only offered then). */
export const isLiveApi = Boolean(BASE_URL) && !observationMode;

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (body.detail !== undefined) return JSON.stringify(body.detail);
  } catch {
    /* non-JSON error body */
  }
  return `status ${res.status}`;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${observationMode ? "" : BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${await errorDetail(res)}`);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${await errorDetail(res)}`);
  }
  return (await res.json()) as T;
}

export async function fetchDashboard(): Promise<Dashboard> {
  if (observationMode) return (await loadObservation()).dashboard;
  return BASE_URL ? getJson<Dashboard>("/dashboard") : mockDashboard;
}

export async function fetchCandles(symbol: string): Promise<Candle[]> {
  if (observationMode) return (await loadObservation()).candles[symbol] ?? [];
  return BASE_URL
    ? getJson<Candle[]>(`/candles/${encodeURIComponent(symbol)}`)
    : mockCandles(symbol);
}

/**
 * Submit a paper bracket order via the backend (capped at 1% account risk server-side).
 * Requires VITE_API_URL; there is deliberately no mock fallback for order placement.
 */
export async function executeOrder(req: OrderRequest): Promise<OrderResult> {
  if (!isLiveApi) {
    throw new Error("Order execution requires API mode.");
  }
  return postJson<OrderResult>("/orders/execute", req);
}
