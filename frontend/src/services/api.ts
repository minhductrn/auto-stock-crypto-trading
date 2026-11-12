import type { Candle, Dashboard, OrderRequest, OrderResult } from "../types";
import { mockCandles, mockDashboard } from "./mock";

/** When VITE_API_URL is unset the UI runs on deterministic mock data. */
const BASE_URL = import.meta.env.VITE_API_URL;

/** True when a live backend is configured (order execution is only offered then). */
export const isLiveApi = Boolean(BASE_URL);

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
  const res = await fetch(`${BASE_URL}${path}`);
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
  return BASE_URL ? getJson<Dashboard>("/dashboard") : mockDashboard;
}

export async function fetchCandles(symbol: string): Promise<Candle[]> {
  return BASE_URL
    ? getJson<Candle[]>(`/candles/${encodeURIComponent(symbol)}`)
    : mockCandles(symbol);
}

/**
 * Submit a paper bracket order via the backend (capped at 1% account risk server-side).
 * Requires VITE_API_URL; there is deliberately no mock fallback for order placement.
 */
export async function executeOrder(req: OrderRequest): Promise<OrderResult> {
  if (!BASE_URL) {
    throw new Error("VITE_API_URL is not set; order execution is disabled in mock mode");
  }
  return postJson<OrderResult>("/orders/execute", req);
}
