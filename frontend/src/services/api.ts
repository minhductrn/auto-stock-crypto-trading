import type { Candle, Dashboard } from "../types";
import { mockCandles, mockDashboard } from "./mock";

/** When VITE_API_URL is unset the UI runs on deterministic mock data. */
const BASE_URL = import.meta.env.VITE_API_URL;

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed with status ${res.status}`);
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
