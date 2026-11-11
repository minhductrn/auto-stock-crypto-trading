import type { Candle, Dashboard } from "../types";

/** Deterministic pseudo-random generator so the demo chart is stable across reloads. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function hash(text: string): number {
  return [...text].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) >>> 0, 7);
}

export function mockCandles(symbol: string, sessions = 60): Candle[] {
  const rand = lcg(hash(symbol));
  const candles: Candle[] = [];
  let price = 100 + rand() * 100;
  const day = new Date(Date.UTC(2024, 0, 2));
  while (candles.length < sessions) {
    const dow = day.getUTCDay();
    if (dow !== 0 && dow !== 6) {
      const open = price;
      const close = open * (1 + (rand() - 0.5) * 0.04);
      const high = Math.max(open, close) * (1 + rand() * 0.01);
      const low = Math.min(open, close) * (1 - rand() * 0.01);
      candles.push({
        time: day.toISOString().slice(0, 10),
        open, high, low, close,
        volume: Math.round(1e6 + rand() * 5e6),
      });
      price = close;
    }
    day.setUTCDate(day.getUTCDate() + 1);
  }
  return candles;
}

export const mockDashboard: Dashboard = {
  symbols: ["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
  signals: [
    { symbol: "NVDA", pXgb: 0.64, pLgbm: 0.62, pLong: 0.63, rank: 1, targetWeight: 0.05 },
    { symbol: "MSFT", pXgb: 0.6, pLgbm: 0.59, pLong: 0.595, rank: 2, targetWeight: 0.05 },
    { symbol: "AAPL", pXgb: 0.57, pLgbm: 0.56, pLong: 0.565, rank: 3, targetWeight: 0.047 },
    { symbol: "JPM", pXgb: 0.52, pLgbm: 0.51, pLong: 0.515, rank: 4, targetWeight: 0 },
    { symbol: "XOM", pXgb: 0.48, pLgbm: 0.5, pLong: 0.49, rank: 5, targetWeight: 0 },
  ],
  risk: {
    pThreshold: 0.55,
    topK: 20,
    minNames: 5,
    maxWeight: 0.05,
    turnoverCap: 0.3,
    minPrice: 5,
    minAdv20Dollar: 20_000_000,
    costBpsPerSide: 5,
  },
};
