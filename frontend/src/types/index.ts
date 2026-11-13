export interface Candle {
  time: string; // session_date, YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** Mirrors the signals schema (spec 8.7), camelCased. */
export interface Signal {
  symbol: string;
  pXgb: number;
  pLgbm: number;
  pLong: number;
  rank: number;
  targetWeight: number;
}

/** Mirrors RiskConfig in risk_management/config.py. */
export interface RiskLimits {
  pThreshold: number;
  topK: number;
  minNames: number;
  maxWeight: number;
  turnoverCap: number;
  minPrice: number;
  minAdv20Dollar: number;
  costBpsPerSide: number;
}

export interface BlockedMlEntry {
  symbol?: string;
  reason: string;
  pXgb?: number | null;
  pLgbm?: number | null;
}

export interface Dashboard {
  symbols: string[];
  signals: Signal[];
  risk: RiskLimits;
  /** Optional extras supplied by the live API (absent in mock mode). */
  accountEquity?: number | null;
  modelStatus?: string;
  blockedMlEntries?: BlockedMlEntry[];
  asOf?: string | null;
}

/** Body of POST /orders/execute. */
export interface OrderRequest {
  symbol: string;
  side: "buy" | "sell";
  notional_or_qty: number;
  unit?: "notional" | "qty";
}

/** Response of POST /orders/execute. */
export interface OrderResult {
  orderId: string;
  clientOrderId: string;
  symbol: string;
  side: string;
  qty: number;
  status: string;
  entryPrice?: number | null;
  stopPrice?: number | null;
  takeProfitPrice?: number | null;
  riskDollars?: number | null;
  riskPctOfEquity?: number | null;
  cappedBy: string[];
  message: string;
}
