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

export interface Dashboard {
  symbols: string[];
  signals: Signal[];
  risk: RiskLimits;
}
