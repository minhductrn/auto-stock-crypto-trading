import { useEffect, useState } from "react";
import { CandleChart } from "./components/CandleChart";
import { Header } from "./components/Header";
import { RiskPanel } from "./components/RiskPanel";
import { SignalsTable } from "./components/SignalsTable";
import { fetchCandles, fetchDashboard } from "./services/api";
import type { Candle, Dashboard } from "./types";

export default function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [symbol, setSymbol] = useState<string>("");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboard()
      .then((d) => {
        setDashboard(d);
        setSymbol(d.symbols[0] ?? "");
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!symbol) return;
    fetchCandles(symbol)
      .then(setCandles)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [symbol]);

  if (error) return <div className="page error">Error: {error}</div>;
  if (!dashboard) return <div className="page">Loading…</div>;

  return (
    <div className="page">
      <Header symbols={dashboard.symbols} selected={symbol} onSelect={setSymbol} />
      <section className="panel">
        <h2>{symbol} · daily candles</h2>
        <CandleChart candles={candles} />
      </section>
      <div className="grid">
        <section className="panel">
          <h2>Signals</h2>
          <SignalsTable
            signals={dashboard.signals}
            threshold={dashboard.risk.pThreshold}
            selected={symbol}
            onSelect={setSymbol}
            accountEquity={dashboard.accountEquity}
          />
        </section>
        <section className="panel">
          <h2>Risk</h2>
          <RiskPanel limits={dashboard.risk} signals={dashboard.signals} />
        </section>
      </div>
    </div>
  );
}
