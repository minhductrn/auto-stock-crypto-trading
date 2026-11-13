import { useEffect, useState } from "react";
import { CandleChart } from "./components/CandleChart";
import { Header } from "./components/Header";
import { RiskPanel } from "./components/RiskPanel";
import { SignalsTable } from "./components/SignalsTable";
import { fetchCandles, fetchDashboard, isLiveApi } from "./services/api";
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
    let active = true;
    setCandles([]);
    fetchCandles(symbol)
      .then((rows) => { if (active) setCandles(rows); })
      .catch((e: unknown) => {
        if (active) setError(e instanceof Error ? e.message : String(e));
      });
    return () => { active = false; };
  }, [symbol]);

  if (error) return <div className="page error">Error: {error}</div>;
  if (!dashboard) return <div className="page">Loading…</div>;

  const dataStatus = dashboard.dataStatus === "Current" && dashboard.staleAfter &&
    Date.now() >= Date.parse(dashboard.staleAfter) ? "Stale" : dashboard.dataStatus;

  return (
    <div className="page">
      <Header symbols={dashboard.symbols} selected={symbol} onSelect={setSymbol} referenceSymbols={dashboard.referenceSymbols} />
      {dashboard.dataStatus && (
        <p>
          Observation · {dataStatus} · {dashboard.source} · Data session {dashboard.asOf}
          {" · "}Next session {dashboard.nextSession}
          {dashboard.updatedAt && ` · Updated ${new Date(dashboard.updatedAt).toLocaleString()}`}
        </p>
      )}
      {symbol && <section className="panel">
        <h2>{symbol} · daily candles</h2>
        {dashboard.referenceSymbols?.includes(symbol) && <p>Chart only · Yahoo Finance daily bars</p>}
        <CandleChart key={symbol} candles={candles} />
      </section>}
      <div className="grid">
        <section className="panel">
          <h2>{dashboard.dataStatus ? "Observation signals" : isLiveApi ? "Signals" : "Demo signals"}</h2>
          {!isLiveApi && !dashboard.dataStatus && <p>Sample scores and prices for preview only.</p>}
          <details>
            <summary>How to read the scores</summary>
            <p>P(long) is the combined model estimate of reaching the upper price barrier before
              the lower barrier within the model horizon. Weight is the proposed share of portfolio capital.</p>
          </details>
          {dashboard.blockedMlEntries?.map((entry) => (
            <p key={entry.symbol ?? "market-data"}>
              {entry.symbol && `${entry.symbol} · `}<strong>NO TRADE</strong> · {entry.reason}
              {entry.pXgb != null && ` · XGBoost ${entry.pXgb.toFixed(3)}`}
              {entry.pLgbm != null && ` · LightGBM ${entry.pLgbm.toFixed(3)}`}
            </p>
          ))}
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
