import type { RiskLimits, Signal } from "../types";

interface RiskPanelProps {
  limits: RiskLimits;
  signals: Signal[];
}

interface MetricProps {
  label: string;
  value: string;
  warn?: boolean;
}

function Metric({ label, value, warn = false }: MetricProps) {
  return (
    <div className={warn ? "metric warn" : "metric"}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

export function RiskPanel({ limits, signals }: RiskPanelProps) {
  const held = signals.filter((s) => s.targetWeight > 0);
  const gross = held.reduce((acc, s) => acc + s.targetWeight, 0);
  const overCap = held.some((s) => s.targetWeight > limits.maxWeight + 1e-9);
  const inCash = held.length < limits.minNames;

  return (
    <div className="metrics">
      <Metric label="Positions" value={`${held.length} / ${limits.topK}`} warn={inCash} />
      <Metric label="Gross exposure" value={`${(gross * 100).toFixed(1)}%`} warn={gross > 1} />
      <Metric label="Per-name cap" value={`${(limits.maxWeight * 100).toFixed(0)}%`} warn={overCap} />
      <Metric label="Signal threshold" value={limits.pThreshold.toFixed(2)} />
      <Metric label="Turnover cap" value={`${(limits.turnoverCap * 100).toFixed(0)}% / day`} />
      <Metric label="Cost per side" value={`${limits.costBpsPerSide} bps`} />
    </div>
  );
}
