import type { Signal } from "../types";

interface SignalsTableProps {
  signals: Signal[];
  threshold: number;
  selected: string;
  onSelect: (symbol: string) => void;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function SignalsTable({ signals, threshold, selected, onSelect }: SignalsTableProps) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>#</th>
          <th>Symbol</th>
          <th>P(XGB)</th>
          <th>P(LGBM)</th>
          <th>P(long)</th>
          <th>Weight</th>
        </tr>
      </thead>
      <tbody>
        {signals.map((s) => (
          <tr
            key={s.symbol}
            className={s.symbol === selected ? "selected" : undefined}
            onClick={() => onSelect(s.symbol)}
          >
            <td>{s.rank}</td>
            <td>{s.symbol}</td>
            <td>{s.pXgb.toFixed(3)}</td>
            <td>{s.pLgbm.toFixed(3)}</td>
            <td className={s.pLong >= threshold ? "pos" : "muted"}>{s.pLong.toFixed(3)}</td>
            <td>{pct(s.targetWeight)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
