import type { Candle } from "../types";

interface CandleChartProps {
  candles: Candle[];
  width?: number;
  height?: number;
}

/** Dependency-free SVG candlestick chart. */
export function CandleChart({ candles, width = 820, height = 320 }: CandleChartProps) {
  if (candles.length === 0) {
    return <div className="empty">No candles to display</div>;
  }
  const pad = 24;
  const lo = Math.min(...candles.map((c) => c.low));
  const hi = Math.max(...candles.map((c) => c.high));
  const span = hi - lo || 1;
  const step = (width - 2 * pad) / candles.length;
  const bodyWidth = Math.max(step * 0.6, 1);
  const y = (price: number) => pad + ((hi - price) / span) * (height - 2 * pad);

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Daily candlestick chart"
    >
      <text x={pad} y={14} className="axis">
        {hi.toFixed(2)}
      </text>
      <text x={pad} y={height - 6} className="axis">
        {lo.toFixed(2)}
      </text>
      {candles.map((c, i) => {
        const x = pad + i * step + step / 2;
        const up = c.close >= c.open;
        const color = up ? "var(--up)" : "var(--down)";
        const top = y(Math.max(c.open, c.close));
        const bodyHeight = Math.max(Math.abs(y(c.open) - y(c.close)), 1);
        return (
          <g key={c.time}>
            <line x1={x} x2={x} y1={y(c.high)} y2={y(c.low)} stroke={color} />
            <rect x={x - bodyWidth / 2} y={top} width={bodyWidth} height={bodyHeight} fill={color} />
          </g>
        );
      })}
    </svg>
  );
}
