import { useState } from "react";
import type { MouseEvent } from "react";
import type { Candle } from "../types";

interface CandleChartProps {
  candles: Candle[];
  width?: number;
  height?: number;
}

/** Dependency-free SVG candlestick chart. */
export function CandleChart({ candles, width = 820, height = 320 }: CandleChartProps) {
  const [selected, setSelected] = useState<number | null>(null);
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

  const index = Math.min(selected ?? candles.length - 1, candles.length - 1);
  const candle = candles[index];
  const x = pad + (index + 0.5) * step;
  const change = ((candle.close - candle.open) / candle.open) * 100;
  const status = candle.close > candle.open ? "Up" : candle.close < candle.open ? "Down" : "Flat";

  function selectCandle(event: MouseEvent<SVGSVGElement>) {
    const transform = event.currentTarget.getScreenCTM();
    if (!transform) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(transform.inverse());
    const clickedIndex = Math.floor((point.x - pad) / step);
    setSelected(Math.max(0, Math.min(clickedIndex, candles.length - 1)));
  }

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Daily candlestick chart"
        onClick={selectCandle}
      >
        <rect width={width} height={height} fill="transparent" />
        <text x={pad} y={14} className="axis">
          {hi.toFixed(2)}
        </text>
        <text x={pad} y={height - 6} className="axis">
          {lo.toFixed(2)}
        </text>
        <line x1={x} x2={x} y1={pad} y2={height - pad}
          stroke="currentColor" strokeOpacity={0.55} strokeDasharray="4 4" />
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
      <div className="candle-inspector">
        <label htmlFor="candle-session">Trading session</label>
        <small className="candle-hint">Click the chart or use the slider to inspect a session.</small>
        <input id="candle-session" type="range" min={0} max={candles.length - 1}
          value={index} aria-valuetext={candle.time}
          disabled={candles.length === 1}
          onChange={(event) => setSelected(event.currentTarget.valueAsNumber)} />
        <div className="candle-range"><span>{candles[0].time}</span><span>{candles[candles.length - 1].time}</span></div>
        <p><strong>{candle.time}</strong> · {status} ({change > 0 ? "+" : ""}{change.toFixed(2)}% from open)</p>
        <dl>
          <div><dt>Open</dt><dd>{candle.open.toFixed(2)}</dd></div>
          <div><dt>High</dt><dd>{candle.high.toFixed(2)}</dd></div>
          <div><dt>Low</dt><dd>{candle.low.toFixed(2)}</dd></div>
          <div><dt>Close</dt><dd>{candle.close.toFixed(2)}</dd></div>
          <div><dt>Volume</dt><dd>{candle.volume.toLocaleString()}</dd></div>
        </dl>
      </div>
    </>
  );
}
