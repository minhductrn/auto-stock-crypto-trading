import { useState } from "react";
import { executeOrder, isLiveApi } from "../services/api";
import type { Signal } from "../types";

interface SignalsTableProps {
  signals: Signal[];
  threshold: number;
  selected: string;
  onSelect: (symbol: string) => void;
  /** Paper account equity; used to turn targetWeight into a dollar notional. */
  accountEquity?: number | null;
}

type RowStatus = { kind: "pending" | "ok" | "error"; text: string };

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const usd = (v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export function SignalsTable({
  signals,
  threshold,
  selected,
  onSelect,
  accountEquity,
}: SignalsTableProps) {
  const [status, setStatus] = useState<Record<string, RowStatus>>({});

  const setRow = (symbol: string, s: RowStatus) => setStatus((prev) => ({ ...prev, [symbol]: s }));

  async function approve(s: Signal) {
    const notional = Math.floor((accountEquity ?? 0) * s.targetWeight);
    if (notional <= 0) return;
    const ok = window.confirm(
      `Approve paper BUY ${s.symbol}?\n\nTarget notional ≈ ${usd(notional)} (${pct(s.targetWeight)} of equity).\n` +
        `A bracket stop-loss is attached and size is capped at 1% account risk.`,
    );
    if (!ok) return;
    setRow(s.symbol, { kind: "pending", text: "Submitting…" });
    try {
      const r = await executeOrder({
        symbol: s.symbol,
        side: "buy",
        notional_or_qty: notional,
        unit: "notional",
      });
      setRow(s.symbol, {
        kind: "ok",
        text: `${r.status}: ${r.qty} sh, SL ${r.stopPrice ?? "–"}, TP ${r.takeProfitPrice ?? "–"}` +
          (r.cappedBy.length ? ` (capped: ${r.cappedBy.join(", ")})` : ""),
      });
    } catch (e: unknown) {
      setRow(s.symbol, { kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  }

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
          {isLiveApi && <th>Action</th>}
        </tr>
      </thead>
      <tbody>
        {signals.map((s) => {
          const row = status[s.symbol];
          const notional = Math.floor((accountEquity ?? 0) * s.targetWeight);
          const canApprove = notional > 0 && row?.kind !== "pending";
          return (
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
              {isLiveApi && (
                <td>
                  <button
                    type="button"
                    className="btn-approve"
                    disabled={!canApprove}
                    title={
                      canApprove
                        ? `Paper BUY ~${usd(notional)} with bracket stop-loss`
                        : "No target weight or account equity unavailable"
                    }
                    onClick={(e) => {
                      e.stopPropagation();
                      void approve(s);
                    }}
                  >
                    {row?.kind === "pending" ? "…" : "Approve / Execute"}
                  </button>
                  {row && row.kind !== "pending" && (
                    <div className={row.kind === "ok" ? "pos order-msg" : "order-msg order-err"}>
                      {row.text}
                    </div>
                  )}
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
