"""Triple-barrier labeller ``tb_v1`` (spec section 6).

Decision at the close of ``t``; entry at ``open[t+1]``; barriers at
``entry +/- 1.5 * ATR14_t``; horizon ``H = 10`` sessions (``t+1 .. t+H``).
Rows without a full forward window are not labelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.errors import DataContractError

from .features import wilder_atr

LABEL_VERSION = "tb_v1"
LABEL_COLUMNS: tuple[str, ...] = (
    "symbol", "session_date", "y_tb", "y_long", "t_exit",
    "ret_fwd", "ret_fwd_net", "sample_weight", "label_version",
)


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon: int = 10
    atr_mult: float = 1.5
    cost_bps: float = 5.0  # per side; round trip = 2x


def average_uniqueness(starts: np.ndarray, ends: np.ndarray, n_bars: int) -> np.ndarray:
    """Average-uniqueness weight (Lopez de Prado) for inclusive spans ``[start, end]``."""
    diff = np.zeros(n_bars + 1)
    np.add.at(diff, starts, 1.0)
    np.add.at(diff, ends + 1, -1.0)
    conc = np.cumsum(diff)[:n_bars]
    inv = np.divide(1.0, conc, out=np.zeros(n_bars), where=conc > 0)
    csum = np.concatenate(([0.0], np.cumsum(inv)))
    return (csum[ends + 1] - csum[starts]) / (ends - starts + 1)


def _label_one(g: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    o, h, l, c = (g[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    dates = g["session_date"].to_numpy()
    n, horizon = len(g), cfg.horizon
    atr = wilder_atr(h, l, c)

    rows: list[tuple[int, int, int, float, float]] = []  # (t, hit_idx, label, entry, exit)
    for t in range(n - horizon):
        if np.isnan(atr[t]):
            continue
        entry = o[t + 1]
        upper = entry + cfg.atr_mult * atr[t]
        lower = entry - cfg.atr_mult * atr[t]
        label, exit_px, hit = 0, c[t + horizon], t + horizon
        for s in range(t + 1, t + horizon + 1):
            if s >= t + 2:  # gap through a barrier at the open
                if o[s] >= upper:
                    label, exit_px, hit = 1, o[s], s
                    break
                if o[s] <= lower:
                    label, exit_px, hit = -1, o[s], s
                    break
            if l[s] <= lower:  # checked first: pessimistic tie-break when both touched
                label, exit_px, hit = -1, lower, s
                break
            if h[s] >= upper:
                label, exit_px, hit = 1, upper, s
                break
        rows.append((t, hit, label, entry, exit_px))

    if not rows:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    arr = np.array(rows, dtype=float)
    t_idx, hit_idx = arr[:, 0].astype(int), arr[:, 1].astype(int)
    y_tb = arr[:, 2].astype("int8")
    ret_fwd = np.log(arr[:, 4] / arr[:, 3])
    return pd.DataFrame(
        {
            "symbol": g["symbol"].iloc[0],
            "session_date": dates[t_idx],
            "y_tb": y_tb,
            "y_long": (y_tb == 1).astype("int8"),
            "t_exit": dates[hit_idx],
            "ret_fwd": ret_fwd.astype("float32"),
            "ret_fwd_net": (ret_fwd - 2.0 * cfg.cost_bps / 1e4).astype("float32"),
            "sample_weight": average_uniqueness(t_idx + 1, hit_idx, n).astype("float32"),
            "label_version": LABEL_VERSION,
        }
    )


def label_triple_barrier(bars: pd.DataFrame, cfg: TripleBarrierConfig = TripleBarrierConfig()) -> pd.DataFrame:
    """Label every ``(symbol, t)`` that has a full ``t+1..t+H`` window.

    ``sample_weight`` is the raw uniqueness; normalise to mean 1 inside each training fold.
    """
    missing = {"symbol", "session_date", "open", "high", "low", "close"} - set(bars.columns)
    if missing:
        raise DataContractError(f"bars missing columns {sorted(missing)}")
    ordered = bars.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    parts = [_label_one(g, cfg) for _, g in ordered.groupby("symbol", sort=True)]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))
    return pd.concat(parts, ignore_index=True)[list(LABEL_COLUMNS)]
