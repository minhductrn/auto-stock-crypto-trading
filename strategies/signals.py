"""Probability -> target-weight signals (spec 8.7 schema, 11.1 rules)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.errors import SchemaContractError
from risk_management.config import DEFAULT_RISK_CONFIG, RiskConfig
from risk_management.rules import validate_target_weights
from risk_management.sizing import select_target_weights

SIGNAL_COLUMNS: tuple[str, ...] = (
    "session_date", "symbol", "p_xgb", "p_lgbm", "p_long", "rank", "target_weight", "run_id",
)
_REQUIRED = {"session_date", "symbol", "p_xgb", "p_lgbm", "p_long", "atr14_pct", "tradable"}


def generate_signals(probs: pd.DataFrame, run_id: str,
                     cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> pd.DataFrame:
    """Rank tradable names per date and assign validated long-only target weights.

    Causal window: each date is processed independently using same-date information only.
    Weights apply from the ``t+1`` open.
    """
    missing = _REQUIRED - set(probs.columns)
    if missing:
        raise SchemaContractError(f"probs missing columns {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for _, g in probs[probs["tradable"]].groupby("session_date", sort=True):
        g = g.sort_values(["p_long", "symbol"], ascending=[False, True]).copy()
        g["rank"] = np.arange(1, len(g) + 1, dtype="int16")
        by_symbol = g.set_index("symbol")
        w = select_target_weights(by_symbol["p_long"], by_symbol["atr14_pct"], cfg)
        validate_target_weights(w, cfg)
        g["target_weight"] = w.reindex(g["symbol"]).fillna(0.0).to_numpy()
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=list(SIGNAL_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out["run_id"] = run_id
    for col in ("p_xgb", "p_lgbm", "p_long", "target_weight"):
        out[col] = out[col].astype("float32")
    return out[list(SIGNAL_COLUMNS)]
