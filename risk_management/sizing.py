"""Position selection and inverse-volatility sizing (spec 11.1)."""

from __future__ import annotations

import pandas as pd

from data_pipeline.errors import RiskLimitError

from .config import DEFAULT_RISK_CONFIG, RiskConfig

_EPS = 1e-12


def inverse_vol_weights(atr_pct: pd.Series, max_weight: float, gross_limit: float = 1.0) -> pd.Series:
    """Weights proportional to ``1/atr_pct``, capped per name and renormalised under the cap.

    Excess above the cap is redistributed to uncapped names. If every name is capped,
    the remainder stays in cash (sum < ``gross_limit``).
    """
    if atr_pct.empty:
        return atr_pct.astype(float)
    if (atr_pct <= 0).any() or atr_pct.isna().any():
        raise RiskLimitError("atr_pct must be positive and finite for sizing")

    raw = (1.0 / atr_pct.astype(float))
    raw = raw / raw.sum() * gross_limit
    weights = raw.copy()
    capped = pd.Series(False, index=weights.index)

    for _ in range(len(weights)):
        newly = (weights > max_weight + _EPS) & ~capped
        if not newly.any():
            break
        capped |= newly
        weights[capped] = max_weight
        free = ~capped
        if not free.any():
            break
        remaining = max(gross_limit - max_weight * int(capped.sum()), 0.0)
        weights[free] = raw[free] / raw[free].sum() * remaining
    return weights


def select_target_weights(p_long: pd.Series, atr_pct: pd.Series,
                          cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> pd.Series:
    """Pick names and size them for one decision date.

    Rules: keep ``p_long >= p_threshold``, top ``top_k`` by probability; hold cash if
    fewer than ``min_names`` qualify. Inputs are indexed by symbol for a single date
    and must already be restricted to tradable names.
    """
    weights = pd.Series(0.0, index=p_long.index)
    vol = atr_pct.reindex(p_long.index)
    eligible = p_long[(p_long >= cfg.p_threshold) & (vol > 0)]
    if len(eligible) < cfg.min_names:
        return weights
    top = eligible.sort_values(ascending=False, kind="stable").head(cfg.top_k)
    sized = inverse_vol_weights(vol.loc[top.index], cfg.max_weight, cfg.gross_limit)
    weights.loc[sized.index] = sized
    return weights
