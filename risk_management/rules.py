"""Hard portfolio limits. Violations raise ``RiskLimitError``."""

from __future__ import annotations

import pandas as pd

from data_pipeline.errors import RiskLimitError

from .config import DEFAULT_RISK_CONFIG, RiskConfig

_TOL = 1e-9


def validate_target_weights(weights: pd.Series, cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> None:
    """Check long-only, per-name cap, position count and unlevered gross exposure."""
    if weights.isna().any():
        raise RiskLimitError("NaN target weight")
    if (weights < -_TOL).any():
        raise RiskLimitError("Short positions are forbidden in Stage 1")
    if (weights > cfg.max_weight + _TOL).any():
        raise RiskLimitError(f"Per-name weight above {cfg.max_weight:.2%}")
    if int((weights > _TOL).sum()) > cfg.max_positions:
        raise RiskLimitError(f"More than {cfg.max_positions} concurrent positions")
    if weights.sum() > cfg.gross_limit + _TOL:
        raise RiskLimitError("Gross exposure above limit (no leverage allowed)")
