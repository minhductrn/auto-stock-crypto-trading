"""Stop / target / time exits matching the triple-barrier label (spec sections 6 and 11.1)."""

from __future__ import annotations

from dataclasses import dataclass

from data_pipeline.errors import RiskLimitError

from .config import DEFAULT_RISK_CONFIG, RiskConfig


@dataclass(frozen=True)
class ExitLevels:
    entry_px: float
    target_px: float
    stop_px: float
    max_holding_sessions: int


def compute_exit_levels(entry_px: float, atr_ref: float,
                        cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> ExitLevels:
    """Barriers are set once at entry from ``ATR14_t`` (known at decision time ``t``)."""
    if entry_px <= 0 or atr_ref <= 0:
        raise RiskLimitError("entry_px and atr_ref must be positive")
    width = cfg.atr_mult * atr_ref
    return ExitLevels(entry_px, entry_px + width, entry_px - width, cfg.horizon)
