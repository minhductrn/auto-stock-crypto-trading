"""Risk parameters. Values are the Stage 1 constants from spec sections 3, 11 and 12."""

from __future__ import annotations

from dataclasses import dataclass

from data_pipeline.errors import ConfigError


@dataclass(frozen=True)
class RiskConfig:
    # Tradable filter (spec section 3)
    min_price: float = 5.0
    min_adv20_dollar: float = 20_000_000.0
    # Signal -> portfolio (spec 11.1)
    p_threshold: float = 0.55
    top_k: int = 20
    min_names: int = 5
    max_weight: float = 0.05
    max_positions: int = 20
    gross_limit: float = 1.0
    turnover_cap: float = 0.30
    # Execution costs (spec 11.2)
    cost_bps_per_side: float = 5.0
    impact_coeff: float = 0.1
    max_adv_participation: float = 0.02
    # Exit barriers (spec section 6)
    atr_mult: float = 1.5
    horizon: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.p_threshold < 1.0:
            raise ConfigError("p_threshold must be in (0, 1)")
        if not 0.0 < self.max_weight <= 1.0:
            raise ConfigError("max_weight must be in (0, 1]")
        if self.gross_limit > 1.0:
            raise ConfigError("Stage 1 is unlevered: gross_limit must be <= 1.0")
        if self.min_names > self.top_k or self.top_k > self.max_positions:
            raise ConfigError("Require min_names <= top_k <= max_positions")
        if not 0.0 < self.turnover_cap <= 1.0:
            raise ConfigError("turnover_cap must be in (0, 1]")


DEFAULT_RISK_CONFIG = RiskConfig()
