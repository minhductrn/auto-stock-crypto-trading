"""Execution cost model (spec 11.2)."""

from __future__ import annotations

from data_pipeline.errors import RiskLimitError

from .config import DEFAULT_RISK_CONFIG, RiskConfig


def participation_limited_dollars(order_dollars: float, adv20_dollar: float,
                                  cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> float:
    """Cap an order at ``max_adv_participation`` of ADV20; the excess is not filled."""
    if adv20_dollar <= 0:
        raise RiskLimitError("adv20_dollar must be positive")
    return min(abs(order_dollars), cfg.max_adv_participation * adv20_dollar)


def trade_cost_bps(order_dollars: float, adv20_dollar: float,
                   cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> float:
    """Per-side cost in bps: fixed slippage/fees plus ``impact_coeff * participation * 1e4``."""
    filled = participation_limited_dollars(order_dollars, adv20_dollar, cfg)
    return cfg.cost_bps_per_side + cfg.impact_coeff * (filled / adv20_dollar) * 1e4
