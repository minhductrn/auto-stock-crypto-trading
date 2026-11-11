"""Tradable mask: price and liquidity filter (spec section 3)."""

from __future__ import annotations

import pandas as pd

from data_pipeline.errors import DataContractError

from .config import DEFAULT_RISK_CONFIG, RiskConfig

ADV_WINDOW = 20


def compute_tradable(bars: pd.DataFrame, cfg: RiskConfig = DEFAULT_RISK_CONFIG) -> pd.DataFrame:
    """Return ``symbol, session_date, adv20_dollar, tradable``.

    Causal window: ``[t-19 .. t]`` of close * volume, evaluated at ``t``.
    ``bars`` must carry the RAW close and volume (spec 3.2 exception).
    """
    missing = {"symbol", "session_date", "close", "volume"} - set(bars.columns)
    if missing:
        raise DataContractError(f"bars missing columns {sorted(missing)}")
    df = bars.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    dollar = df["close"].astype(float) * df["volume"].astype(float)
    adv = dollar.groupby(df["symbol"]).transform(lambda s: s.rolling(ADV_WINDOW).mean())
    out = df[["symbol", "session_date"]].copy()
    out["adv20_dollar"] = adv
    out["tradable"] = (df["close"] >= cfg.min_price) & (adv >= cfg.min_adv20_dollar)
    return out
