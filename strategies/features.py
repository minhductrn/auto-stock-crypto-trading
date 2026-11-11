"""Baseline price/volume features (subset of spec Group P, 12 of the 16 columns).

This is a template, NOT ``sdf_fs_v1``: the SDF geometry, cross-sectional, and
context groups are not implemented yet. ``FEATURE_SET_VERSION`` is therefore
distinct so artifacts can never be confused with the 82-column contract.
Every feature at ``(symbol, t)`` uses only bars with ``session_date <= t``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from data_pipeline.errors import DataContractError

FEATURE_SET_VERSION = "baseline_p12_v1"
KEYS: tuple[str, ...] = ("symbol", "session_date")
FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    "rv_20", "rv_60", "atr14_pct", "gap_pct", "range_pct", "close_loc", "vol_z20",
)
_REQUIRED = {"symbol", "session_date", "open", "high", "low", "close", "volume"}


def feature_hash(columns: tuple[str, ...] = FEATURE_COLUMNS) -> str:
    """SHA-256 of the ordered column list (column order is API)."""
    return hashlib.sha256(",".join(columns).encode()).hexdigest()


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder ATR, seeded with the simple mean of the first ``window`` true ranges.

    Causal: ``atr[i]`` depends only on bars ``0..i``. Floored at ``1e-8 * close``.
    """
    n = len(close)
    atr = np.full(n, np.nan)
    if n < window:
        return atr
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr[window - 1] = tr[:window].mean()
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return np.maximum(atr, 1e-8 * close)


def _features_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    o, h, l, c, v = (g[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    logc = np.log(c)
    ret_1 = logc.diff()

    f = pd.DataFrame({"symbol": g["symbol"], "session_date": g["session_date"]})
    for n in (1, 5, 10, 20, 60):
        f[f"ret_{n}"] = logc - logc.shift(n)
    f["rv_20"] = ret_1.rolling(20).std(ddof=1)
    f["rv_60"] = ret_1.rolling(60).std(ddof=1)
    atr = pd.Series(wilder_atr(h.to_numpy(), l.to_numpy(), c.to_numpy()), index=g.index)
    f["atr14_pct"] = atr / c
    f["gap_pct"] = o / c.shift(1) - 1.0
    f["range_pct"] = (h - l) / c
    f["close_loc"] = ((c - l) / (h - l)).where(h > l, 0.5)
    lv = np.log1p(v)
    std = lv.rolling(20).std(ddof=1)
    z = (lv - lv.rolling(20).mean()) / std
    f["vol_z20"] = z.where(std > 0, 0.0).where(std.notna())

    f[list(FEATURE_COLUMNS)] = f[list(FEATURE_COLUMNS)].astype("float32")
    return f


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the baseline feature matrix, sorted by (symbol, session_date).

    Causal window: row ``t`` uses bars ``<= t`` of the same symbol only.
    """
    missing = _REQUIRED - set(bars.columns)
    if missing:
        raise DataContractError(f"bars missing columns {sorted(missing)}")
    ordered = bars.sort_values(list(KEYS)).reset_index(drop=True)
    parts = [_features_one(g) for _, g in ordered.groupby("symbol", sort=True)]
    if not parts:
        return pd.DataFrame(columns=list(KEYS) + list(FEATURE_COLUMNS))
    return pd.concat(parts, ignore_index=True)
