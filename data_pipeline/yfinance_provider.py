"""yfinance adapter for US-equity daily candles (spec section 9 ``DataProvider``).

Caveat: Yahoo serves split-adjusted (but not dividend-adjusted) closes even with
``auto_adjust=False``. Spec section 3 wants truly unadjusted raw prices with
corporate actions stored separately; that requires a different vendor before
Stage 1 exit. The corporate-action table is fetched here so the adjuster can be
built against the final contract.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from .errors import DataProviderError
from .types import ACTION_COLUMNS, BAR_COLUMNS, DateRange, Frame

logger = logging.getLogger(__name__)


def _session_dates(index: pd.Index) -> np.ndarray:
    """Convert a (possibly tz-aware) DatetimeIndex to exchange-local ``date`` objects."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)  # yfinance already localises to America/New_York
    return idx.normalize().date


class YFinanceProvider:
    """Fetches 1-day bars and corporate actions from Yahoo Finance."""

    vendor = "yfinance"

    def __init__(self, max_retries: int = 3, backoff_seconds: float = 1.5) -> None:
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    # ------------------------------------------------------------------ public
    def fetch_bars_1d(self, symbols: Sequence[str], rng: DateRange) -> Frame:
        """Return bars matching spec 8.1, sorted by (symbol, session_date).

        Causal window: only bars with ``rng.start <= session_date <= rng.end``.
        """
        frames = [self._bars_one(sym.upper(), rng) for sym in symbols]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["symbol", "session_date"]).reset_index(drop=True)

    def fetch_corporate_actions(self, symbols: Sequence[str], rng: DateRange) -> Frame:
        """Return splits and cash dividends matching spec 8.2."""
        rows: list[dict[str, object]] = []
        for sym in (s.upper() for s in symbols):
            raw = self._history(sym, rng, actions=True)
            if raw.empty or not {"Dividends", "Stock Splits"} <= set(raw.columns):
                continue
            dates = _session_dates(raw.index)
            dividends = raw["Dividends"].to_numpy(dtype=float)
            splits = raw["Stock Splits"].to_numpy(dtype=float)
            for d, amount in zip(dates, dividends):
                if amount > 0:
                    rows.append(
                        {"symbol": sym, "ex_date": d, "action_type": "CASH_DIVIDEND",
                         "split_ratio": np.nan, "cash_amount": float(amount)}
                    )
            for d, ratio in zip(dates, splits):
                if ratio > 0:
                    rows.append(
                        {"symbol": sym, "ex_date": d, "action_type": "SPLIT",
                         "split_ratio": float(ratio), "cash_amount": np.nan}
                    )
        if not rows:
            return pd.DataFrame(columns=list(ACTION_COLUMNS))
        out = pd.DataFrame(rows, columns=list(ACTION_COLUMNS))
        return out.sort_values(["symbol", "ex_date", "action_type"]).reset_index(drop=True)

    def fetch_index_levels(self, symbol: str, rng: DateRange) -> Frame:
        """Return daily levels for a context index such as ``^VIX`` (read-only context)."""
        return self._bars_one(symbol.upper(), rng)

    # ----------------------------------------------------------------- private
    def _history(self, symbol: str, rng: DateRange, actions: bool) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return yf.Ticker(symbol).history(
                    start=rng.start.isoformat(),
                    end=(rng.end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                    interval="1d",
                    auto_adjust=False,
                    actions=actions,
                )
            except Exception as exc:  # noqa: BLE001 - vendor/network errors are heterogeneous
                last_exc = exc
                logger.warning("yfinance %s attempt %d/%d failed: %s",
                               symbol, attempt, self.max_retries, exc)
                time.sleep(self.backoff_seconds * attempt)
        raise DataProviderError(f"yfinance failed for {symbol}") from last_exc

    def _bars_one(self, symbol: str, rng: DateRange) -> Frame:
        raw = self._history(symbol, rng, actions=False)
        if raw.empty:
            logger.warning("No bars returned for %s in %s..%s", symbol, rng.start, rng.end)
            return pd.DataFrame(columns=list(BAR_COLUMNS))
        raw = raw.rename(columns=str.lower)
        missing = {"open", "high", "low", "close", "volume"} - set(raw.columns)
        if missing:
            raise DataProviderError(f"{symbol}: vendor payload missing columns {sorted(missing)}")
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "session_date": _session_dates(raw.index),
                "open": raw["open"].to_numpy(dtype="float64"),
                "high": raw["high"].to_numpy(dtype="float64"),
                "low": raw["low"].to_numpy(dtype="float64"),
                "close": raw["close"].to_numpy(dtype="float64"),
                "volume": raw["volume"].fillna(0).to_numpy(dtype="int64"),
                "vendor": self.vendor,
            }
        )
        frame["ingested_at"] = pd.Timestamp.now(tz="UTC")
        return frame[list(BAR_COLUMNS)]
