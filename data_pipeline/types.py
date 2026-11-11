"""Shared lightweight types (mirrors spec section 9)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .errors import ConfigError

Frame = pd.DataFrame

BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vendor",
    "ingested_at",
)

ACTION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ex_date",
    "action_type",
    "split_ratio",
    "cash_amount",
)


@dataclass(frozen=True)
class DateRange:
    """Inclusive date range."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ConfigError(f"DateRange start {self.start} is after end {self.end}")
