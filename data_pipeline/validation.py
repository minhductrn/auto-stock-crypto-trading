"""Data-quality gates for daily bars (spec section 8.5).

The NYSE session-calendar and missing-session checks need the calendar module
(``exchange_calendars``), which is not wired yet; a weekday check is used as a
conservative proxy until then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import DataContractError
from .types import Frame

DQ_LARGE_MOVE = 1 << 0  # |1-day return| > 40% (no corporate-action table joined yet)
DQ_ZERO_VOLUME = 1 << 1


def validate_bars(bars: Frame, max_quarantine_share: float = 0.02) -> tuple[Frame, Frame]:
    """Split bars into ``(valid, quarantined)``.

    Fatal rules send a row to quarantine (with a ``reason`` column). Warnings are
    encoded in the ``dq_flags`` int32 bitmask on valid rows.

    Raises:
        DataContractError: if the quarantined share of any session date exceeds
            ``max_quarantine_share``.
    """
    if bars.empty:
        return bars.assign(dq_flags=np.array([], dtype="int32")), bars.assign(reason=pd.Series(dtype=str))

    df = bars.reset_index(drop=True)
    ohlc = df[["open", "high", "low", "close"]]
    weekday = pd.to_datetime(df["session_date"]).dt.weekday

    masks = pd.DataFrame(
        {
            "null_value": df[["symbol", "session_date", "open", "high", "low", "close", "volume"]]
            .isna().any(axis=1),
            "non_positive_price": (ohlc <= 0).any(axis=1),
            "negative_volume": df["volume"] < 0,
            "ohlc_inconsistent": (df["high"] < ohlc[["open", "close", "low"]].max(axis=1))
            | (df["low"] > ohlc[["open", "close", "high"]].min(axis=1)),
            "duplicate_key": df.duplicated(["symbol", "session_date"], keep=False),
            "non_session_weekend": weekday >= 5,
        }
    )
    fatal = masks.any(axis=1)

    reasons = masks.apply(lambda row: ";".join(masks.columns[row.to_numpy()]), axis=1)
    quarantined = df[fatal].assign(reason=reasons[fatal])
    valid = df[~fatal].copy()

    # Non-fatal warnings
    valid = valid.sort_values(["symbol", "session_date"])
    one_day = valid.groupby("symbol")["close"].pct_change().abs()
    flags = np.zeros(len(valid), dtype="int32")
    flags |= np.where(one_day.to_numpy() > 0.40, DQ_LARGE_MOVE, 0).astype("int32")
    flags |= np.where(valid["volume"].to_numpy() == 0, DQ_ZERO_VOLUME, 0).astype("int32")
    valid["dq_flags"] = flags
    valid = valid.reset_index(drop=True)

    share = fatal.groupby(df["session_date"]).mean()
    worst = share[share > max_quarantine_share]
    if not worst.empty:
        raise DataContractError(
            f"Quarantined share exceeds {max_quarantine_share:.0%} on {len(worst)} date(s); "
            f"worst {worst.idxmax()} at {worst.max():.0%}"
        )
    return valid, quarantined.reset_index(drop=True)
