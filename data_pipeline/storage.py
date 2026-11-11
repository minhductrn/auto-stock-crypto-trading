"""Atomic, idempotent parquet I/O for the L0 raw layer (spec section 8)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from .errors import DataContractError
from .types import Frame


def write_parquet_atomic(df: Frame, path: Path) -> None:
    """Write to a temp file then rename, so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _batch_id(symbols: list[str]) -> str:
    return hashlib.sha1(",".join(sorted(set(symbols))).encode()).hexdigest()[:8]


def write_raw_bars(bars: Frame, root: Path) -> list[Path]:
    """Write bars to ``root/date=YYYY-MM-DD/part-<batch>.parquet``.

    The batch id is derived from the symbol set, so identical inputs rewrite
    identical paths (idempotent) while different symbol batches never collide.
    """
    batch = _batch_id(bars["symbol"].unique().tolist())
    written: list[Path] = []
    for session_date, group in bars.groupby("session_date", sort=True):
        path = root / f"date={session_date.isoformat()}" / f"part-{batch}.parquet"
        write_parquet_atomic(group, path)
        written.append(path)
    return written


def write_quarantine(quarantined: Frame, root: Path) -> Path | None:
    """Write quarantined rows (with ``reason``) under ``root``; no-op when empty."""
    if quarantined.empty:
        return None
    batch = _batch_id(quarantined["symbol"].unique().tolist())
    path = root / f"quarantine-{batch}.parquet"
    write_parquet_atomic(quarantined, path)
    return path


def read_bars(root: Path) -> Frame:
    """Read every parquet part under ``root`` and return bars sorted by (symbol, session_date)."""
    files = sorted(Path(root).rglob("*.parquet"))
    if not files:
        raise DataContractError(f"No parquet files found under {root}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df.sort_values(["symbol", "session_date"]).reset_index(drop=True)
