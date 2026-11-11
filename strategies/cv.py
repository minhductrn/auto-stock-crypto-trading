"""Purged, embargoed, expanding-window walk-forward splitter (spec 7.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.errors import DataContractError


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_years: int = 5
    test_months: int = 6
    step_months: int = 6
    embargo_sessions: int = 10
    holdout_months: int = 12


@dataclass(frozen=True)
class Fold:
    index: int
    train_idx: np.ndarray  # positional indices into the dataset
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def purged_walk_forward(dataset: pd.DataFrame, cfg: WalkForwardConfig = WalkForwardConfig()) -> list[Fold]:
    """Generate folds from a frame with ``session_date`` and ``t_exit`` columns.

    - Training rows must end before ``test_start`` (``t_exit < test_start``: purge).
    - The first ``embargo_sessions`` sessions after each earlier test block are removed
      from later training sets.
    - The most recent ``holdout_months`` are never used by any fold.
    """
    for col in ("session_date", "t_exit"):
        if col not in dataset.columns:
            raise DataContractError(f"dataset missing column {col!r}")

    dates = pd.to_datetime(dataset["session_date"]).to_numpy()
    t_exit = pd.to_datetime(dataset["t_exit"]).to_numpy()
    sessions = np.sort(np.unique(dates))
    if len(sessions) == 0:
        return []

    first = pd.Timestamp(sessions[0])
    holdout_cut = pd.Timestamp(sessions[-1]) - pd.DateOffset(months=cfg.holdout_months)
    test_start = first + pd.DateOffset(years=cfg.min_train_years)

    folds: list[Fold] = []
    embargoed = np.array([], dtype=dates.dtype)
    while test_start <= holdout_cut:
        block_end_excl = min(test_start + pd.DateOffset(months=cfg.test_months),
                             holdout_cut + pd.Timedelta(days=1))
        ts, te = np.datetime64(test_start), np.datetime64(block_end_excl)
        test_mask = (dates >= ts) & (dates < te)
        train_mask = (dates < ts) & (t_exit < ts) & ~np.isin(dates, embargoed)
        if not test_mask.any():
            break
        if train_mask.any():
            folds.append(Fold(
                index=len(folds),
                train_idx=np.flatnonzero(train_mask),
                test_idx=np.flatnonzero(test_mask),
                test_start=test_start,
                test_end=block_end_excl - pd.Timedelta(days=1),
            ))
        pos = int(np.searchsorted(sessions, te, side="left"))
        embargoed = np.union1d(embargoed, sessions[pos:pos + cfg.embargo_sessions])
        test_start = test_start + pd.DateOffset(months=cfg.step_months)
    return folds


def split_early_stopping(train: pd.DataFrame, val_fraction: float = 0.10
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last ``val_fraction`` of training dates becomes the early-stopping set.

    Remaining training rows whose label span reaches the validation start are removed
    (purge), so the validation set never overlaps training labels.
    """
    d = pd.to_datetime(train["session_date"])
    te = pd.to_datetime(train["t_exit"])
    unique = np.sort(d.unique())
    cut = unique[min(int(len(unique) * (1.0 - val_fraction)), len(unique) - 1)]
    val = train[d >= cut]
    fit = train[(d < cut) & (te < cut)]
    if fit.empty or val.empty:
        raise DataContractError("Empty fit or validation set after purge")
    return fit, val
