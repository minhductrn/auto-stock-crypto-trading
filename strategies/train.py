"""Walk-forward training entry point.

Run from the repository root (after ``python -m data_pipeline.fetch_candles``)::

    python -m strategies.train --bars data/raw/bars_1d --run-id baseline_001
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from data_pipeline.errors import DataContractError
from data_pipeline.storage import read_bars
from risk_management.liquidity import compute_tradable

from .cv import WalkForwardConfig, purged_walk_forward, split_early_stopping
from .features import FEATURE_COLUMNS, FEATURE_SET_VERSION, KEYS, build_features, feature_hash
from .labels import LABEL_VERSION, label_triple_barrier
from .models import LGBM_PARAMS, SEED, XGB_PARAMS, LGBMStageModel, XGBStageModel, ensemble_proba

logger = logging.getLogger("strategies.train")
SPEC_VERSION = "stage1-v1.0.0"


def build_dataset(bars: pd.DataFrame, warmup_sessions: int = 252) -> pd.DataFrame:
    """Join features, labels and the tradable mask; keep eligible rows (spec 5.5)."""
    bars = bars.sort_values(list(KEYS)).reset_index(drop=True)
    bars["n_prior"] = bars.groupby("symbol").cumcount()
    tradable = compute_tradable(bars)
    ds = (
        build_features(bars)
        .merge(label_triple_barrier(bars), on=list(KEYS), how="inner")
        .merge(tradable[list(KEYS) + ["tradable"]], on=list(KEYS), how="inner")
        .merge(bars[list(KEYS) + ["n_prior"]], on=list(KEYS), how="inner")
    )
    finite = np.isfinite(ds[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    keep = ds["tradable"].to_numpy() & (ds["n_prior"].to_numpy() >= warmup_sessions) & finite
    logger.info("Dataset rows: %d total, %d eligible, %d excluded", len(ds), keep.sum(), (~keep).sum())
    out = ds[keep].sort_values(["session_date", "symbol"]).reset_index(drop=True)
    if out.empty:
        raise DataContractError("No eligible rows after warm-up, liquidity and finiteness filters")
    return out


def _normalised(w: pd.Series) -> pd.Series:
    return w / w.mean()


def _metrics(y: pd.Series, p: pd.Series) -> dict[str, float | None]:
    def safe(fn, *a):  # single-class folds make AUC/log-loss undefined
        try:
            v = float(fn(*a))
            return None if math.isnan(v) else v
        except ValueError:
            return None
    pc = p.clip(1e-6, 1 - 1e-6)
    return {"auc": safe(roc_auc_score, y, p), "logloss": safe(log_loss, y, pc, ),
            "brier": safe(brier_score_loss, y, p)}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_training(bars: pd.DataFrame, run_id: str, out_root: Path,
                 wf_cfg: WalkForwardConfig = WalkForwardConfig(), warmup_sessions: int = 252,
                 val_fraction: float = 0.10, n_jobs: int = 1) -> dict:
    """Run purged walk-forward training; save last-fold models and a manifest."""
    ds = build_dataset(bars, warmup_sessions)
    folds = purged_walk_forward(ds, wf_cfg)
    if not folds:
        raise DataContractError("No walk-forward folds: not enough history for the configured CV")

    cols = list(FEATURE_COLUMNS)
    fold_reports: list[dict] = []
    models: dict[str, XGBStageModel | LGBMStageModel] = {}
    for fold in folds:
        fit, val = split_early_stopping(ds.iloc[fold.train_idx], val_fraction)
        test = ds.iloc[fold.test_idx]
        xgb_m, lgb_m = XGBStageModel(n_jobs), LGBMStageModel(n_jobs)
        for m in (xgb_m, lgb_m):
            m.fit(fit[cols], fit["y_long"], _normalised(fit["sample_weight"]),
                  val[cols], val["y_long"], _normalised(val["sample_weight"]))
        p = {"xgb": xgb_m.predict_proba(test[cols]), "lgbm": lgb_m.predict_proba(test[cols])}
        p["ensemble"] = ensemble_proba({"xgb": p["xgb"], "lgbm": p["lgbm"]})
        report = {
            "fold": fold.index,
            "test_start": str(fold.test_start.date()), "test_end": str(fold.test_end.date()),
            "n_train": int(len(fit)), "n_val": int(len(val)), "n_test": int(len(test)),
            "metrics": {k: _metrics(test["y_long"], v) for k, v in p.items()},
        }
        logger.info("fold %d AUC ens=%s", fold.index, report["metrics"]["ensemble"]["auc"])
        fold_reports.append(report)
        models = {"xgb": xgb_m, "lgbm": lgb_m}

    out_dir = out_root / run_id
    for m in models.values():
        m.save(str(out_dir))
    manifest = {
        "spec_version": SPEC_VERSION, "fs_version": FEATURE_SET_VERSION,
        "label_version": LABEL_VERSION, "feature_names": cols,
        "feature_hash_sha256": feature_hash(), "seed": SEED,
        "train_start": str(pd.to_datetime(ds["session_date"]).min().date()),
        "train_end": str(pd.to_datetime(ds["session_date"]).max().date()),
        "walk_forward": asdict(wf_cfg), "folds": fold_reports,
        "hyperparameters": {"xgb": XGB_PARAMS, "lgbm": LGBM_PARAMS},
        "library_versions": {"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
                             "sklearn": sklearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__},
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Saved models and manifest to %s", out_dir)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Purged walk-forward training (XGBoost + LightGBM).")
    p.add_argument("--bars", type=Path, default=Path("data/raw/bars_1d"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/models"))
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--min-train-years", type=int, default=5)
    p.add_argument("--holdout-months", type=int, default=12)
    p.add_argument("--warmup", type=int, default=252)
    a = p.parse_args(argv)
    cfg = WalkForwardConfig(min_train_years=a.min_train_years, holdout_months=a.holdout_months)
    run_training(read_bars(a.bars), a.run_id, a.out, cfg, a.warmup, n_jobs=a.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
