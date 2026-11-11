"""XGBoost / LightGBM stage models with isotonic calibration (spec section 7).

Both models implement ``fit / predict_proba / save / load`` (``StageModel`` protocol).
``predict_proba`` returns calibrated P(y_long = 1). The baseline feature template has
no categorical columns; categorical handling arrives with ``sdf_fs_v1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

SEED = 42
EARLY_STOPPING_ROUNDS = 50

XGB_PARAMS: dict[str, object] = dict(
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    max_depth=4, learning_rate=0.03, n_estimators=1000, min_child_weight=50,
    subsample=0.7, colsample_bytree=0.6, reg_lambda=10.0, reg_alpha=1.0,
    gamma=0.0, max_cat_to_onehot=4, random_state=SEED,
)

LGBM_PARAMS: dict[str, object] = dict(
    objective="binary", metric="binary_logloss", boosting_type="gbdt",
    num_leaves=15, max_depth=-1, learning_rate=0.03, n_estimators=1000,
    min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_lambda=10.0, reg_alpha=1.0, cat_smooth=20, deterministic=True,
    force_row_wise=True, random_state=SEED, verbose=-1,
)


def _fit_calibrator(raw: np.ndarray, y: pd.Series, w: pd.Series) -> IsotonicRegression:
    cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal.fit(raw, y.to_numpy(), sample_weight=w.to_numpy())
    return cal


class XGBStageModel:
    name = "xgb"

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs
        self._booster: xgb.Booster | None = None
        self._best_iter: int | None = None
        self._calibrator: IsotonicRegression | None = None

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        assert self._booster is not None, "model not fitted"
        rng = (0, self._best_iter + 1) if self._best_iter is not None else (0, 0)
        return self._booster.predict(xgb.DMatrix(X), iteration_range=rng)

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series, w_val: pd.Series) -> None:
        clf = xgb.XGBClassifier(**XGB_PARAMS, n_jobs=self.n_jobs,
                                early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        clf.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[w_val], verbose=False)
        self._booster = clf.get_booster()
        self._best_iter = int(clf.best_iteration)
        self._calibrator = _fit_calibrator(self._raw(X_val), y_val, w_val)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        assert self._calibrator is not None, "model not fitted"
        return pd.Series(self._calibrator.predict(self._raw(X)), index=X.index, name="p_xgb")

    def save(self, path: str) -> None:
        assert self._booster is not None and self._calibrator is not None
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(out / "xgb.ubj")
        joblib.dump(self._calibrator, out / "xgb_calibrator.joblib")
        (out / "xgb_meta.json").write_text(json.dumps({"best_iteration": self._best_iter}))

    @classmethod
    def load(cls, path: str) -> "XGBStageModel":
        src = Path(path)
        model = cls()
        model._booster = xgb.Booster()
        model._booster.load_model(src / "xgb.ubj")
        model._best_iter = json.loads((src / "xgb_meta.json").read_text())["best_iteration"]
        model._calibrator = joblib.load(src / "xgb_calibrator.joblib")
        return model


class LGBMStageModel:
    name = "lgbm"

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs
        self._booster: lgb.Booster | None = None
        self._best_iter: int | None = None
        self._calibrator: IsotonicRegression | None = None

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        assert self._booster is not None, "model not fitted"
        return np.asarray(self._booster.predict(X, num_iteration=self._best_iter))

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series, w_val: pd.Series) -> None:
        clf = lgb.LGBMClassifier(**LGBM_PARAMS, n_jobs=self.n_jobs)
        clf.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        self._booster = clf.booster_
        self._best_iter = int(clf.best_iteration_) if clf.best_iteration_ else None
        self._calibrator = _fit_calibrator(self._raw(X_val), y_val, w_val)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        assert self._calibrator is not None, "model not fitted"
        return pd.Series(self._calibrator.predict(self._raw(X)), index=X.index, name="p_lgbm")

    def save(self, path: str) -> None:
        assert self._booster is not None and self._calibrator is not None
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        # Persist only the best iterations so a reloaded model predicts identically.
        self._booster.save_model(str(out / "lgbm.txt"), num_iteration=self._best_iter)
        joblib.dump(self._calibrator, out / "lgbm_calibrator.joblib")

    @classmethod
    def load(cls, path: str) -> "LGBMStageModel":
        src = Path(path)
        model = cls()
        model._booster = lgb.Booster(model_file=str(src / "lgbm.txt"))
        model._best_iter = None
        model._calibrator = joblib.load(src / "lgbm_calibrator.joblib")
        return model


def ensemble_proba(probs: Mapping[str, pd.Series],
                   weights: Mapping[str, float] | None = None) -> pd.Series:
    """Weighted mean of calibrated probabilities (default 0.5 / 0.5)."""
    weights = weights or {name: 1.0 / len(probs) for name in probs}
    total = sum(weights.values())
    out = sum(probs[name] * (weights[name] / total) for name in probs)
    return out.rename("p_long")
