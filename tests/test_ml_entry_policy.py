import json

import numpy as np
import pandas as pd
import pytest

from api import server


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    return {
        "fs_version": server.FEATURE_SET_VERSION,
        "feature_names": list(server.FEATURE_COLUMNS),
        "feature_hash_sha256": server.feature_hash(),
    }


@pytest.mark.parametrize("problem", ["missing", "version", "order", "hash"])
def test_incompatible_manifest_blocks_artifact_loading(manifest, monkeypatch, problem):
    if problem != "missing":
        if problem == "version":
            manifest["fs_version"] = "baseline_p12_v1"
        elif problem == "order":
            manifest["feature_names"].reverse()
        else:
            manifest["feature_hash_sha256"] = "invalid"
        (server.MODEL_DIR / "manifest.json").write_text(json.dumps(manifest))

    def unexpected_load(*args):
        pytest.fail("Artifact loader called before compatibility was established")

    monkeypatch.setattr(server.XGBStageModel, "load", unexpected_load)
    monkeypatch.setattr(server.LGBMStageModel, "load", unexpected_load)
    models, reasons = server.load_models()
    assert models == {}
    assert reasons


def test_compatible_manifest_loads_both_models(manifest, monkeypatch):
    (server.MODEL_DIR / "manifest.json").write_text(json.dumps(manifest))
    xgb, lgbm = object(), object()
    monkeypatch.setattr(server.XGBStageModel, "load", lambda path: xgb)
    monkeypatch.setattr(server.LGBMStageModel, "load", lambda path: lgbm)
    assert server.load_models() == ({"xgb": xgb, "lgbm": lgbm}, [])


class Model:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        if self.probabilities is None:
            raise ValueError("Inference failed")
        return pd.Series(self.probabilities, index=features.index)


@pytest.fixture
def bars(monkeypatch):
    bars = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC", "DDD"],
        "session_date": pd.to_datetime(["2026-11-12"] * 3 + ["2026-11-11"]),
    })
    features = bars.copy()
    for name in server.FEATURE_COLUMNS:
        features[name] = 0.02
    features.loc[2, "atr14_pct"] = np.nan
    monkeypatch.setattr(server, "build_features", lambda data: features)
    monkeypatch.setattr(server, "compute_tradable", lambda data, cfg: data.assign(tradable=True))
    monkeypatch.setattr(server, "load_bars", lambda: bars)
    monkeypatch.setattr(server, "_account_equity", lambda: None)
    return bars


@pytest.mark.parametrize(
    ("xgb", "lgbm", "ranked", "status"),
    [
        ([0.7, 0.8], [0.8, 0.9], ["BBB", "AAA"], "xgb+lgbm"),
        ([0.7, 0.8], [0.8, np.inf], ["AAA"], "partial-predictions"),
        ([0.7, 0.8], [0.8, 1.01], ["AAA"], "partial-predictions"),
        ([0.7, 0.8], None, [], "model-unavailable"),
        (None, None, [], "model-unavailable"),
    ],
)
def test_dashboard_ranks_only_symbols_with_two_valid_scores(
    bars, monkeypatch, xgb, lgbm, ranked, status,
):
    monkeypatch.setattr(
        server, "load_models", lambda: ({"xgb": Model(xgb), "lgbm": Model(lgbm)}, []),
    )
    response = server.dashboard()
    assert response["modelStatus"] == status
    assert [row["symbol"] for row in response["signals"]] == ranked
    blocked = {row["symbol"]: row for row in response["blockedMlEntries"]}
    assert set(blocked) == set(bars["symbol"]) - set(ranked)
    assert all(row["reason"] for row in blocked.values())
    if xgb and lgbm is None:
        assert blocked["AAA"]["pXgb"] == 0.7
        assert blocked["AAA"]["pLgbm"] is None


def test_missing_market_data_blocks_dashboard_entries(monkeypatch):
    def missing_bars():
        raise server.DataContractError("No parquet files")

    monkeypatch.setattr(server, "load_bars", missing_bars)
    monkeypatch.setattr(server, "_account_equity", lambda: None)
    response = server.dashboard()
    assert response["signals"] == []
    assert response["asOf"] is None
    assert response["blockedMlEntries"][0]["reason"]

def test_missing_model_keeps_surviving_score_for_diagnostics(bars, monkeypatch):
    monkeypatch.setattr(
        server, "load_models",
        lambda: ({"lgbm": Model([0.8, 0.9])}, ["xgb: model artifact unavailable."]),
    )

    def unexpected_order(*args):
        pytest.fail("Dashboard must not submit an order")

    monkeypatch.setattr(server, "execute_order", unexpected_order)
    response = server.dashboard()
    assert response["signals"] == []
    blocked = next(row for row in response["blockedMlEntries"] if row["symbol"] == "AAA")
    assert blocked["pXgb"] is None
    assert blocked["pLgbm"] == 0.8
    assert "xgb" in blocked["reason"]
