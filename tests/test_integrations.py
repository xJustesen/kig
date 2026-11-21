"""Tests of the integrations: each logger sends a store to a local tracker and reads it back."""

import mlflow
import numpy as np
import pandas as pd
import pytest
import toy

from kig.aggregate import group_view, summary_metrics
from kig.core import Level, Run, summarize
from kig.explainers import PrecomputedExplainer
from kig.integrations import mlflow as kig_mlflow
from kig.store import Store


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)
    attributions = np.column_stack([X * np.array([1.0, 0.5]), np.zeros(200)])
    run = Run(
        ("fold=1",),
        None,
        {"train": np.arange(100), "valid": np.arange(100, 200)},
        explainer=PrecomputedExplainer(attributions),
    )
    store = Store(tmp_path_factory.mktemp("store"))
    summarize(
        [run], X, y, store, levels=[Level("Fold", ordered=False)], features=["signal", "noise"], loss=toy.mse_loss
    )
    return store


def test_mlflow_run_gets_the_store_the_report_and_the_metrics(store, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    with mlflow.start_run() as run:
        kig_mlflow.log_kig(store)
    pd.testing.assert_frame_equal(group_view(kig_mlflow.load_store(run.info.run_id)), group_view(store))
    assert mlflow.artifacts.list_artifacts(run_id=run.info.run_id, artifact_path="kig")[0].path == "kig/report.html"
    assert mlflow.get_run(run.info.run_id).data.metrics == summary_metrics(store)
