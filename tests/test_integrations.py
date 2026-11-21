"""Tests of the integrations: each logger sends a store to a local tracker and reads it back."""

import shutil

import mlflow
import numpy as np
import pandas as pd
import pytest
import toy
import wandb

from kig.aggregate import group_view, summary_metrics
from kig.core import Level, Run, summarize
from kig.explainers import PrecomputedExplainer
from kig.integrations import mlflow as kig_mlflow
from kig.integrations import wandb as kig_wandb
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


def test_wandb_offline_run_gets_the_store_the_report_and_the_metrics(store, tmp_path, monkeypatch):
    for name in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_CONFIG_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    with wandb.init(mode="offline") as run:
        staged = []
        log_artifact = run.log_artifact
        monkeypatch.setattr(run, "log_artifact", lambda artifact: staged.append(artifact) or log_artifact(artifact))

        class Server:
            def artifact(self, name, type):
                return self

            def download(self):
                for entry in staged[0].manifest.entries.values():
                    shutil.copy(entry.local_path, tmp_path / entry.path)
                return tmp_path

        # An offline run cannot download an artifact, so the files that the run staged stand in for the server.
        monkeypatch.setattr(wandb, "Api", Server)
        kig_wandb.log_kig(store, run)
        pd.testing.assert_frame_equal(group_view(kig_wandb.load_store("kig-store:latest")), group_view(store))
        assert run.summary.get("kig/report") is not None
        assert {name: run.summary[name] for name in summary_metrics(store)} == summary_metrics(store)
