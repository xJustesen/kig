"""MLflow integration: log the store and the report to a run, and load the store of a run."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow

from kig.aggregate import summary_metrics
from kig.report import build_report
from kig.store import Store


def log_kig(store: Store, run_id: str | None = None, artifact_path: str = "kig") -> None:
    """Log the store directory, `report.html` and the summary metrics to a run (str | None: the active run)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.html"
        build_report(store, report)
        mlflow.log_artifact(str(report), artifact_path, run_id=run_id)
    mlflow.log_artifacts(str(store.path), f"{artifact_path}/store", run_id=run_id)
    mlflow.log_metrics(summary_metrics(store), run_id=run_id)


def load_store(run_id: str, artifact_path: str = "kig") -> Store:
    """Download the store artifact of a run (str) and return it (Store) as a local copy."""
    return Store(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{artifact_path}/store"))
