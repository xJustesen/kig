"""Weights & Biases integration: log the store and the report to a run, and load the store of an artifact."""

from __future__ import annotations

import tempfile
from pathlib import Path

import wandb

from kig.aggregate import summary_metrics
from kig.report import build_report
from kig.store import Store

ARTIFACT_TYPE = "kig-store"


def log_kig(store: Store, run: wandb.Run | None = None) -> None:
    """Log the store directory as an artifact, the report and the summary metrics to a run (wandb.Run | None: the active run)."""
    run = wandb.run if run is None else run
    artifact = wandb.Artifact("kig-store", type=ARTIFACT_TYPE)
    artifact.add_dir(str(store.path))
    run.log_artifact(artifact)
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.html"
        build_report(store, report)
        run.log({"kig/report": wandb.Html(report, inject=False)})
    run.summary.update(summary_metrics(store))


def load_store(artifact_name: str) -> Store:
    """Download a store artifact (str, for example "entity/project/kig-store:latest") and return it (Store) as a local copy."""
    return Store(wandb.Api().artifact(artifact_name, type=ARTIFACT_TYPE).download())
