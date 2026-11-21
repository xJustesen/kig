"""kig: a diagnostic tool for fitted models."""

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kig.core import Run, summarize
from kig.explainers import register_builtin_explainers
from kig.report import build_report
from kig.store import Store
from kig.views import register_builtin_views

register_builtin_explainers()
register_builtin_views()


def diagnose(
    model,
    X,
    y,
    *,
    sets: dict[str, np.ndarray],
    out: str | Path,
    features: list[str] | None = None,
    loss: Callable | None = None,
):
    """Write the report of one model and its row sets (dict[str, np.ndarray]); return the result of `build_report`."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(tmp)
        summarize([Run((), model, sets)], X, y, store, levels=[], features=features, loss=loss)
        return build_report(store, out)
