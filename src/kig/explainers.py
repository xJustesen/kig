"""Explainers: one adapter for each model type. The core calls them through the `Explainer` contract only."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from kig.core import register_explainer


class LightGBMExplainer:
    """Slices of a LightGBM booster (lightgbm.Booster): one slice for each boosting round."""
    stage_label = "Boosting round"

    def __init__(self, model):
        if model.dump_model(num_iteration=1)["average_output"]:
            raise TypeError("a booster that averages its trees (boosting='rf') has no additive stage axis")
        self.model = model
        self.n_stages = model.current_iteration()

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) of each boosting round in the raw score space."""
        for t in range(self.n_stages):
            yield self.model.predict(X, pred_contrib=True, start_iteration=t, num_iteration=1)


class PrecomputedExplainer:
    """Slices from arrays of the caller: (rows of the shared X, features + 1) or (stages, rows, features + 1)."""
    by_index = True

    def __init__(self, attributions: np.ndarray, stage_label: str = "Stage"):
        self.attributions = attributions if attributions.ndim == 3 else attributions[None]
        self.n_stages = len(self.attributions)
        self.stage_label = stage_label

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the slice (np.ndarray) of each stage for the row indices `X` (np.ndarray)."""
        for stage in self.attributions:
            yield stage[X]


def register_builtin_explainers() -> None:
    """Register the explainers that are part of the package."""
    register_explainer("lightgbm.basic.Booster", LightGBMExplainer)
