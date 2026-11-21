"""Explainers: one adapter for each model type. The core calls them through the `Explainer` contract only."""

from __future__ import annotations

import json
import threading
from typing import Iterator

import numpy as np

from kig.core import register_explainer


# A LightGBM booster keeps the iteration range of a prediction in its own state, so one booster in more
# than one run gives wrong slices when threads predict at the same time.
_LIGHTGBM_LOCK = threading.Lock()


class LightGBMExplainer:
    """Slices of a LightGBM booster (lightgbm.Booster): one slice for each boosting round."""

    stage_label = "Boosting round"

    def __init__(self, model):
        if model.num_model_per_iteration() > 1:
            raise TypeError("multiclass models are not supported")
        if model.dump_model(num_iteration=1)["average_output"]:
            raise TypeError("a booster that averages its trees (boosting='rf') has no additive stage axis")
        self.model = model
        self.n_stages = model.best_iteration if model.best_iteration > 0 else model.current_iteration()

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) of each boosting round in the raw score space."""
        for t in range(self.n_stages):
            with _LIGHTGBM_LOCK:
                s = self.model.predict(X, pred_contrib=True, start_iteration=t, num_iteration=1)
            yield s


class XGBoostExplainer:
    """Slices of an XGBoost booster (xgboost.core.Booster): one slice for each round up to `best_iteration` (int | None)."""

    stage_label = "Boosting round"

    def __init__(self, model, best_iteration: int | None = None, missing: float = np.nan):
        learner = json.loads(model.save_config())["learner"]
        if max(int(learner["learner_model_param"][key]) for key in ("num_class", "num_target")) > 1:
            raise TypeError("multiclass and multi-output models are not supported")
        self.model = model
        self.missing = missing
        if learner["gradient_booster"]["name"] == "gblinear":
            self.n_stages = 1
        elif best_iteration is None:
            self.n_stages = model.num_boosted_rounds()
        else:
            self.n_stages = best_iteration + 1

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) of each boosting round in the raw score space."""
        import xgboost

        data = xgboost.DMatrix(
            X, missing=self.missing, feature_names=self.model.feature_names, feature_types=self.model.feature_types
        )
        # A gblinear booster ignores the range.
        yield self.model.predict(data, pred_contribs=True, iteration_range=(0, 1))
        # XGBoost adds the base score of the model to the base term of each call. A zero margin removes it
        # after the first round.
        data.set_base_margin(np.zeros(len(X)))
        for t in range(1, self.n_stages):
            # XGBoost refuses a contribution range that does not start at round 0, so each later round is a
            # slice of the model.
            yield self.model[t : t + 1].predict(data, pred_contribs=True)


class CatBoostExplainer:
    """Slices of a CatBoost model (catboost.core.CatBoost) with symmetric trees: one slice for each boosting round."""

    stage_label = "Boosting round"

    def __init__(self, model):
        if np.ndim(model.get_scale_and_bias()[1]):
            raise TypeError("multiclass and multi-output models are not supported")
        self.model = model
        self.n_stages = model.tree_count_

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) of each boosting round in the raw score space."""
        from catboost import Pool

        pool = Pool(X, cat_features=self.model.get_cat_feature_indices())
        for t in range(self.n_stages):
            # CatBoost has no tree range for SHAP values. shrink keeps the model bias only in the copy that starts at tree 0.
            tree = self.model.copy()
            tree.shrink(ntree_start=t, ntree_end=t + 1)
            yield tree.get_feature_importance(type="ShapValues", data=pool)


class LinearExplainer:
    """Slices of a linear model with `coef_` and `intercept_` (scikit-learn): one slice in closed form."""

    n_stages = 1
    stage_label = "Model"

    def __init__(self, model, reference: np.ndarray | None = None):
        coef = np.atleast_2d(model.coef_)
        if len(coef) > 1:
            raise TypeError("multiclass and multi-output models are not supported")
        self.coef = coef[0]
        self.reference = np.zeros(len(self.coef)) if reference is None else np.asarray(reference, dtype=float)
        self.base = np.asarray(model.intercept_).item() + self.coef @ self.reference

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) relative to the reference point in the raw score space."""
        yield np.column_stack([self.coef * (X - self.reference), np.full(len(X), self.base)])


class ShapExplainer:
    """One slice from an explainer of the `shap` library (for example shap.TreeExplainer) in the raw score space."""

    n_stages = 1

    def __init__(self, shap_explainer, stage_label: str = "Model"):
        if getattr(shap_explainer, "model_output", "raw") != "raw":
            raise ValueError("the shap explainer must give attributions in the raw score space (model_output='raw')")
        if np.size(shap_explainer.expected_value) != 1:
            raise TypeError("multiclass and multi-output models are not supported")
        self.shap_explainer = shap_explainer
        self.stage_label = stage_label

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the SHAP values (np.ndarray) with the expected value as the base term."""
        yield np.column_stack([self.shap_explainer.shap_values(X), np.full(len(X), self.shap_explainer.expected_value)])


def _tree_shap(model) -> ShapExplainer:
    """Return the explainer (ShapExplainer) of a tree model of scikit-learn from shap.TreeExplainer."""
    if len(getattr(model, "classes_", ())) > 2:
        raise TypeError("multiclass and multi-output models are not supported")
    # shap reads these models wrong, and its additivity check compares with its own copy of the trees.
    if (
        getattr(model, "loss", None) == "exponential"
        or np.any(getattr(model, "is_categorical_", None))
        or (hasattr(model, "classes_") and getattr(getattr(model, "init_", None), "strategy", "prior") != "prior")
    ):
        raise ValueError(
            "shap gives no exact raw score for loss='exponential', for categorical features "
            "or for a classifier init other than the class prior"
        )
    import shap

    return ShapExplainer(shap.TreeExplainer(model))


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


class CheckpointExplainer:
    """Slices of a model with checkpoints: the difference between adjacent one-stage explainers (list[Explainer])."""

    def __init__(self, explainers: list, stage_label: str = "Checkpoint"):
        self.explainers = explainers
        self.n_stages = len(explainers)
        self.stage_label = stage_label
        self.by_index = getattr(explainers[0], "by_index", False)

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the change (np.ndarray) of the attributions from one checkpoint to the next."""
        before = 0.0
        for explainer in self.explainers:
            (now,) = explainer.slices(X)
            yield now - before
            before = now


class OneStageExplainer:
    """One-stage form of a staged explainer (Explainer), for a report that also has one-stage runs."""

    n_stages = 1
    stage_label = "Model"

    def __init__(self, explainer):
        self.explainer = explainer
        self.by_index = getattr(explainer, "by_index", False)

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the sum (np.ndarray) of all slices of the staged explainer."""
        yield sum(self.explainer.slices(X))


def register_builtin_explainers() -> None:
    """Register the explainers that are part of the package."""
    register_explainer("lightgbm.basic.Booster", LightGBMExplainer)
    register_explainer("lightgbm.sklearn.LGBMModel", lambda model: LightGBMExplainer(model.booster_))
    register_explainer("xgboost.core.Booster", XGBoostExplainer)
    register_explainer(
        "xgboost.sklearn.XGBModel",
        lambda model: XGBoostExplainer(model.get_booster(), getattr(model, "best_iteration", None), model.missing),
    )
    register_explainer("catboost.core.CatBoost", CatBoostExplainer)
    for name in ("_base.LinearModel", "_base.LinearClassifierMixin", "_stochastic_gradient.BaseSGDRegressor"):
        register_explainer(f"sklearn.linear_model.{name}", LinearExplainer)
    for name in (
        "_forest.RandomForestRegressor",
        "_forest.ExtraTreesRegressor",
        "_gb.GradientBoostingRegressor",
        "_gb.GradientBoostingClassifier",
        "_hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor",
        "_hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
    ):
        register_explainer(f"sklearn.ensemble.{name}", _tree_shap)
