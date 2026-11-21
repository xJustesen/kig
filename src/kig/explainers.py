"""Explainers: one adapter for each model type. The core calls them through the `Explainer` contract only."""

from __future__ import annotations

import json
import threading
from typing import Iterator

import numpy as np

from kig.core import Run, explainer_for, register_explainer


def _squared_error(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return half the squared error (np.ndarray) of each row."""
    return 0.5 * (raw - y) ** 2


def _squared_error_grad(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return the gradient (np.ndarray) of half the squared error in the raw score."""
    return raw - y


def _log_loss(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return the binary log loss (np.ndarray) of each row for labels 0 and 1 and a logit."""
    return np.logaddexp(0.0, raw) - y * raw


def _log_loss_grad(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return the gradient (np.ndarray) of the binary log loss in the logit."""
    return np.exp(-np.logaddexp(0.0, -raw)) - y


def _softmax_loss(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return the softmax cross-entropy (np.ndarray) of each row for class indices and scores (rows, classes)."""
    return np.logaddexp.reduce(raw, axis=1) - raw[np.arange(len(y)), y]


def _softmax_loss_grad(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return the gradient (np.ndarray) of the softmax cross-entropy in the score of each class."""
    grad = np.exp(raw - np.logaddexp.reduce(raw, axis=1, keepdims=True))
    grad[np.arange(len(y)), y] -= 1.0
    return grad


_SQUARED_ERROR = (_squared_error, _squared_error_grad)
_LOG_LOSS = (_log_loss, _log_loss_grad)
_SOFTMAX_LOSS = (_softmax_loss, _softmax_loss_grad)
_OBJECTIVES = {
    "regression": _SQUARED_ERROR,
    "binary sigmoid:1": _LOG_LOSS,
    "multiclass": _SOFTMAX_LOSS,
    "reg:squarederror": _SQUARED_ERROR,
    "binary:logistic": _LOG_LOSS,
    "multi:softprob": _SOFTMAX_LOSS,
    "multi:softmax": _SOFTMAX_LOSS,
    "RMSE": _SQUARED_ERROR,
    "Logloss": _LOG_LOSS,
    "squared_error": _SQUARED_ERROR,
    "log_loss": _LOG_LOSS,
}
_SKLEARN_MODELS_WITH_ONE_LOSS = {
    **dict.fromkeys(("LinearRegression", "Ridge", "Lasso", "ElasticNet"), "squared_error"),
    "LogisticRegression": "log_loss",
}


# A LightGBM booster keeps the iteration range of a prediction in its own state, so one booster in more
# than one run gives wrong slices when threads predict at the same time.
_LIGHTGBM_LOCK = threading.Lock()


class LightGBMExplainer:
    """Slices of a LightGBM booster (lightgbm.Booster): one slice for each boosting round."""

    stage_label = "Boosting round"

    def __init__(self, model):
        dump = model.dump_model(num_iteration=1)
        if dump["average_output"]:
            raise TypeError("a booster that averages its trees (boosting='rf') has no additive stage axis")
        self.model = model
        self.n_classes = model.num_model_per_iteration()
        self.loss_and_grad = _OBJECTIVES.get(dump.get("objective", "").split(" num_class:")[0])
        self.n_stages = model.best_iteration if model.best_iteration > 0 else model.current_iteration()

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) of each boosting round in the raw score space."""
        for t in range(self.n_stages):
            with _LIGHTGBM_LOCK:
                s = self.model.predict(X, pred_contrib=True, start_iteration=t, num_iteration=1)
            yield s if self.n_classes == 1 else s.reshape(len(s), self.n_classes, -1)


class XGBoostExplainer:
    """Slices of an XGBoost booster (xgboost.core.Booster): one slice for each round up to `best_iteration` (int | None)."""

    stage_label = "Boosting round"

    def __init__(self, model, best_iteration: int | None = None, missing: float = np.nan):
        learner = json.loads(model.save_config())["learner"]
        if int(learner["learner_model_param"]["num_target"]) > 1:
            raise TypeError("multi-output models are not supported")
        self.model = model
        self.n_scores = max(int(learner["learner_model_param"]["num_class"]), 1)
        self.missing = missing
        self.loss_and_grad = _OBJECTIVES.get(learner["objective"]["name"])
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
        data.set_base_margin(np.zeros((len(X), self.n_scores)))
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
        self.loss_and_grad = _OBJECTIVES.get(model.get_all_params()["loss_function"])

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the attributions (np.ndarray) of each boosting round in the raw score space."""
        from catboost import Pool

        pool = Pool(X, cat_features=self.model.get_cat_feature_indices())
        for t in range(self.n_stages):
            # CatBoost has no tree range for SHAP values. shrink keeps the model bias only in the copy that starts at tree 0.
            tree = self.model.copy()
            tree.shrink(ntree_start=t, ntree_end=t + 1)
            yield tree.get_feature_importance(type="ShapValues", data=pool)


def _sklearn_loss(model) -> tuple | None:
    """Return the loss and its gradient (tuple | None) of a scikit-learn model by its loss, criterion or class."""
    fixed = _SKLEARN_MODELS_WITH_ONE_LOSS.get(type(model).__name__)
    return _OBJECTIVES.get(getattr(model, "loss", None) or getattr(model, "criterion", None) or fixed)


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
        self.loss_and_grad = _sklearn_loss(model)

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

    explainer = ShapExplainer(shap.TreeExplainer(model))
    explainer.loss_and_grad = _sklearn_loss(model)
    return explainer


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
        self.loss_and_grad = getattr(explainer, "loss_and_grad", None)

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the sum (np.ndarray) of all slices of the staged explainer."""
        yield sum(self.explainer.slices(X))


class PipelineExplainer:
    """Slices of the last step of a pipeline (sklearn.pipeline.Pipeline) for the columns that its other steps give."""

    def __init__(self, model):
        self.transformer = model[:-1]
        self.explainer = explainer_for(Run((), model[-1], {}))
        self.features = list(self.transformer.get_feature_names_out())
        self.n_stages = self.explainer.n_stages
        self.stage_label = self.explainer.stage_label
        self.loss_and_grad = getattr(self.explainer, "loss_and_grad", None)

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield the slices (np.ndarray) of the last step for the transformed rows."""
        rows = self.transformer.transform(X)
        return self.explainer.slices(rows.toarray() if hasattr(rows, "toarray") else rows)


def _xgboost_wrapper(model) -> XGBoostExplainer:
    """Return the explainer (XGBoostExplainer) of a scikit-learn wrapper of XGBoost (xgboost.sklearn.XGBModel)."""
    explainer = XGBoostExplainer(model.get_booster(), getattr(model, "best_iteration", None), model.missing)
    # The booster of a wrapper with a custom objective reports the default objective.
    if callable(model.objective):
        explainer.loss_and_grad = None
    return explainer


def register_builtin_explainers() -> None:
    """Register the explainers that are part of the package."""
    register_explainer("lightgbm.basic.Booster", LightGBMExplainer)
    register_explainer("lightgbm.sklearn.LGBMModel", lambda model: LightGBMExplainer(model.booster_))
    register_explainer("xgboost.core.Booster", XGBoostExplainer)
    register_explainer("xgboost.sklearn.XGBModel", _xgboost_wrapper)
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
    register_explainer("sklearn.pipeline.Pipeline", PipelineExplainer)
