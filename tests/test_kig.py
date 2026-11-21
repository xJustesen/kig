"""Acceptance tests at the four seams: summary pass, aggregation, report data and extension contracts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import toy
from kig.aggregate import group_view
from kig.core import Level, Run, explainer_for, register_explainer, register_view, summarize
from kig.explainers import (
    CheckpointExplainer,
    LightGBMExplainer,
    LinearExplainer,
    OneStageExplainer,
    PrecomputedExplainer,
)
from kig.report import report_data
from kig.store import SCHEMA, Store

TOL = 1e-9


@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: float


class ClosedFormExplainer:
    n_stages = 1
    stage_label = "Model"

    def __init__(self, model):
        self.model = model

    def slices(self, X):
        yield np.column_stack([X * self.model.coef, np.full(len(X), self.model.intercept)])


class CoefView:
    name = "Coefficients"
    js = "export function render(container, payload, api) { container.textContent = payload.coef.join(', '); }"

    def applies_to(self, run_info):
        return run_info["model_kind"] == "LinearModel"

    def collect(self, run, explainer, data):
        return dict(coef=run.model.coef.tolist())


register_explainer(LinearModel, ClosedFormExplainer)
register_view(CoefView())

LINEAR_FEATURES = ["a", "b", "c"]
LINEAR_LEVELS = [Level("Fold", ordered=False)]


@pytest.fixture(scope="module")
def linear():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([1.0, -0.5, 0.0]) + rng.normal(size=300)
    runs = [
        Run(
            ("fold=1",),
            LinearModel(np.array([0.9, -0.4, 0.3]), 0.1),
            dict(train=np.arange(0, 150), valid=np.arange(150, 300)),
        ),
        Run(
            ("fold=2",),
            LinearModel(np.array([1.1, -0.6, -0.2]), -0.1),
            dict(train=np.arange(140, 300), valid=np.arange(0, 150)),
        ),
    ]
    return X, y, runs


def linear_store(linear, path, **kwargs):
    X, y, runs = linear
    store = Store(path)
    computed = summarize(runs, X, y, store, levels=LINEAR_LEVELS, features=LINEAR_FEATURES, loss=toy.mse_loss, **kwargs)
    return store, computed


def tables(store):
    return {name: store.read(name) for name in SCHEMA}


@pytest.fixture(scope="module")
def toy_p09(tmp_path_factory):
    X, y, period = toy.make_data(seed=0)
    runs = toy.make_runs(X, y, period, steps=[9])
    store = Store(tmp_path_factory.mktemp("toy"))
    summarize(runs, X, y, store, levels=toy.LEVELS, features=toy.FEATURES, loss=toy.mse_loss, max_rows=None)
    return X, y, runs, store


@pytest.mark.parametrize("k", [10, toy.ROUNDS])
def test_first_k_slices_equal_the_attributions_of_the_model_with_k_stages(toy_p09, k):
    X, _, runs, _ = toy_p09
    rows = X[runs[0].sets["valid"]]
    slices = LightGBMExplainer(runs[0].model).slices(rows)
    total = sum(s for _, s in zip(range(k), slices))
    assert np.abs(total - runs[0].model.predict(rows, pred_contrib=True, num_iteration=k)).max() < TOL


def test_last_loss_sum_equals_the_loss_of_a_direct_prediction(toy_p09):
    X, y, runs, store = toy_p09
    idx = runs[0].sets["valid"]
    loss = store.read("stage_loss").query("run_id == 'wf=p09' and set == 'valid'").sort_values("stage").loss_sum
    assert abs(loss.iloc[-1] - toy.mse_loss(y[idx], runs[0].model.predict(X[idx])).sum()) < TOL


def test_loss_reductions_of_each_stage_add_up_to_the_loss_change(toy_p09):
    _, y, runs, store = toy_p09
    stage_loss = store.read("stage_loss").query("run_id == 'wf=p09' and set == 'valid'").sort_values("stage")
    gain = store.read("summary").query("run_id == 'wf=p09' and set == 'valid'").groupby("stage").gain_sum.sum()
    before = np.concatenate([[toy.mse_loss(y[runs[0].sets["valid"]], 0.0).sum()], stage_loss.loss_sum.to_numpy()[:-1]])
    change = stage_loss.loss_sum.to_numpy() - before
    assert np.abs(gain.to_numpy() + stage_loss.base_gain_sum.to_numpy() + change).max() < TOL


def test_one_booster_in_more_than_one_run_gives_the_same_sums_with_parallel_runs(toy_p09, tmp_path):
    X, y, runs, _ = toy_p09
    shared = [Run((f"wf=p{i:02d}",), runs[0].model, dict(valid=np.arange(i * 500, (i + 1) * 500))) for i in range(8)]
    stores = [Store(tmp_path / str(n_jobs)) for n_jobs in (1, 4)]
    for store, n_jobs in zip(stores, (1, 4)):
        summarize(shared, X, y, store, levels=toy.LEVELS, features=toy.FEATURES, loss=toy.mse_loss, n_jobs=n_jobs)
    pd.testing.assert_frame_equal(stores[0].read("summary"), stores[1].read("summary"))


def test_toy_regime_is_positive_in_the_nested_folds_and_negative_on_forward_validation(toy_p09):
    view = group_view(toy_p09[3], ("wf=p09",)).query("set == 'valid' and feature == 'regime'")
    folds = view[view.child.notna()].gain
    pool = view[view.child.isna() & (view.source == "nested")].gain.item()
    forward = view[view.child.isna() & (view.source == "own")].gain.item()
    assert (folds > 0).sum() == 3
    assert pool > 0.5 and forward < -0.5


def test_pool_of_a_prefix_equals_a_direct_computation_on_the_pooled_rows(linear, tmp_path):
    X, y, runs = linear
    store, _ = linear_store(linear, tmp_path, max_rows=None)
    direct, n = np.zeros(3), 0
    for run in runs:
        idx = run.sets["valid"]
        phi, a = X[idx] * run.model.coef, run.model.intercept
        b = a + phi.sum(1)
        mean_grad = (toy.mse_loss(y[idx], b) - toy.mse_loss(y[idx], a)) / (b - a)
        direct -= (phi * mean_grad[:, None]).sum(0)
        n += len(idx)
    pool = group_view(store).query("child.isna() and source == 'nested' and set == 'valid'").set_index("feature")
    assert np.abs(pool.gain[LINEAR_FEATURES].to_numpy() - direct / n).max() < TOL


def test_linear_model_with_one_stage_goes_through_the_report_data(linear, tmp_path):
    X, y, runs = linear
    store, _ = linear_store(linear, tmp_path)
    data = report_data(store)
    run = data["runs"]["fold=1"]
    idx = runs[0].sets["valid"]
    raw = X[idx] @ runs[0].model.coef + runs[0].model.intercept
    assert run["n_stages"] == 1 and run["stage_label"] == "Model"
    assert run["loss"]["valid"] == pytest.approx([toy.mse_loss(y[idx], raw).mean()], rel=1e-5)
    assert data["nodes"][""]["children"] == ["fold=1", "fold=2"]
    assert data["nodes"]["fold=1"]["own"]["valid"]["gain"] == pytest.approx([row[0] for row in run["gain"]["valid"]])


def test_payload_of_a_model_view_is_in_the_report_data_of_the_runs_it_applies_to(linear, toy_p09, tmp_path):
    store, _ = linear_store(linear, tmp_path)
    views = report_data(store)["runs"]["fold=2"]["views"]
    assert views == [dict(name="Summary", payload={}), dict(name="Coefficients", payload=dict(coef=[1.1, -0.6, -0.2]))]
    assert [v["name"] for v in report_data(toy_p09[3])["runs"]["wf=p09"]["views"]] == ["Stages", "Summary"]


def test_overlap_counts_the_rows_in_both_the_train_set_and_the_validation_set(linear, tmp_path):
    store, _ = linear_store(linear, tmp_path)
    sets = store.read("sets").set_index(["run_id", "set"])
    assert sets.overlap_with_train[("fold=1", "valid")] == 0
    assert sets.overlap_with_train[("fold=2", "valid")] == 10


def test_same_seed_gives_the_same_store(linear, tmp_path):
    first, _ = linear_store(linear, tmp_path / "1", max_rows=50, seed=3)
    second, _ = linear_store(linear, tmp_path / "2", max_rows=50, seed=3)
    assert first.read("sets").n.max() == 50
    for name in SCHEMA:
        pd.testing.assert_frame_equal(first.read(name), second.read(name))


def test_second_summary_pass_with_the_same_inputs_computes_no_run(linear, tmp_path):
    _, first = linear_store(linear, tmp_path)
    before = tables(Store(tmp_path))
    store, second = linear_store(linear, tmp_path)
    assert first == ["fold=1", "fold=2"] and second == []
    for name in SCHEMA:
        pd.testing.assert_frame_equal(store.read(name), before[name])


def test_row_weights_give_weighted_sums_and_counts(linear, tmp_path):
    plain, _ = linear_store(linear, tmp_path / "1")
    weighted, _ = linear_store(linear, tmp_path / "2", weights=np.full(300, 2.0))
    assert np.allclose(weighted.read("sets").n, 2 * plain.read("sets").n)
    assert np.allclose(weighted.read("summary").gain_sum, 2 * plain.read("summary").gain_sum)


def test_cross_evaluation_adds_the_validation_rows_of_the_sibling_runs(linear, tmp_path):
    store, _ = linear_store(linear, tmp_path, cross_eval=True)
    sets = store.read("sets").set_index(["run_id", "set"])
    assert sets.n[("fold=1", "valid@fold=2")] == sets.n[("fold=2", "valid")]


def test_frame_gives_the_store_of_its_array_and_its_column_names(linear, tmp_path):
    X, y, runs = linear
    store, _ = linear_store(linear, tmp_path / "1")
    other = Store(tmp_path / "2")
    summarize(runs, pd.DataFrame(X, columns=LINEAR_FEATURES), y, other, levels=LINEAR_LEVELS, loss=toy.mse_loss)
    for name in SCHEMA:
        pd.testing.assert_frame_equal(other.read(name), store.read(name))


def test_categorical_column_of_a_frame_goes_to_the_booster_with_its_dtype(tmp_path):
    X, y, _ = toy.make_data(seed=0)
    frame = pd.DataFrame(X[:1000, :2], columns=["a", "b"]).assign(c=pd.Categorical(np.where(X[:1000, 4] > 0, "u", "d")))
    model = pytest.importorskip("lightgbm").LGBMRegressor(n_estimators=5, verbose=-1).fit(frame, y[:1000])
    idx = np.arange(500, 1000)
    store = Store(tmp_path)
    summarize([Run(("m",), model, {"valid": idx})], frame, y, store, levels=LINEAR_LEVELS, loss=toy.mse_loss)
    direct = toy.mse_loss(y[idx], model.predict(frame.iloc[idx])).sum()
    assert store.read("stage_loss").loss_sum.iloc[-1] == pytest.approx(direct)


def test_array_with_no_feature_names_is_refused(linear, tmp_path):
    X, y, runs = linear
    with pytest.raises(TypeError):
        summarize(runs, X, y, Store(tmp_path), levels=LINEAR_LEVELS, loss=toy.mse_loss)


def test_precomputed_attributions_give_the_sums_of_the_model(linear, tmp_path):
    X, y, runs = linear
    store, _ = linear_store(linear, tmp_path / "1")
    pre = [
        Run(
            r.path,
            None,
            r.sets,
            explainer=PrecomputedExplainer(np.column_stack([X * r.model.coef, np.full(300, r.model.intercept)])),
        )
        for r in runs
    ]
    other = Store(tmp_path / "2")
    summarize(pre, X, y, other, levels=LINEAR_LEVELS, features=LINEAR_FEATURES, loss=toy.mse_loss)
    pd.testing.assert_frame_equal(other.read("summary"), store.read("summary"))


def test_group_view_refuses_a_mix_of_staged_and_one_stage_runs(linear, tmp_path):
    X, y, runs = linear
    staged = Run(("fold=3",), None, runs[0].sets, explainer=PrecomputedExplainer(np.zeros((2, 300, 4))))
    store = Store(tmp_path)
    summarize([*runs, staged], X, y, store, levels=LINEAR_LEVELS, features=LINEAR_FEATURES, loss=toy.mse_loss)
    with pytest.raises(ValueError):
        group_view(store)


def lightgbm_wrapper(X, y):
    model = pytest.importorskip("lightgbm").LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
    return (
        model,
        lambda rows: model.predict(rows, raw_score=True),
        lambda rows, k: model.booster_.predict(rows, pred_contrib=True, num_iteration=k),
        1e-9,
    )


def xgboost_booster(X, y):
    xgb = pytest.importorskip("xgboost")
    model = xgb.train(dict(max_depth=3), xgb.DMatrix(X, y), num_boost_round=20)
    return (
        model,
        lambda rows: model.predict(xgb.DMatrix(rows), output_margin=True),
        lambda rows, k: model.predict(xgb.DMatrix(rows), pred_contribs=True, iteration_range=(0, k)),
        1e-4,
    )


def xgboost_wrapper(X, y):
    xgb = pytest.importorskip("xgboost")
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3).fit(X, y > 0)
    return (
        model,
        lambda rows: model.predict(rows, output_margin=True),
        lambda rows, k: model.get_booster().predict(xgb.DMatrix(rows), pred_contribs=True, iteration_range=(0, k)),
        1e-4,
    )


def catboost_model(X, y):
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostRegressor(iterations=20, depth=3, verbose=0, allow_writing_files=False).fit(X, y)

    def truncated(rows, k):
        first = model.copy()
        first.shrink(ntree_end=k)
        return first.get_feature_importance(type="ShapValues", data=catboost.Pool(rows))

    return model, lambda rows: model.predict(rows, prediction_type="RawFormulaVal"), truncated, 1e-9


def sklearn_model(module, name, **params):
    def family(X, y):
        cls = getattr(pytest.importorskip(f"sklearn.{module}"), name)
        if module == "ensemble":
            pytest.importorskip("shap")
        if name.endswith("Classifier") or name == "LogisticRegression":
            model = cls(**params).fit(X, y > 0)
            return model, model.decision_function, None, 1e-9
        model = cls(**params).fit(X, y)
        return model, model.predict, None, 1e-9

    return pytest.param(family, id=name)


FAMILIES = [
    lightgbm_wrapper,
    xgboost_booster,
    xgboost_wrapper,
    catboost_model,
    sklearn_model("linear_model", "Ridge"),
    sklearn_model("linear_model", "LogisticRegression"),
    sklearn_model("linear_model", "SGDRegressor"),
    sklearn_model("ensemble", "RandomForestRegressor", n_estimators=10, min_samples_leaf=5),
    sklearn_model("ensemble", "ExtraTreesRegressor", n_estimators=10, min_samples_leaf=5),
    sklearn_model("ensemble", "GradientBoostingRegressor", n_estimators=20),
    sklearn_model("ensemble", "GradientBoostingClassifier", n_estimators=20),
    sklearn_model("ensemble", "HistGradientBoostingRegressor", max_iter=20),
    sklearn_model("ensemble", "HistGradientBoostingClassifier", max_iter=20),
]


@pytest.mark.parametrize("family", FAMILIES)
def test_slices_of_each_model_family_add_up_to_the_raw_score(family):
    X, y, _ = toy.make_data(seed=0)
    X, y = X[:1000], y[:1000]
    model, raw, truncated, tol = family(X, y)
    explainer = explainer_for(Run(("m",), model, {}))
    assert np.abs(sum(explainer.slices(X)).sum(1) - raw(X)).max() < tol
    if truncated:
        first = sum(s for _, s in zip(range(5), explainer.slices(X)))
        assert explainer.n_stages == 20
        assert np.abs(first - truncated(X, 5)).max() < tol


def lightgbm_classifier(X, y):
    model = pytest.importorskip("lightgbm").LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y > 0)
    return model, lambda rows: model.predict(rows, raw_score=True), None, 1e-9


def catboost_classifier(X, y):
    catboost = pytest.importorskip("catboost")
    model = catboost.CatBoostClassifier(iterations=20, depth=3, verbose=0, allow_writing_files=False).fit(X, y > 0)
    return model, lambda rows: model.predict(rows, prediction_type="RawFormulaVal"), None, 1e-9


@pytest.mark.parametrize("family", [*FAMILIES, lightgbm_classifier, catboost_classifier])
def test_summary_pass_with_no_loss_uses_the_loss_and_the_gradient_of_the_model_objective(family, tmp_path):
    X, y, _ = toy.make_data(seed=0)
    X, y = X[:1000], y[:1000]
    model, raw, _, _ = family(X, y)
    binary = len(getattr(model, "classes_", ())) == 2
    label, score = (y > 0 if binary else y), raw(X).astype(float)
    direct = np.logaddexp(0.0, score) - label * score if binary else toy.mse_loss(y, score)
    run = Run(("m",), model, {"valid": np.arange(1000)})
    store = Store(tmp_path)
    summarize([run], X, label, store, levels=LINEAR_LEVELS, features=toy.FEATURES, max_rows=None)
    assert store.read("stage_loss").loss_sum.iloc[-1] == pytest.approx(direct.sum(), rel=1e-4)
    loss, grad = explainer_for(run).loss_and_grad
    slope = (loss(label, score + 1e-6) - loss(label, score - 1e-6)) / 2e-6
    assert np.abs(grad(label, score) - slope).max() < 1e-4


def custom_objective(label, score):
    return score - label, np.ones_like(label)


CUSTOM = [
    ("lightgbm", lambda lib: lib.LGBMRegressor(n_estimators=2, verbose=-1, objective=custom_objective)),
    ("xgboost", lambda lib: lib.XGBRegressor(n_estimators=2, objective=custom_objective)),
]


@pytest.mark.parametrize("library, make", CUSTOM, ids=[library for library, _ in CUSTOM])
def test_summary_pass_with_no_loss_refuses_a_custom_objective(library, make, tmp_path):
    X, y, _ = toy.make_data(seed=0)
    run = Run(("m",), make(pytest.importorskip(library)).fit(X[:300], y[:300]), {"valid": np.arange(300)})
    with pytest.raises(ValueError, match="give `loss`"):
        summarize([run], X, y, Store(tmp_path), levels=LINEAR_LEVELS, features=toy.FEATURES)


def test_summary_pass_with_no_loss_refuses_runs_with_different_losses(linear, tmp_path):
    X, y, _ = linear
    lm = pytest.importorskip("sklearn.linear_model")
    runs = [
        Run(("fold=1",), lm.Ridge().fit(X, y), {"valid": np.arange(300)}),
        Run(("fold=2",), lm.LogisticRegression().fit(X, y > 0), {"valid": np.arange(300)}),
    ]
    with pytest.raises(ValueError, match="one loss"):
        summarize(runs, X, y, Store(tmp_path), levels=LINEAR_LEVELS, features=LINEAR_FEATURES)


MULTICLASS = [
    ("lightgbm", lambda lib: lib.LGBMClassifier(n_estimators=2, verbose=-1)),
    ("xgboost", lambda lib: lib.XGBClassifier(n_estimators=2)),
    ("catboost", lambda lib: lib.CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False)),
    ("sklearn.linear_model", lambda lib: lib.LogisticRegression()),
    ("sklearn.ensemble", lambda lib: lib.HistGradientBoostingClassifier(max_iter=2)),
]


@pytest.mark.parametrize("library, make", MULTICLASS, ids=[library for library, _ in MULTICLASS])
def test_each_model_family_refuses_a_multiclass_model(library, make):
    X, y, _ = toy.make_data(seed=0)
    model = make(pytest.importorskip(library)).fit(X[:300], np.digitize(y[:300], [-1.0, 1.0]))
    with pytest.raises(TypeError):
        explainer_for(Run(("m",), model, {}))


def test_checkpoint_slices_add_up_to_the_last_checkpoint(linear):
    X, _, runs = linear
    checkpoints = [ClosedFormExplainer(run.model) for run in runs]
    explainer = CheckpointExplainer(checkpoints)
    assert explainer.n_stages == 2
    assert np.abs(sum(explainer.slices(X)) - next(checkpoints[-1].slices(X))).max() < TOL


def test_linear_reference_moves_the_base_term_and_keeps_the_raw_score(linear):
    X, _, _ = linear
    model = pytest.importorskip("sklearn.linear_model").Ridge().fit(X, X @ np.array([1.0, -0.5, 0.0]) + 2.0)
    (zero,), (shifted,) = LinearExplainer(model).slices(X), LinearExplainer(model, reference=np.ones(3)).slices(X)
    assert np.abs(zero.sum(1) - shifted.sum(1)).max() < TOL
    assert shifted[0, -1] == pytest.approx(model.intercept_ + model.coef_.sum())


def test_one_stage_form_of_a_staged_run_goes_in_a_group_view_with_one_stage_runs(linear, tmp_path):
    X, y, runs = linear
    staged = PrecomputedExplainer(np.ones((2, 300, 4)))
    store = Store(tmp_path)
    summarize(
        [*runs, Run(("fold=3",), None, runs[0].sets, explainer=OneStageExplainer(staged))],
        X,
        y,
        store,
        levels=LINEAR_LEVELS,
        features=LINEAR_FEATURES,
        loss=toy.mse_loss,
    )
    assert set(group_view(store).child.dropna()) == {"fold=1", "fold=2", "fold=3"}
