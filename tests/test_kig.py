"""Acceptance tests at the four seams: summary pass, aggregation, report data and extension contracts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import toy
from kig.aggregate import group_view
from kig.core import Level, Run, register_explainer, register_view, summarize
from kig.explainers import LightGBMExplainer, PrecomputedExplainer
from kig.report import report_data
from kig.store import SCHEMA, Store

TOL = 1e-9


@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: float


class LinearExplainer:
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


register_explainer(LinearModel, LinearExplainer)
register_view(CoefView())

LINEAR_FEATURES = ["a", "b", "c"]
LINEAR_LEVELS = [Level("Fold", ordered=False)]


@pytest.fixture(scope="module")
def linear():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([1.0, -0.5, 0.0]) + rng.normal(size=300)
    runs = [Run(("fold=1",), LinearModel(np.array([0.9, -0.4, 0.3]), 0.1), dict(train=np.arange(0, 150), valid=np.arange(150, 300))),
            Run(("fold=2",), LinearModel(np.array([1.1, -0.6, -0.2]), -0.1), dict(train=np.arange(140, 300), valid=np.arange(0, 150)))]
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


def test_precomputed_attributions_give_the_sums_of_the_model(linear, tmp_path):
    X, y, runs = linear
    store, _ = linear_store(linear, tmp_path / "1")
    pre = [Run(r.path, None, r.sets, explainer=PrecomputedExplainer(np.column_stack([X * r.model.coef, np.full(300, r.model.intercept)])))
           for r in runs]
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
