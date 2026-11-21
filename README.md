# kig

`kig` is a diagnostic tool for fitted models. It shows which features help or hurt, at which stage of the fit, and in which split.

The tool never fits a model and never makes a split. You give it fitted models, row indices and a loss function. Give it development data only.

## Use

```python
from kig.core import Level, Run, summarize
from kig.report import build_report
from kig.store import Store

runs = [
    Run(path=("wf=2025-03",), model=booster, sets={"train": idx_tr, "valid": idx_va}, note="refit"),
    Run(path=("wf=2025-03", "kfold=1"), model=fold_booster, sets={"train": idx_1, "valid": idx_2}),
]
levels = [Level("Walk-forward step", ordered=True), Level("K-fold", ordered=False)]


def loss(y, raw):  # per row, in the raw score space
    return 0.5 * (raw - y) ** 2


store = Store("kig-store")  # Parquet files in one directory
summarize(runs, X, y, store, levels=levels, features=names, loss=loss)
build_report(store, "report.html")  # one static HTML file, no server
```

- `X` must take a NumPy index array for its rows, for example a NumPy array. `y` can be an array or a pandas Series.
- The set names `train` and `valid` have a meaning: the report uses them for the metrics. More sets are permitted.
- `summarize` options: `grad`, `weights`, `max_rows` (default 5,000 for each set, `None` for all rows), `seed`, `n_jobs`, `cross_eval`.
- A run whose cache key is in the store is not computed again.
- The cache key holds the model, the row indices, the row sample, the loss function and the package version. It does not hold `X` or `y`: use a new store directory when the data changes and the model does not. The model must be picklable.
- A LightGBM booster with `boosting='rf'` has no additive stage axis. The LightGBM explainer refuses it.
- `kig.aggregate.group_view(store, prefix)` gives the table of one group view.

`uv run python tests/toy.py report.html` builds the report for the toy data.

## Supported models

`Run(path, model, sets)` works with no explainer for these models. Each library is an optional extra. Install from a checkout of this repository, for example `pip install ".[xgboost]"` or `pip install ".[all]"`.

- **LightGBM** (`Booster`, `LGBMRegressor`, binary `LGBMClassifier`, `LGBMRanker`; extra `lightgbm`): one stage for each boosting round, up to the best iteration of a booster with early stopping.
- **XGBoost** (`Booster`, `XGBRegressor`, binary `XGBClassifier`, `XGBRanker`; extra `xgboost`): one stage for each boosting round. XGBoost computes in float32, so its sums agree with its own prediction to approximately 1e-5, not 1e-9. A wrapper with early stopping uses the rounds up to its best iteration. A `gblinear` booster has one stage.
- **CatBoost** (`CatBoostRegressor`, binary `CatBoostClassifier`; extra `catboost`): one stage for each boosting round. Symmetric trees only, which is the default grow policy.
- **Linear models of scikit-learn** (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDRegressor`, binary `LogisticRegression` and the other models with the same base classes; extra `sklearn`): one stage, closed form. The attribution of a feature is `coef * (x - reference)`. The default reference is zero, which is correct for centred features. For other features, give `explainer=LinearExplainer(model, reference=X[idx_train].mean(0))`.
- **Tree models of scikit-learn** (`RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor` and the binary classifiers of the two boosters; extras `sklearn` and `shap`): one stage, from `shap.TreeExplainer`. The forest classifiers have no raw score, so they are not supported.

All attributions are in the raw score space: the margin or the logit, not a probability. Your loss function must take the raw score. Multiclass and multi-output models are refused.

More explainers in `kig.explainers`:

- `ShapExplainer(shap_explainer)`: one stage from an explainer of the `shap` library that has `shap_values` and `expected_value`.
- `CheckpointExplainer([explainer_1, explainer_2, ...])`: a staged explainer for a model with checkpoints, for example a neural network after each epoch. Each item is a one-stage explainer of one checkpoint.
- `OneStageExplainer(explainer)`: the one-stage form of a staged explainer. One report cannot hold staged runs and one-stage runs, because their loss splits are not equal. To compare a boosted model with a linear model or a forest, wrap the staged runs: `Run(..., explainer=OneStageExplainer(LightGBMExplainer(model)))`. A wrapped run has no stage view. A booster with one round, for example after early stopping, is a one-stage run: if a report has such a run, wrap all its staged runs.
- `PrecomputedExplainer(attributions)`: attributions that you computed before.

A scikit-learn `Pipeline` has no explainer: the attributions must be for the columns of the shared `X`.

## Extend

- **Explainer** (`kig.core.Explainer`): give `n_stages`, `stage_label` and `slices(X)`. Register it with `register_explainer(ModelClass, factory)`, or give it to a run with `Run(..., explainer=...)`. `kig.explainers.PrecomputedExplainer` takes attributions that you computed before.
- **Model view** (`kig.core.ModelView`): give `name`, `applies_to`, `collect` and `js`. `js` is one module that exports `render(container, payload, api)`. Register it with `register_view(view)` before `summarize` and `build_report`. The report shows it as one tab.

`api` in `render` has: `features`, `run` (the standard tables of the run), `valid` and `train` (set names), `colors`, `fmt`, `el` and `bars()`. `render` can return a function that the report calls before it removes the view.

## Test

```
uv run pytest
```
