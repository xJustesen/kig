<h1 align="center">👀 kig</h1>

<p align="center">A diagnostic tool for fitted models: which features help or hurt, at which stage, in which split.</p>

<h4 align="center">
  <a href="#-getting-started">Install</a>
  ·
  <a href="#-usage">Usage</a>
  ·
  <a href="#-supported-models">Models</a>
  ·
  <a href="#-explainers">Explainers</a>
  ·
  <a href="#-extend">Extend</a>
</h4>


kig shows which features help or hurt a fitted model, at which stage of the fit, and in which
split. It never fits a model and never makes a split: you give it fitted models and row indices.
Give it development data only.

## ✨ Features

- 👀 **Feature diagnostics** in the loss: which features help or hurt, by stage and by split
- 🌲 **LightGBM, XGBoost, CatBoost and scikit-learn** models work with no explainer to write
- 🪜 **Stage axis** for boosted models: one stage for each boosting round
- 🗂️ **Run levels** such as walk-forward steps and k-folds, with a group view for each level
- 📄 **One static HTML report**: no server
- 💾 **Parquet store** with a cache: a run that is in the store is not computed again
- 🔌 **Extensible**: register your own explainers and your own report tabs

## ⚡️ Requirements

- **Python** 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) to install from a checkout
- The library of your model, as an extra: `lightgbm`, `xgboost`, `catboost`, `sklearn`, `shap`
  **_(optional)_**

## 🚀 Getting Started

Add kig to your project from a checkout of this repository, with the extras for your models:

```bash
uv add /path/to/kig --extra xgboost    # or --extra all
```

To try it first, build the report for the toy data in the checkout:

```bash
uv run python tests/toy.py report.html
```

## 📖 Usage

For one fitted model, one call writes the report:

```python
import kig

kig.diagnose(model, X, y, sets={"train": idx_train, "valid": idx_valid}, out="report.html")
```

`features` and `loss` are optional arguments, with the defaults of `summarize` below.

For more than one model, for example walk-forward steps and k-folds, give the runs and their levels:

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

- `X` must take a NumPy index array for its rows, for example a NumPy array, or be a pandas
  DataFrame. For a DataFrame, `features` defaults to its column names and the model gets the rows
  with their dtypes, for example categorical columns. `y` can be an array or a pandas Series.
- `loss` is optional for the supported models. With no `loss`, kig uses the loss and the analytic
  gradient of the model objective: squared error, binary log loss for the labels 0 and 1, or
  softmax cross-entropy for the integer class indices 0, 1, 2, and so on. All runs must have the
  same loss. For another objective, for example a custom objective, give `loss`.
- The set names `train` and `valid` have a meaning: the report uses them for the metrics. More
  sets are permitted.
- `kig.aggregate.group_view(store, prefix)` gives the table of one group view.

<details><summary>Options of <code>summarize</code></summary>

`grad`, `weights`, `max_rows` (default 5,000 for each set, `None` for all rows), `seed`, `n_jobs`,
`cross_eval`.

</details>

<details><summary>The cache</summary>

- A run whose cache key is in the store is not computed again.
- The cache key holds the model, the row indices, the row sample, the loss function and the
  package version. It does not hold `X` or `y`: use a new store directory when the data changes
  and the model does not.
- A loss from the model objective and a loss that you give do not share the cache, also when they
  are equal.
- The model must be picklable.

</details>

## 🧩 Supported models

`Run(path, model, sets)` works with no explainer for these models:

| Library | Models | Extra | Stages |
|---|---|---|---|
| **LightGBM** | `Booster`, `LGBMRegressor`, `LGBMClassifier`, `LGBMRanker` | `lightgbm` | one for each boosting round |
| **XGBoost** | `Booster`, `XGBRegressor`, `XGBClassifier`, `XGBRanker` | `xgboost` | one for each boosting round |
| **CatBoost** | `CatBoostRegressor`, binary `CatBoostClassifier` | `catboost` | one for each boosting round |
| **scikit-learn, linear** | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDRegressor`, binary `LogisticRegression` and the other models with the same base classes | `sklearn` | one, closed form |
| **scikit-learn, trees** | `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor` and the binary classifiers of the two boosters | `sklearn`, `shap` | one, from `shap.TreeExplainer` |
| **scikit-learn, pipeline** | `Pipeline` with one of the models above as its last step | `sklearn` | those of the last step |

All attributions are in the raw score space: the margin or the logit, not a probability. Your loss
function must take the raw score.

A multiclass model of LightGBM or XGBoost has one raw score for each class. Its loss takes the
scores as an array of shape (rows, classes), and its gradient gives one value for each class. The
gain and the mean |SHAP| of a feature are sums over the classes: the report has no view for each
class. The other multiclass models and all multi-output models are refused.

<details><summary>Notes for each library</summary>

- **LightGBM**: a booster with early stopping uses the rounds up to its best iteration. A booster
  with `boosting='rf'` has no additive stage axis, so the explainer refuses it.
- **XGBoost**: XGBoost computes in float32, so its sums agree with its own prediction to
  approximately 1e-5, not 1e-9. A wrapper with early stopping uses the rounds up to its best
  iteration. A `gblinear` booster has one stage. A booster from `xgboost.train` with a custom
  `obj` reports the default objective: give `loss` for it.
- **CatBoost**: symmetric trees only, which is the default grow policy.
- **scikit-learn, linear**: the attribution of a feature is `coef * (x - reference)`. The default
  reference is zero, which is correct for centred features. For other features, give
  `explainer=LinearExplainer(model, reference=X[idx_train].mean(0))`.
- **scikit-learn, trees**: the forest classifiers have no raw score, so they are not supported.
- **scikit-learn, pipeline**: the attributions are those of the last step, for the columns that
  the other steps give. When these are not the columns of `X`, give their names as `features`:
  `features=list(pipeline[:-1].get_feature_names_out())`. All runs of one report must have the same
  transformed columns.

</details>

## 🧠 Explainers

More explainers in `kig.explainers`:

- **`ShapExplainer(shap_explainer)`**: one stage from an explainer of the `shap` library that has
  `shap_values` and `expected_value`.
- **`CheckpointExplainer([explainer_1, explainer_2, ...])`**: a staged explainer for a model with
  checkpoints, for example a neural network after each epoch. Each item is a one-stage explainer
  of one checkpoint.
- **`OneStageExplainer(explainer)`**: the one-stage form of a staged explainer. A wrapped run has
  no stage view.
- **`PrecomputedExplainer(attributions)`**: attributions that you computed before.

One report cannot hold staged runs and one-stage runs, because their loss splits are not equal. To
compare a boosted model with a linear model or a forest, wrap the staged runs:

```python
Run(..., explainer=OneStageExplainer(LightGBMExplainer(model)))
```

A booster with one round, for example after early stopping, is a one-stage run: if a report has
such a run, wrap all its staged runs.

## 🔌 Extend

- **Explainer** (`kig.core.Explainer`): give `n_stages`, `stage_label` and `slices(X)`. Register
  it with `register_explainer(ModelClass, factory)`, or give it to a run with
  `Run(..., explainer=...)`. To let `summarize` work with no `loss`, also give `loss_and_grad`: the
  pair of the loss function and its gradient for the model objective. If the attributions are not
  for the columns of `X`, also give `features`: the names of their columns.
- **Model view** (`kig.core.ModelView`): give `name`, `applies_to`, `collect` and `js`. `js` is one
  module that exports `render(container, payload, api)`. Register it with `register_view(view)`
  before `summarize` and `build_report`. The report shows it as one tab.

<details><summary>The <code>api</code> of <code>render</code></summary>

`api` has `features`, `run` (the standard tables of the run), `valid` and `train` (set names),
`colors`, `fmt`, `el` and `bars()`. `render` can return a function that the report calls before it
removes the view.

</details>

## 🛠️ Development

```bash
uv run pytest
```
