# Store schema

A kig store is one directory with one Parquet file for each table and one `meta.json` file.
`kig.store.Store` reads and writes it. The store is a public interface: other tools can read the
Parquet files directly. This page is for schema version 1.

## `meta.json`

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | integer | The schema version of the store: `kig.store.SCHEMA_VERSION` of the kig that wrote it. |
| `kig_version` | string | The version of the kig package that wrote the store last. |

A store with no `meta.json` has schema version 1. `Store.read` raises a `ValueError` for a store
with a schema version that is newer than its own.

## When the schema version increases

The schema version increases by one when a column is removed, renamed or gets a new definition.
An added column or an added table does not increase it, so a reader must ignore the columns and
tables that it does not know.

## Tables

`<table>.parquet` holds the table. A table that is not on disk is empty.

The sums are over the rows of one set of one run. If `summarize` gets `weights`, each row counts
with its weight. A mean for each row is a sum divided by `sets.n`.

### `runs`

One row for each run.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | The path keys of the run, joined with `/`. It is the key of the run in the other tables. |
| `path` | list of string | The path keys of the run, one for each level from the top. |
| `depth` | int64 | The number of path keys. |
| `note` | string | The free text of `Run.note`. |
| `model_kind` | string | The class name of the model. |
| `n_stages` | int64 | The number of stages of the explainer. A one-stage run has 1. |
| `stage_label` | string | The name of one stage, for example `Boosting round`. |
| `cache_key` | string | A SHA-256 hash of the inputs of the run. A run with the same key is not computed again. |

### `levels`

One row for each level of the path hierarchy.

| Column | Type | Meaning |
|---|---|---|
| `depth` | int64 | The position of the level, from 0 at the top. Level `depth` names the path key at index `depth`. |
| `name` | string | The name of the level, for example `Walk-forward step`. |
| `ordered` | bool | True if the keys of the level have an order, for example time steps. |

### `sets`

One row for each run and row set.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | The run. |
| `set` | string | The name of the row set. `train` and `valid` have a meaning in the report. `valid@<key>` is the validation set of a sibling run, from `cross_eval`. |
| `n` | double | The number of rows that the sums hold, after the row sample. With `weights`, the sum of the weights. |
| `overlap_with_train` | int64 | The number of rows of the full set that are also in the `train` set of the run. |

### `stage_loss`

One row for each run, row set and stage.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | The run. |
| `set` | string | The row set. |
| `stage` | int64 | The stage, from 0. |
| `loss_sum` | double | The sum of the loss after this stage. |
| `base_gain_sum` | double | The loss reduction of this stage that the base term gets: the part of the raw score that belongs to no feature. |

### `summary`

One row for each run, row set, feature and stage.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | The run. |
| `set` | string | The row set. |
| `feature` | string | The feature name. |
| `stage` | int64 | The stage, from 0. |
| `abs_sum` | double | The sum of the absolute attribution of the feature, cumulative up to and including this stage. |
| `gain_sum` | double | The loss reduction that the feature gets in this stage only. A negative value is a loss increase. |

For each run, row set and stage, the sum of `gain_sum` over the features plus `base_gain_sum` is
the loss before the stage minus the loss after it. To get the gain of a feature for the full
model, add `gain_sum` over the stages. `abs_sum` is cumulative: use the last stage.

### `view_payload`

One row for each run and each model view that applies to it.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | The run. |
| `view` | string | The name of the model view, which is the name of its tab in the report. |
| `payload` | string | The data of the view for this run, as JSON text. |
