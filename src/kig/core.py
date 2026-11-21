"""Core: runs, the registries and the summary pass. The core knows no model type."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib.metadata import version
from typing import Any, Callable, Iterator, Protocol

import numpy as np
import pandas as pd

from kig.store import Store

TRAIN = "train"
VALID = "valid"


@dataclass(frozen=True)
class Level:
    """One depth of the split hierarchy: its name (str) and if its keys have an order (bool)."""

    name: str
    ordered: bool


@dataclass
class Run:
    """One fitted model, its path (tuple[str, ...]) and its named row sets (dict[str, np.ndarray])."""

    path: tuple[str, ...]
    model: Any
    sets: dict[str, np.ndarray]
    note: str = ""
    explainer: Explainer | None = None


class Explainer(Protocol):
    """Adapter that gives the slices of one model; set `by_index = True` to receive row indices in place of rows."""

    n_stages: int
    stage_label: str

    def slices(self, X) -> Iterator[np.ndarray]:
        """Yield one array (rows, features + 1) for each stage; the last column is the base term."""


class ModelView(Protocol):
    """Plug-in view of one run: a JSON-safe payload from `collect` and a JavaScript module in `js` (str)."""

    name: str
    js: str

    def applies_to(self, run_info: dict) -> bool:
        """Return True if the view applies to the run with this `runs` table row (dict)."""

    def collect(self, run: Run, explainer: Explainer, data: dict) -> dict:
        """Return the JSON-safe payload of the view for one run."""


_EXPLAINERS: dict[type | str, Callable[[Any], Explainer]] = {}
_VIEWS: dict[str, ModelView] = {}


def register_explainer(model_type: type | str, factory: Callable[[Any], Explainer]) -> None:
    """Register an explainer factory for a model class (type) or its qualified name (str)."""
    _EXPLAINERS[model_type] = factory


def register_view(view: ModelView) -> None:
    """Register a model view (ModelView) under its name."""
    _VIEWS[view.name] = view


def model_views() -> list[ModelView]:
    """Return the registered model views (list[ModelView]) in registration order."""
    return list(_VIEWS.values())


def explainer_for(run: Run) -> Explainer:
    """Return the explainer (Explainer) of a run: its own, or one from the registry by model type."""
    if run.explainer is not None:
        return run.explainer
    for cls in type(run.model).__mro__:
        for key in (cls, f"{cls.__module__}.{cls.__qualname__}"):
            if key in _EXPLAINERS:
                return _EXPLAINERS[key](run.model)
    raise TypeError(f"no explainer registered for {type(run.model).__qualname__}")


def run_id(path: tuple[str, ...]) -> str:
    """Return the store id (str) of a path."""
    return "/".join(path)


def _identity(fn: Callable | None) -> str:
    code = getattr(fn, "__code__", None)
    with np.printoptions(threshold=sys.maxsize):
        return repr(
            (
                getattr(fn, "__module__", ""),
                getattr(fn, "__qualname__", repr(fn)),
                code and (code.co_code, code.co_consts),
                [cell.cell_contents for cell in getattr(fn, "__closure__", None) or ()],
            )
        )


def _cache_key(run: Run, loss, grad, weights, max_rows, seed, features) -> str:
    h = hashlib.sha256(pickle.dumps((run.model, run.explainer)))
    for name in sorted(run.sets):
        h.update(name.encode())
        h.update(np.asarray(run.sets[name], dtype=np.int64).tobytes())
    if weights is not None:
        h.update(np.asarray(weights, dtype=np.float64).tobytes())
    h.update(
        repr(
            (
                max_rows,
                seed,
                _identity(loss),
                _identity(grad),
                list(features),
                run.note,
                [v.name for v in model_views()],
                version("kig"),
            )
        ).encode()
    )
    return h.hexdigest()


def _with_sibling_sets(runs: list[Run]) -> list[Run]:
    valid = {r.path: r.sets[VALID] for r in runs if VALID in r.sets}
    out = []
    for r in runs:
        extra = {
            f"{VALID}@{p[-1]}": idx
            for p, idx in valid.items()
            if p and len(p) == len(r.path) and p[:-1] == r.path[:-1] and p != r.path
        }
        out.append(replace(r, sets={**r.sets, **extra}))
    return out


def _inferred_loss(runs: list[Run]) -> tuple[Callable, Callable]:
    """Return the loss and its gradient (tuple[Callable, Callable]) of the model objective that all runs share."""
    found = {run_id(r.path): getattr(explainer_for(r), "loss_and_grad", None) for r in runs}
    unknown = [rid for rid, pair in found.items() if pair is None]
    if unknown:
        raise ValueError(f"no loss is known for the model objective of the runs {unknown}: give `loss`")
    if len(set(found.values())) != 1:
        raise ValueError("the runs must have one loss for all their model objectives: give `loss`")
    return next(iter(found.values()))


def _numeric_grad(loss) -> Callable:
    h = 1e-6
    return lambda y, raw: (loss(y, raw + h) - loss(y, raw - h)) / (2 * h)


def _start_at_base(s: np.ndarray, n_feat: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the start score and the slice of a one-stage model: the base term starts the split and gets no gain."""
    return s[:, n_feat], np.column_stack([s[:, :n_feat], np.zeros(len(s))])


def _reduce(explainer: Explainer, rows, y, w, loss, grad, n_feat: int) -> dict[str, np.ndarray]:
    """Read the slices in stage order; keep sums only, so memory stays at rows x features."""
    n_stages = explainer.n_stages
    abs_sum = np.empty((n_stages, n_feat))
    gain_sum = np.empty((n_stages, n_feat))
    loss_sum = np.empty(n_stages)
    base_gain_sum = np.empty(n_stages)
    a = np.zeros(len(y))
    cum = np.zeros((len(y), n_feat))
    slices = (np.asarray(s, dtype=float) for s in explainer.slices(rows))
    if n_stages == 1:
        a, only = _start_at_base(next(slices), n_feat)
        slices = [only]
    for t, s in zip(range(n_stages), slices, strict=True):
        phi, base = s[:, :n_feat], s[:, n_feat]
        step = phi.sum(1) + base
        b = a + step
        flat = b == a
        mean_grad = (loss(y, b) - loss(y, a)) / np.where(flat, 1.0, step)
        if flat.any():
            mean_grad = np.where(flat, grad(y, a), mean_grad)
        wg = w * mean_grad
        gain_sum[t] = -(phi * wg[:, None]).sum(0)
        base_gain_sum[t] = -(base * wg).sum()
        cum += phi
        abs_sum[t] = (np.abs(cum) * w[:, None]).sum(0)
        loss_sum[t] = (w * loss(y, b)).sum()
        a = b
    return dict(abs_sum=abs_sum, gain_sum=gain_sum, loss_sum=loss_sum, base_gain_sum=base_gain_sum)


def _summarize_run(run: Run, key: str, X, y, weights, features, loss, grad, max_rows, seed) -> dict[str, pd.DataFrame]:
    rid = run_id(run.path)
    explainer = explainer_for(run)
    own = getattr(explainer, "features", None)
    if own is not None and list(own) != list(features):
        raise ValueError(f"the attributions of run {rid!r} are for the columns {list(own)}: give them as `features`")
    n_stages, n_feat = explainer.n_stages, len(features)
    stage = np.arange(n_stages)
    info = dict(
        run_id=rid,
        path=list(run.path),
        depth=len(run.path),
        note=run.note,
        model_kind=type(run.model).__name__,
        n_stages=n_stages,
        stage_label=explainer.stage_label,
        cache_key=key,
    )
    take = X.iloc if isinstance(X, pd.DataFrame) else X
    sets, stage_loss, summary, sampled = [], [], [], {}
    for name, full in run.sets.items():
        idx = np.asarray(full)
        if max_rows is not None and len(idx) > max_rows:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))
        sampled[name] = idx
        w = np.ones(len(idx)) if weights is None else np.asarray(weights, dtype=float)[idx]
        rows = idx if getattr(explainer, "by_index", False) else take[idx]
        sums = _reduce(explainer, rows, y[idx], w, loss, grad, n_feat)
        overlap = len(np.intersect1d(run.sets[TRAIN], full)) if TRAIN in run.sets else 0
        sets.append(dict(run_id=rid, set=name, n=float(w.sum()), overlap_with_train=overlap))
        stage_loss.append(
            pd.DataFrame(
                dict(run_id=rid, set=name, stage=stage, loss_sum=sums["loss_sum"], base_gain_sum=sums["base_gain_sum"])
            )
        )
        summary.append(
            pd.DataFrame(
                dict(
                    run_id=rid,
                    set=name,
                    feature=np.tile(features, n_stages),
                    stage=np.repeat(stage, n_feat),
                    abs_sum=sums["abs_sum"].ravel(),
                    gain_sum=sums["gain_sum"].ravel(),
                )
            )
        )
    data = dict(X=X, y=y, weights=weights, sets=sampled, features=list(features))
    payloads = [
        dict(run_id=rid, view=v.name, payload=json.dumps(v.collect(run, explainer, data), allow_nan=False))
        for v in model_views()
        if v.applies_to(info)
    ]
    return dict(
        runs=pd.DataFrame([info]),
        sets=pd.DataFrame(sets),
        stage_loss=pd.concat(stage_loss),
        summary=pd.concat(summary),
        view_payload=pd.DataFrame(payloads, columns=["run_id", "view", "payload"]),
    )


def summarize(
    runs: list[Run],
    X,
    y,
    store: Store,
    *,
    levels: list[Level],
    features: list[str] | None = None,
    loss: Callable | None = None,
    grad: Callable | None = None,
    weights=None,
    max_rows: int | None = 5000,
    seed: int = 0,
    n_jobs: int = 1,
    cross_eval: bool = False,
) -> list[str]:
    """Write the sums of all runs to the store and return the ids (list[str]) of the runs that were not in the cache."""
    if features is None and not isinstance(X, pd.DataFrame):
        raise TypeError("give `features` when `X` is not a pandas DataFrame")
    paths = [r.path for r in runs]
    if len(set(paths)) != len(paths):
        raise ValueError("each run must have a different path")
    if any(not key or "/" in key for p in paths for key in p):
        raise ValueError("a path key must not be empty and must not contain '/'")
    if max(map(len, paths), default=0) > len(levels):
        raise ValueError("declare one level for each depth of the paths")

    y = np.asarray(y)
    if features is None:
        features = [str(name) for name in X.columns]
    if loss is None:
        loss, grad = _inferred_loss(runs)
    if cross_eval:
        runs = _with_sibling_sets(runs)
    keys = {run_id(r.path): _cache_key(r, loss, grad, weights, max_rows, seed, features) for r in runs}
    grad = grad or _numeric_grad(loss)
    old = store.read("runs")
    cached = {rid for rid, key in zip(old.run_id, old.cache_key) if keys.get(rid) == key}
    todo = [r for r in runs if run_id(r.path) not in cached]

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        new = list(
            pool.map(
                lambda r: _summarize_run(r, keys[run_id(r.path)], X, y, weights, features, loss, grad, max_rows, seed),
                todo,
            )
        )
    order = {rid: i for i, rid in enumerate(keys)}
    frames = {}
    for table in ("runs", "sets", "stage_loss", "summary", "view_payload"):
        kept = store.read(table)
        frame = pd.concat([kept[kept.run_id.isin(cached)], *(n[table] for n in new)], ignore_index=True)
        frames[table] = frame.sort_values("run_id", key=lambda s: s.map(order), kind="stable", ignore_index=True)
    store.replace_runs(frames)
    store.write(
        "levels",
        pd.DataFrame(
            dict(depth=range(len(levels)), name=[lv.name for lv in levels], ordered=[lv.ordered for lv in levels])
        ),
    )
    return [run_id(r.path) for r in todo]
