"""Aggregation: pool the sums of the store along the path hierarchy."""

from __future__ import annotations

import pandas as pd

from kig.core import VALID
from kig.store import Store

OWN = "own"
NESTED = "nested"


def node_table(store: Store) -> pd.DataFrame:
    """Return one row (pd.DataFrame) for each node ("/"-joined prefix), source ("own" or "nested"), set and feature."""
    runs = store.read("runs")
    if (runs.n_stages > 1).nunique() > 1:
        raise ValueError("the staged split and the one-stage split must not be in one group view")

    summary = store.read("summary").sort_values("stage", kind="stable")
    totals = (
        summary.groupby(["run_id", "set", "feature"], sort=False)
        .agg(gain_sum=("gain_sum", "sum"), abs_sum=("abs_sum", "last"))
        .reset_index()
        .merge(store.read("sets")[["run_id", "set", "n"]], on=["run_id", "set"])
        .merge(runs[["run_id", "path"]], on="run_id")
    )

    paths = set(runs.path)
    terminal = {p for p in paths if not any(len(q) > len(p) and q[: len(p)] == p for q in paths)}
    own = totals.assign(node=totals.path.map("/".join), source=OWN)
    below = totals[totals.path.isin(terminal)]
    prefixes = below.path.map(lambda p: ["/".join(p[:d]) for d in range(len(p))])
    nested = below.assign(node=prefixes, source=NESTED).explode("node").dropna(subset="node")
    table = (
        pd.concat([own, nested])
        .groupby(["node", "source", "set", "feature"], sort=False)[["gain_sum", "abs_sum", "n"]]
        .sum()
        .reset_index()
    )
    return table.assign(gain=table.gain_sum / table.n, mean_abs=table.abs_sum / table.n)


def group_view(store: Store, prefix: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return the group-view table (pd.DataFrame) at a path prefix; a row with no `child` belongs to the prefix."""
    table = node_table(store)
    depth = len(prefix)
    path = table.node.map(lambda node: tuple(node.split("/")) if node else ())
    view = table[path.map(lambda p: p[:depth] == prefix and len(p) <= depth + 1)]
    child = path[view.index].map(lambda p: p[depth] if len(p) > depth else None)
    return view.assign(child=child).reset_index(drop=True)


def summary_metrics(store: Store) -> dict[str, float]:
    """Return two metrics (dict[str, float]) of the top node on the validation set: the count of features with a negative gain and the lowest gain."""
    view = group_view(store)
    top = view[view.child.isna() & (view.set == VALID)]
    own = top[top.source == OWN]
    gain = (own if len(own) else top).gain
    return {"kig/negative_valid_features": float((gain < 0).sum()), "kig/min_valid_gain": float(gain.min())}
