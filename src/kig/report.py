"""Report: the data payload of the report and the builder of the static HTML file."""
from __future__ import annotations

import html
import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np

from kig.aggregate import node_table
from kig.core import TRAIN, model_views
from kig.store import Store

MAX_BAR_STAGES = 40


def _sig(values) -> list:
    return np.vectorize(lambda v: float(f"{v:.6g}") if np.isfinite(v) else None, otypes=[object])(values).tolist()


def _nodes(store: Store, paths: list[tuple[str, ...]], features: list[str]) -> dict:
    nodes = {}
    for path in paths:
        for depth in range(len(path) + 1):
            node = nodes.setdefault("/".join(path[:depth]), dict(path=list(path[:depth]), children=[], own=None, nested=None))
            if depth < len(path) and path[depth] not in node["children"]:
                node["children"].append(path[depth])
    for node in nodes.values():
        node["children"].sort()
    for (node, source, name), g in node_table(store).groupby(["node", "source", "set"], sort=False):
        g = g.set_index("feature").loc[features]
        nodes[node][source] = {**(nodes[node][source] or {}),
                               name: dict(n=float(g.n.iloc[0]), gain=_sig(g.gain), abs=_sig(g.mean_abs))}
    return nodes


def _run_tables(summary, stage_loss, counts: dict[str, float], features: list[str]) -> dict:
    loss = {name: (g.sort_values("stage").loss_sum / counts[name]).to_numpy() for name, g in stage_loss.groupby("set", sort=False)}
    n_stages = len(next(iter(loss.values())))
    stages = set(np.round(np.geomspace(1, n_stages, min(n_stages, MAX_BAR_STAGES))).astype(int) - 1)
    stages |= {int(np.argmin(v)) for name, v in loss.items() if name != TRAIN}
    stages = sorted(int(s) for s in stages)
    gain, mean_abs = {}, {}
    for name, g in summary.groupby("set", sort=False):
        wide = g.pivot(index="feature", columns="stage", values=["gain_sum", "abs_sum"]).loc[features]
        gain[name] = _sig(wide["gain_sum"].to_numpy().cumsum(1)[:, stages] / counts[name])
        mean_abs[name] = _sig(wide["abs_sum"].to_numpy()[:, stages] / counts[name])
    return dict(stages=stages, loss={name: _sig(v) for name, v in loss.items()}, gain=gain, abs=mean_abs)


def report_data(store: Store) -> dict:
    """Return the data payload (dict) of the report: the levels, the nodes of the hierarchy and the tables of each run."""
    runs, sets = store.read("runs"), store.read("sets")
    summary, stage_loss, payloads = store.read("summary"), store.read("stage_loss"), store.read("view_payload")
    features = list(summary.feature.unique())
    registered = [v.name for v in model_views()]
    out = {}
    for run in runs.itertuples():
        mine = sets[sets.run_id == run.run_id]
        counts = dict(zip(mine.set, mine.n))
        stored = payloads[payloads.run_id == run.run_id].set_index("view").payload
        out[run.run_id] = dict(
            path=list(run.path), note=run.note, model_kind=run.model_kind, n_stages=int(run.n_stages),
            stage_label=run.stage_label,
            sets={s.set: dict(n=float(s.n), overlap=int(s.overlap_with_train)) for s in mine.itertuples()},
            **_run_tables(summary[summary.run_id == run.run_id], stage_loss[stage_loss.run_id == run.run_id], counts, features),
            views=[dict(name=name, payload=json.loads(stored[name])) for name in registered if name in stored.index])
    levels = store.read("levels").sort_values("depth")
    return dict(version=version("kig"), features=features,
                levels=[dict(name=lv.name, ordered=bool(lv.ordered)) for lv in levels.itertuples()],
                nodes=_nodes(store, list(runs.path), features), runs=out)


def _embed(value) -> str:
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")


def build_report(store: Store, out: str | Path, title: str = "kig report") -> Path:
    """Write the report as one static HTML file with all data, styles and scripts inside; return its path (Path)."""
    static = files("kig").joinpath("static")
    title = html.escape(title)
    # Each model view is one inline module. An inline module cannot be imported, so one added line hands
    # its `render` to the report shell.
    views = "\n".join(
        f'<script type="module">\n{v.js.replace("</script", "<\\/script")}\n'
        f'(globalThis.kigViews ??= {{}})[{_embed(v.name)}] = {{ render }};\n</script>' for v in model_views())
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<style>
{static.joinpath("report.css").read_text(encoding="utf-8")}
</style>
</head>
<body>
<div class="page">
  <p class="tool">{title}</p>
  <h1 class="crumbs" id="crumbs"></h1>
  <p class="lede" id="lede"></p>
  <div id="tabs"></div>
  <div class="controls" id="controls"></div>
  <main id="view"></main>
  <p class="foot">Loss reduction is the part of the loss change that belongs to one feature, per row. A positive value lowers the loss and a negative value raises it. Mean |SHAP| shows how much a feature moves the prediction, and it cannot tell a useful feature from a harmful feature. All numbers come from development data.</p>
</div>
<script type="application/json" id="kig-data">{_embed(report_data(store))}</script>
{views}
<script type="module">
{static.joinpath("app.js").read_text(encoding="utf-8")}
</script>
</body>
</html>
"""
    out = Path(out)
    out.write_text(page, encoding="utf-8")
    return out
