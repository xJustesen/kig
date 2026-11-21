"""Toy data: synthetic rows in 12 periods, walk-forward steps with purged K-fold inside. There is no test set.

Run `uv run python tests/toy.py out.html` to build the toy report.
"""

from __future__ import annotations

import sys
import tempfile

import lightgbm as lgb
import numpy as np

from kig.core import Level, Run, summarize
from kig.report import build_report
from kig.store import Store

FEATURES = ["linear", "step", "pair_a", "pair_b", "regime", "fading", "noise_1", "noise_2"]
LEVELS = [Level("Walk-forward step", ordered=True), Level("Purged K-fold", ordered=False)]
N_PERIODS, ROWS_PER_PERIOD, WINDOW, K, PURGE, ROUNDS = 12, 500, 4, 3, 25, 150
PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20, verbose=-1)


def mse_loss(y: np.ndarray, raw: np.ndarray) -> np.ndarray:
    return 0.5 * (raw - y) ** 2


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    period = np.repeat(np.arange(1, N_PERIODS + 1), ROWS_PER_PERIOD)
    X = rng.normal(size=(len(period), len(FEATURES)))
    regime_on = (period >= 5) & (period <= 8)
    fade = 1.2 * np.clip(1 - (period - 1) / 8, 0, 1)
    y = (
        1.5 * X[:, 0]
        + 1.2 * (X[:, 1] > 0.5)
        + 1.0 * X[:, 2] * X[:, 3]
        + 1.5 * regime_on * X[:, 4]
        + fade * X[:, 5]
        + rng.normal(scale=1.2, size=len(period))
    )
    return X, y, period


def fit(X, y, idx):
    return lgb.train(PARAMS, lgb.Dataset(X[idx], y[idx]), num_boost_round=ROUNDS)


def make_runs(X, y, period, steps=range(WINDOW + 1, N_PERIODS + 1)) -> list[Run]:
    runs = []
    for p in steps:
        key = f"wf=p{p:02d}"
        window = np.flatnonzero((period >= p - WINDOW) & (period < p))
        forward = np.flatnonzero(period == p)
        runs.append(
            Run(
                (key,),
                fit(X, y, window),
                dict(train=window, valid=forward),
                note=f"train p{p - WINDOW:02d}–p{p - 1:02d}, valid p{p:02d}",
            )
        )
        for k, block in enumerate(np.array_split(window, K), start=1):
            keep = (window < block[0] - PURGE) | (window > block[-1] + PURGE)
            runs.append(
                Run(
                    (key, f"k={k}"),
                    fit(X, y, window[keep]),
                    dict(train=window[keep], valid=block),
                    note=f"valid block {k} of {K} inside the window, {PURGE} rows purged each side",
                )
            )
    return runs


if __name__ == "__main__":
    X, y, period = make_data()
    store = Store(tempfile.mkdtemp())
    summarize(
        make_runs(X, y, period), X, y, store, levels=LEVELS, features=FEATURES, loss=mse_loss, n_jobs=4, cross_eval=True
    )
    print("wrote", build_report(store, sys.argv[1] if len(sys.argv) > 1 else "kig-toy.html", title="kig: toy data"))
