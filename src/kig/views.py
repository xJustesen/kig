"""Model views: plug-in views of one run. The stage view and the summary view are part of the package."""
from __future__ import annotations

from importlib.resources import files

from kig.core import register_view


class StageView:
    """Stage time-lapse, loss chart and bars of a run with more than one stage; it reads the standard tables only."""
    name = "Stages"
    js = files("kig").joinpath("static/stage_view.js").read_text(encoding="utf-8")

    def applies_to(self, run_info: dict) -> bool:
        """Return True for a run (dict) that has a stage axis."""
        return run_info["n_stages"] > 1

    def collect(self, run, explainer, data: dict) -> dict:
        """Return an empty payload (dict)."""
        return {}


class SummaryView:
    """Bars of the full model with no stage control; it applies to each run and reads the standard tables only."""
    name = "Summary"
    js = files("kig").joinpath("static/summary_view.js").read_text(encoding="utf-8")

    def applies_to(self, run_info: dict) -> bool:
        """Return True for each run (dict)."""
        return True

    def collect(self, run, explainer, data: dict) -> dict:
        """Return an empty payload (dict)."""
        return {}


def register_builtin_views() -> None:
    """Register the model views that are part of the package."""
    register_view(StageView())
    register_view(SummaryView())
