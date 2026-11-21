"""kig: a diagnostic tool for fitted models."""
from kig.explainers import register_builtin_explainers
from kig.views import register_builtin_views

register_builtin_explainers()
register_builtin_views()
