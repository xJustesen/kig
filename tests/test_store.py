"""Tests of the store: the schema version on disk."""

import json

import pandas as pd
import pytest

from kig.store import SCHEMA, SCHEMA_VERSION, Store


def test_new_store_has_a_meta_file_with_the_schema_version(tmp_path):
    Store(tmp_path).write("levels", pd.DataFrame(columns=SCHEMA["levels"]))
    assert json.loads((tmp_path / "meta.json").read_text())["schema_version"] == SCHEMA_VERSION


def test_read_refuses_a_store_with_a_newer_schema_version(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}))
    with pytest.raises(ValueError, match="schema version"):
        Store(tmp_path).read("runs")
