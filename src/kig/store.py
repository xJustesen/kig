"""Store: the tables on disk that hold the sums, as Parquet files in one directory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SCHEMA = {
    "runs": ["run_id", "path", "depth", "note", "model_kind", "n_stages", "stage_label", "cache_key"],
    "levels": ["depth", "name", "ordered"],
    "sets": ["run_id", "set", "n", "overlap_with_train"],
    "stage_loss": ["run_id", "set", "stage", "loss_sum", "base_gain_sum"],
    "summary": ["run_id", "set", "feature", "stage", "abs_sum", "gain_sum"],
    "view_payload": ["run_id", "view", "payload"],
}


class Store:
    """Directory (str | Path) of one Parquet file for each table in `SCHEMA`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self, table: str) -> pd.DataFrame:
        """Return one table (pd.DataFrame); a table that is not on disk is empty."""
        file = self.path / f"{table}.parquet"
        if not file.exists():
            return pd.DataFrame(columns=SCHEMA[table])
        frame = pd.read_parquet(file)
        if table == "runs":
            frame["path"] = frame["path"].map(tuple)
        return frame

    def write(self, table: str, frame: pd.DataFrame) -> None:
        """Write one table (pd.DataFrame) with the columns of `SCHEMA`."""
        self.path.mkdir(parents=True, exist_ok=True)
        frame = frame[SCHEMA[table]]
        if table == "runs":
            frame = frame.assign(path=frame["path"].map(list))
        frame.to_parquet(self.path / f"{table}.parquet", index=False)

    def replace_runs(self, frames: dict[str, pd.DataFrame]) -> None:
        """Write the run tables (dict[str, pd.DataFrame]); a write that stops in the middle leaves no run for the cache."""
        self.write("runs", frames["runs"].iloc[:0])
        for table, frame in frames.items():
            if table != "runs":
                self.write(table, frame)
        self.write("runs", frames["runs"])
