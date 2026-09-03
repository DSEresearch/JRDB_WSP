from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class RunningMoments:
    """Streaming per-feature mean and variance."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.shape[0] == 0:
            return

        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        assert self.m2 is not None
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total

    def as_dict(self, minimum_std: float = 1e-8) -> dict[str, Any]:
        if self.mean is None or self.m2 is None or self.count == 0:
            raise RuntimeError("Cannot summarize empty running statistics.")
        variance = self.m2 / max(self.count, 1)
        std = np.maximum(np.sqrt(variance), minimum_std)
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
        }


class NPZShardWriter:
    """Accumulate fixed-schema rows and atomically emit compressed NPZ shards."""

    def __init__(self, output_dir: Path, shard_size: int):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, list[np.ndarray | float | int | bool]] = {}
        self._schema: tuple[str, ...] | None = None
        self.shard_index = 0
        self.total_rows = 0
        self.shards: list[dict[str, Any]] = []

    def append(self, row: dict[str, np.ndarray | float | int | bool]) -> None:
        schema = tuple(sorted(row))
        if self._schema is None:
            self._schema = schema
            self._rows = {name: [] for name in schema}
        elif schema != self._schema:
            raise ValueError(f"Row schema changed from {self._schema} to {schema}")

        for name in self._schema:
            self._rows[name].append(row[name])
        if len(next(iter(self._rows.values()))) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows or not next(iter(self._rows.values())):
            return
        arrays = {name: np.asarray(values) for name, values in self._rows.items()}
        row_count = len(next(iter(arrays.values())))
        filename = f"part-{self.shard_index:05d}.npz"
        destination = self.output_dir / filename
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)

        self.shards.append({"file": filename, "rows": row_count})
        self.shard_index += 1
        self.total_rows += row_count
        self._rows = {name: [] for name in self._schema or ()}

    def close(self) -> dict[str, Any]:
        self.flush()
        return {
            "rows": self.total_rows,
            "shards": self.shards,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)

