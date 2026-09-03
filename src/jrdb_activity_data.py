from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FORMAT_VERSION = "jrdb_activity_relational_v1"
REQUIRED_COLUMNS = (
    "sequence_id",
    "frame_id",
    "track_id",
    "timestamp",
    "x",
    "y",
    "z",
    "action_id",
)
GEOMETRY_COLUMNS = ("length", "width", "height")
OPTIONAL_COLUMNS = (*GEOMETRY_COLUMNS, "heading", "occlusion")
OPTIONAL_ALIASES = {
    "length": ("box_length",),
    "width": ("box_width",),
    "height": ("box_height",),
    "heading": ("yaw",),
    "occlusion": ("occluded",),
}

FOCAL_FEATURE_NAMES = (
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "length",
    "width",
    "height",
    "heading_sin",
    "heading_cos",
    "occlusion",
    "presence",
)
NEIGHBOR_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity_z",
    "distance",
    "length",
    "width",
    "height",
    "relative_heading_sin",
    "relative_heading_cos",
    "occlusion",
    "presence",
)


@dataclass(frozen=True)
class ActivityPreparationConfig:
    """Configuration for canonical 3D activity window preparation."""

    k_neighbors: int = 4
    window_length: int = 8
    window_stride: int = 4
    seed: int = 2027
    train_seen_steps: tuple[int, ...] = (1, 2, 3)
    unseen_gap_steps: tuple[int, ...] = (4, 5, 6)
    train_gap_min_seconds: float = 1.0
    train_gap_max_seconds: float = 3.0
    unseen_gap_min_seconds: float = 4.0
    unseen_gap_max_seconds: float = 6.0
    burst_min: int = 2
    burst_max: int = 4
    include_partial_windows: bool = True
    heading_degrees: bool = False

    def validate(self) -> None:
        if self.k_neighbors < 0:
            raise ValueError("k_neighbors must be non-negative")
        if self.window_length <= 0 or self.window_stride <= 0:
            raise ValueError("window_length and window_stride must be positive")
        for name, values in (
            ("train_seen_steps", self.train_seen_steps),
            ("unseen_gap_steps", self.unseen_gap_steps),
        ):
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        overlap = set(self.train_seen_steps).intersection(self.unseen_gap_steps)
        if overlap:
            raise ValueError(
                "unseen_gap_steps must be disjoint from train_seen_steps; "
                f"overlap={sorted(overlap)}"
            )
        if self.burst_min <= 0 or self.burst_max < self.burst_min:
            raise ValueError("burst bounds must satisfy 0 < burst_min <= burst_max")
        if not 0 < self.train_gap_min_seconds <= self.train_gap_max_seconds:
            raise ValueError("training gap bounds must be positive and ordered")
        if not 0 < self.unseen_gap_min_seconds <= self.unseen_gap_max_seconds:
            raise ValueError("unseen gap bounds must be positive and ordered")
        if self.train_gap_max_seconds >= self.unseen_gap_min_seconds:
            raise ValueError("training and unseen physical-time ranges must be disjoint")


@dataclass(frozen=True)
class _Observation:
    timestamp: float
    frame_id: int
    x: np.ndarray
    y: np.ndarray


def _canonical_label(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("action_id must not be missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    label = str(value).strip()
    if not label:
        raise ValueError("action_id must not be blank")
    return label


def _label_sort_key(label: str) -> tuple[int, float | str, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, label, label)


def _resolve_optional_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    frame = frame.copy()
    available: dict[str, bool] = {}
    for canonical in OPTIONAL_COLUMNS:
        source = canonical if canonical in frame.columns else None
        if source is None:
            source = next(
                (alias for alias in OPTIONAL_ALIASES.get(canonical, ()) if alias in frame.columns),
                None,
            )
        available[canonical] = source is not None
        frame[canonical] = 0.0 if source is None else frame[source]
    return frame, available


def load_canonical_activity_csv(
    path: str | Path,
    *,
    heading_degrees: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, bool]]:
    """Load and collapse the canonical long-form activity table.

    Repeated rows for the same sequence/frame/track are interpreted as
    multilabel annotations. Their state fields must agree.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            "Canonical activity CSV is missing required columns: " + ", ".join(missing)
        )
    frame, optional_available = _resolve_optional_columns(frame)

    for name in ("sequence_id", "track_id"):
        if frame[name].isna().any():
            raise ValueError(f"{name} must not be missing")
        frame[name] = frame[name].astype(str).str.strip()
        if frame[name].eq("").any():
            raise ValueError(f"{name} must not be blank")

    frame_number = pd.to_numeric(frame["frame_id"], errors="raise")
    if not np.isfinite(frame_number.to_numpy(dtype=np.float64)).all():
        raise ValueError("frame_id must be finite")
    if not np.equal(frame_number, np.floor(frame_number)).all():
        raise ValueError("frame_id must contain integer-valued identifiers")
    frame["frame_id"] = frame_number.astype(np.int64)

    numeric_columns = ["timestamp", "x", "y", "z", *OPTIONAL_COLUMNS]
    for name in numeric_columns:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    required_numeric = frame[["timestamp", "x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(required_numeric).all():
        raise ValueError("timestamp and x/y/z must be finite")
    optional_values = frame[list(OPTIONAL_COLUMNS)].to_numpy(dtype=np.float64)
    if np.isinf(optional_values).any():
        raise ValueError("optional geometry, heading, and occlusion values must not be infinite")
    frame[list(OPTIONAL_COLUMNS)] = frame[list(OPTIONAL_COLUMNS)].fillna(0.0)
    if (frame[list(GEOMETRY_COLUMNS)] < 0).any().any():
        raise ValueError("length, width, and height must be non-negative")
    if heading_degrees:
        frame["heading"] = np.deg2rad(frame["heading"].to_numpy(dtype=np.float64))
    frame["action_id"] = frame["action_id"].map(_canonical_label)

    keys = ["sequence_id", "frame_id", "track_id"]
    state_columns = ["timestamp", "x", "y", "z", *OPTIONAL_COLUMNS]
    variation = frame.groupby(keys, sort=False)[state_columns].nunique(dropna=False)
    inconsistent = variation.gt(1).any(axis=1)
    if inconsistent.any():
        example = inconsistent[inconsistent].index[0]
        raise ValueError(
            "Repeated multilabel rows must have identical state fields; "
            f"first inconsistent key={example}"
        )

    labels = sorted(frame["action_id"].unique().tolist(), key=_label_sort_key)
    label_sets = (
        frame.groupby(keys, sort=False)["action_id"]
        .agg(lambda values: tuple(sorted(set(values), key=_label_sort_key)))
        .rename("action_ids")
        .reset_index()
    )
    entities = frame.drop_duplicates(keys, keep="first")[keys + state_columns]
    entities = entities.merge(label_sets, on=keys, how="inner", validate="one_to_one")
    entities = entities.sort_values(
        ["sequence_id", "track_id", "timestamp", "frame_id"], kind="stable"
    ).reset_index(drop=True)

    grouped = entities.groupby(["sequence_id", "track_id"], sort=False)
    elapsed = grouped["timestamp"].diff()
    first = grouped.cumcount().eq(0)
    invalid_elapsed = (~first) & (elapsed <= 0)
    if invalid_elapsed.any():
        row = entities.loc[invalid_elapsed].iloc[0]
        raise ValueError(
            "Timestamps must be strictly increasing within each track; "
            f"sequence={row.sequence_id}, track={row.track_id}, frame={row.frame_id}"
        )
    for axis in ("x", "y", "z"):
        displacement = grouped[axis].diff()
        velocity = displacement.div(elapsed).where(~first, 0.0)
        entities[f"velocity_{axis}"] = velocity.fillna(0.0).astype(np.float32)
    return entities, labels, optional_available


def relational_feature_names(k_neighbors: int) -> list[str]:
    if k_neighbors < 0:
        raise ValueError("k_neighbors must be non-negative")
    names = [f"focal.{name}" for name in FOCAL_FEATURE_NAMES]
    for index in range(k_neighbors):
        names.extend(f"neighbor_{index}.{name}" for name in NEIGHBOR_FEATURE_NAMES)
    return names


def build_relational_observations(
    entities: pd.DataFrame,
    labels: Sequence[str],
    *,
    k_neighbors: int,
    optional_available: Mapping[str, bool],
) -> dict[tuple[str, str], list[_Observation]]:
    """Build target-centered focal-plus-neighbor observations for every track."""

    if k_neighbors < 0:
        raise ValueError("k_neighbors must be non-negative")
    class_index = {label: index for index, label in enumerate(labels)}
    tracks: dict[tuple[str, str], list[_Observation]] = {}
    position_columns = ["x", "y", "z"]
    velocity_columns = ["velocity_x", "velocity_y", "velocity_z"]

    for (sequence_id, _), group in entities.groupby(
        ["sequence_id", "frame_id"], sort=False
    ):
        group = group.sort_values("track_id", kind="stable").reset_index(drop=True)
        positions = group[position_columns].to_numpy(dtype=np.float32)
        velocities = group[velocity_columns].to_numpy(dtype=np.float32)
        geometry = group[list(GEOMETRY_COLUMNS)].to_numpy(dtype=np.float32)
        headings = group["heading"].to_numpy(dtype=np.float32)
        occlusions = group["occlusion"].to_numpy(dtype=np.float32)
        track_ids = group["track_id"].astype(str).to_numpy()

        for focal_index, focal_row in group.iterrows():
            heading_present = bool(optional_available.get("heading", False))
            focal_heading = headings[focal_index]
            focal_heading_pair = (
                (float(np.sin(focal_heading)), float(np.cos(focal_heading)))
                if heading_present
                else (0.0, 0.0)
            )
            focal = np.asarray(
                [
                    *velocities[focal_index],
                    *geometry[focal_index],
                    *focal_heading_pair,
                    occlusions[focal_index],
                    1.0,
                ],
                dtype=np.float32,
            )

            offsets = positions - positions[focal_index]
            distances = np.linalg.norm(offsets, axis=1)
            candidates = np.flatnonzero(np.arange(len(group)) != focal_index)
            if candidates.size:
                stable_order = np.lexsort((track_ids[candidates], distances[candidates]))
                candidates = candidates[stable_order]

            neighbor_blocks: list[np.ndarray] = []
            for neighbor_index in candidates[:k_neighbors]:
                relative_heading = headings[neighbor_index] - focal_heading
                heading_pair = (
                    (float(np.sin(relative_heading)), float(np.cos(relative_heading)))
                    if heading_present
                    else (0.0, 0.0)
                )
                neighbor_blocks.append(
                    np.asarray(
                        [
                            *offsets[neighbor_index],
                            *(velocities[neighbor_index] - velocities[focal_index]),
                            distances[neighbor_index],
                            *geometry[neighbor_index],
                            *heading_pair,
                            occlusions[neighbor_index],
                            1.0,
                        ],
                        dtype=np.float32,
                    )
                )
            while len(neighbor_blocks) < k_neighbors:
                neighbor_blocks.append(np.zeros(len(NEIGHBOR_FEATURE_NAMES), dtype=np.float32))

            feature = np.concatenate([focal, *neighbor_blocks]).astype(np.float32, copy=False)
            target = np.zeros(len(labels), dtype=np.float32)
            for action_id in focal_row.action_ids:
                target[class_index[action_id]] = 1.0
            key = (str(sequence_id), str(focal_row.track_id))
            tracks.setdefault(key, []).append(
                _Observation(
                    timestamp=float(focal_row.timestamp),
                    frame_id=int(focal_row.frame_id),
                    x=feature,
                    y=target,
                )
            )

    for key, observations in tracks.items():
        observations.sort(key=lambda item: (item.timestamp, item.frame_id))
        times = np.asarray([item.timestamp for item in observations], dtype=np.float64)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"Non-increasing timestamps after relational assembly for {key}")
    return tracks


def _load_split_map(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split JSON must contain an object")
    allowed = {"train", "val", "test"}
    if set(payload).issubset(allowed) and all(isinstance(value, list) for value in payload.values()):
        result: dict[str, str] = {}
        for split, sequence_ids in payload.items():
            for sequence_id in sequence_ids:
                key = str(sequence_id)
                if key in result:
                    raise ValueError(f"Sequence {key!r} appears in multiple split lists")
                result[key] = split
        return result
    result = {str(sequence_id): str(split) for sequence_id, split in payload.items()}
    invalid = sorted(set(result.values()) - allowed)
    if invalid:
        raise ValueError(f"Invalid split names in JSON: {invalid}")
    return result


def assign_sequence_splits(
    sequence_ids: Iterable[str],
    *,
    seed: int,
    split_json: str | Path | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> dict[str, str]:
    """Assign whole sequences to train/validation/test without leakage."""

    sequences = sorted({str(value) for value in sequence_ids})
    if len(sequences) < 3:
        raise ValueError("At least three sequences are required for disjoint train/val/test splits")
    if split_json is not None:
        assignment = _load_split_map(split_json)
        missing = sorted(set(sequences) - set(assignment))
        unknown = sorted(set(assignment) - set(sequences))
        if missing or unknown:
            raise ValueError(
                f"split JSON mismatch: missing sequences={missing}, unknown sequences={unknown}"
            )
    else:
        if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
            raise ValueError("train_fraction and val_fraction must lie in (0, 1)")
        if train_fraction + val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be less than 1")
        shuffled = np.asarray(sequences, dtype=object)
        np.random.default_rng(seed).shuffle(shuffled)
        count = len(sequences)
        train_count = max(1, int(np.floor(count * train_fraction)))
        val_count = max(1, int(np.floor(count * val_fraction)))
        if train_count + val_count >= count:
            val_count = 1
            train_count = count - 2
        assignment = {}
        for index, sequence_id in enumerate(shuffled.tolist()):
            split = "train" if index < train_count else "val" if index < train_count + val_count else "test"
            assignment[str(sequence_id)] = split

    for split in ("train", "val", "test"):
        if split not in assignment.values():
            raise ValueError(f"split assignment contains no {split} sequences")
    return assignment


def _rng_for(seed: int, *parts: str) -> np.random.Generator:
    message = "\x1f".join((str(seed), *parts)).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(message).digest()[:8], "little")
    return np.random.default_rng(derived)


def _step_schedule_indices(
    count: int,
    steps: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    selected = [0]
    while selected[-1] < count - 1:
        remaining = count - 1 - selected[-1]
        feasible = [int(step) for step in steps if step <= remaining]
        if not feasible:
            break
        step = feasible[int(rng.integers(0, len(feasible)))]
        selected.append(selected[-1] + step)
    return np.asarray(selected, dtype=np.int64)


def _bounded_time_schedule_indices(
    timestamps: np.ndarray,
    steps: Sequence[int],
    minimum_seconds: float,
    maximum_seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if timestamps.size == 0:
        return np.empty(0, dtype=np.int64)
    selected = [0]
    while selected[-1] < timestamps.size - 1:
        current = selected[-1]
        feasible = []
        for step in steps:
            candidate = current + int(step)
            if candidate >= timestamps.size:
                continue
            elapsed = timestamps[candidate] - timestamps[current]
            if minimum_seconds <= elapsed <= maximum_seconds:
                feasible.append(candidate)
        if not feasible:
            break
        selected.append(feasible[int(rng.integers(0, len(feasible)))])
    return np.asarray(selected, dtype=np.int64)


def _retained_indices(
    count: int,
    mode: str,
    rng: np.random.Generator,
    config: ActivityPreparationConfig,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if mode == "regular":
        return np.arange(count, dtype=np.int64)
    if mode == "seen_steps":
        if timestamps is None:
            return _step_schedule_indices(count, config.train_seen_steps, rng)
        return _bounded_time_schedule_indices(
            timestamps, config.train_seen_steps,
            config.train_gap_min_seconds, config.train_gap_max_seconds, rng,
        )
    if mode == "unseen_steps":
        if timestamps is None:
            return _step_schedule_indices(count, config.unseen_gap_steps, rng)
        return _bounded_time_schedule_indices(
            timestamps, config.unseen_gap_steps,
            config.unseen_gap_min_seconds, config.unseen_gap_max_seconds, rng,
        )
    if mode in {"random_75", "random_50"}:
        probability = 0.75 if mode == "random_75" else 0.50
        keep = rng.random(count) < probability
        keep[0] = True
        if count > 1:
            keep[-1] = True
        return np.flatnonzero(keep)
    if mode == "burst":
        keep = np.ones(count, dtype=bool)
        largest = min(config.burst_max, max(count - 2, 0))
        if largest >= config.burst_min:
            length = int(rng.integers(config.burst_min, largest + 1))
            start_max = count - length - 1
            start = int(rng.integers(1, start_max + 1))
            keep[start : start + length] = False
        return np.flatnonzero(keep)
    raise ValueError(f"Unknown observation condition: {mode}")


def _empty_window_arrays(
    window_length: int,
    feature_dim: int,
    class_count: int,
) -> dict[str, np.ndarray]:
    return {
        "x": np.zeros((0, window_length, feature_dim), dtype=np.float32),
        "delta_t": np.zeros((0, window_length), dtype=np.float32),
        "valid_mask": np.zeros((0, window_length), dtype=bool),
        "target_mask": np.zeros((0, window_length), dtype=bool),
        "y": np.zeros((0, window_length, class_count), dtype=np.float32),
        "sequence_ids": np.asarray([], dtype="<U1"),
        "track_ids": np.asarray([], dtype="<U1"),
        "frame_ids": np.full((0, window_length), -1, dtype=np.int64),
        "target_frame_ids": np.full((0, window_length), -1, dtype=np.int64),
    }


def build_activity_windows(
    tracks: Mapping[tuple[str, str], Sequence[_Observation]],
    *,
    sequence_ids: set[str],
    condition: str,
    config: ActivityPreparationConfig,
    feature_dim: int,
    class_count: int,
) -> dict[str, np.ndarray]:
    """Create fixed-length transition windows after true observation removal."""

    rows: dict[str, list[np.ndarray | str]] = {
        "x": [],
        "delta_t": [],
        "valid_mask": [],
        "target_mask": [],
        "y": [],
        "sequence_ids": [],
        "track_ids": [],
        "frame_ids": [],
        "target_frame_ids": [],
    }
    for (sequence_id, track_id), observations in sorted(tracks.items()):
        if sequence_id not in sequence_ids or len(observations) < 2:
            continue
        rng = _rng_for(config.seed, condition, sequence_id, track_id)
        timestamps = np.asarray(
            [observation.timestamp for observation in observations], dtype=np.float64
        )
        retained = _retained_indices(
            len(observations), condition, rng, config, timestamps=timestamps
        )
        transition_count = max(len(retained) - 1, 0)
        if transition_count == 0:
            continue
        if config.include_partial_windows:
            if transition_count <= config.window_length:
                starts = [0]
            else:
                starts = list(range(
                    0,
                    transition_count - config.window_length + 1,
                    config.window_stride,
                ))
                # Cover the terminal transition once without emitting several
                # overlapping partial windows with the same primary target.
                terminal_start = transition_count - config.window_length
                if terminal_start not in starts:
                    starts.append(terminal_start)
        else:
            stop = transition_count - config.window_length + 1
            starts = range(0, max(stop, 0), config.window_stride)

        for start in starts:
            valid_count = min(config.window_length, transition_count - start)
            if valid_count <= 0 or (
                valid_count < config.window_length and not config.include_partial_windows
            ):
                continue
            x = np.zeros((config.window_length, feature_dim), dtype=np.float32)
            delta_t = np.zeros(config.window_length, dtype=np.float32)
            valid_mask = np.zeros(config.window_length, dtype=bool)
            target_mask = np.zeros(config.window_length, dtype=bool)
            y = np.zeros((config.window_length, class_count), dtype=np.float32)
            frame_ids = np.full(config.window_length, -1, dtype=np.int64)
            target_frame_ids = np.full(config.window_length, -1, dtype=np.int64)

            for local_index in range(valid_count):
                retained_index = start + local_index
                current = observations[int(retained[retained_index])]
                target = observations[int(retained[retained_index + 1])]
                gap = target.timestamp - current.timestamp
                if not np.isfinite(gap) or gap <= 0:
                    raise ValueError(
                        "Retained observations must have positive outgoing gaps; "
                        f"sequence={sequence_id}, track={track_id}"
                    )
                x[local_index] = current.x
                delta_t[local_index] = np.float32(gap)
                valid_mask[local_index] = True
                y[local_index] = target.y
                frame_ids[local_index] = current.frame_id
                target_frame_ids[local_index] = target.frame_id
            target_mask[valid_count - 1] = True
            rows["x"].append(x)
            rows["delta_t"].append(delta_t)
            rows["valid_mask"].append(valid_mask)
            rows["target_mask"].append(target_mask)
            rows["y"].append(y)
            rows["sequence_ids"].append(sequence_id)
            rows["track_ids"].append(track_id)
            rows["frame_ids"].append(frame_ids)
            rows["target_frame_ids"].append(target_frame_ids)

    if not rows["x"]:
        return _empty_window_arrays(config.window_length, feature_dim, class_count)
    return {
        "x": np.stack(rows["x"]).astype(np.float32, copy=False),
        "delta_t": np.stack(rows["delta_t"]).astype(np.float32, copy=False),
        "valid_mask": np.stack(rows["valid_mask"]).astype(bool, copy=False),
        "target_mask": np.stack(rows["target_mask"]).astype(bool, copy=False),
        "y": np.stack(rows["y"]).astype(np.float32, copy=False),
        "sequence_ids": np.asarray(rows["sequence_ids"], dtype=str),
        "track_ids": np.asarray(rows["track_ids"], dtype=str),
        "frame_ids": np.stack(rows["frame_ids"]).astype(np.int64, copy=False),
        "target_frame_ids": np.stack(rows["target_frame_ids"]).astype(np.int64, copy=False),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _delta_t_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values = arrays["delta_t"][arrays["valid_mask"]].astype(np.float64)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "unique": []}
    unique = np.unique(np.round(values, decimals=9))
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "unique": unique[:100].tolist(),
        "unique_truncated": bool(unique.size > 100),
    }


def prepare_jrdb_activity_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    config: ActivityPreparationConfig | None = None,
    split_json: str | Path | None = None,
    overwrite: bool = False,
    source_kind: str = "canonical",
) -> dict[str, Any]:
    """Prepare sequence-disjoint irregular-observation activity files."""

    config = config or ActivityPreparationConfig()
    config.validate()
    output = Path(output_dir)
    filenames = {
        "train_seen": ("train", "seen_steps", "train_seen.npz"),
        "val_seen": ("val", "seen_steps", "val_seen.npz"),
        "test_seen": ("test", "regular", "test_seen.npz"),
        "test_random_75": ("test", "random_75", "test_random_75.npz"),
        "test_random_50": ("test", "random_50", "test_random_50.npz"),
        "test_burst": ("test", "burst", "test_burst.npz"),
        "test_unseen_gap": ("test", "unseen_steps", "test_unseen_gap.npz"),
    }
    destinations = [output / value[2] for value in filenames.values()] + [output / "metadata.json"]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist; pass overwrite=True or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    entities, labels, optional_available = load_canonical_activity_csv(
        input_csv, heading_degrees=config.heading_degrees
    )
    assignment = assign_sequence_splits(
        entities["sequence_id"].unique(), seed=config.seed, split_json=split_json
    )
    tracks = build_relational_observations(
        entities,
        labels,
        k_neighbors=config.k_neighbors,
        optional_available=optional_available,
    )
    feature_names = relational_feature_names(config.k_neighbors)
    split_sequences = {
        split: sorted(sequence for sequence, assigned in assignment.items() if assigned == split)
        for split in ("train", "val", "test")
    }

    file_metadata: dict[str, Any] = {}
    for name, (split, condition, filename) in filenames.items():
        arrays = build_activity_windows(
            tracks,
            sequence_ids=set(split_sequences[split]),
            condition=condition,
            config=config,
            feature_dim=len(feature_names),
            class_count=len(labels),
        )
        if name == "train_seen" and len(arrays["x"]) == 0:
            raise ValueError("No training windows were produced; check tracks and window settings")
        _write_npz(output / filename, arrays)
        file_metadata[name] = {
            "file": filename,
            "split": split,
            "observation_condition": condition,
            "windows": int(len(arrays["x"])),
            "valid_transitions": int(arrays["valid_mask"].sum()),
            "primary_targets": int(arrays["target_mask"].sum()),
            "delta_t_seconds": _delta_t_summary(arrays),
            "sha256": _sha256_file(output / filename),
        }

    metadata = {
        "format_version": FORMAT_VERSION,
        "source_kind": str(source_kind),
        "source_csv": str(Path(input_csv).resolve()),
        "source_csv_sha256": _sha256_file(Path(input_csv)),
        "canonical_required_columns": list(REQUIRED_COLUMNS),
        "optional_columns_available": optional_available,
        "classes": labels,
        "class_to_index": {label: index for index, label in enumerate(labels)},
        "feature_names": feature_names,
        "feature_dim": len(feature_names),
        "focal_feature_dim": len(FOCAL_FEATURE_NAMES),
        "neighbor_feature_dim": len(NEIGHBOR_FEATURE_NAMES),
        "config": {**asdict(config), "train_seen_steps": list(config.train_seen_steps), "unseen_gap_steps": list(config.unseen_gap_steps)},
        "split_sequences": split_sequences,
        "sequence_assignment": assignment,
        "files": file_metadata,
        "semantics": {
            "x_t": "Target-centered focal and nearest-neighbor state at a retained observation.",
            "delta_t_t": "Actual timestamp difference to the next retained observation.",
            "y_t": "Multi-hot activity state at that next retained observation.",
            "valid_mask": "True where x, delta_t, and y form a real transition; False for padding.",
            "target_mask": "True only at the final valid anchor-to-query transition in each window.",
            "irregularity": "Created by removing observations before computing outgoing timestamp gaps; timestamps are never synthetically perturbed.",
        },
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def generate_synthetic_canonical_csv(
    path: str | Path,
    *,
    seed: int = 2027,
    sequences: int = 6,
    frames: int = 24,
    tracks: int = 4,
) -> Path:
    """Generate a deterministic canonical table for smoke tests only."""

    if sequences < 3 or frames < 2 or tracks < 1:
        raise ValueError("synthetic data requires sequences>=3, frames>=2, tracks>=1")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for sequence_index in range(sequences):
        sequence_id = f"synthetic_{sequence_index:02d}"
        phase = rng.uniform(-0.25, 0.25)
        for frame_id in range(frames):
            timestamp = float(frame_id)
            for track_index in range(tracks):
                track_id = f"person_{track_index:02d}"
                x = 1.5 * track_index + 0.12 * frame_id
                y = 0.6 * np.sin(0.2 * frame_id + track_index + phase)
                z = 0.9 + 0.02 * track_index
                action_ids = [f"motion_{(frame_id // 4 + track_index) % 3}"]
                if (frame_id + track_index + sequence_index) % 5 == 0:
                    action_ids.append("social_interaction")
                for action_id in action_ids:
                    rows.append(
                        {
                            "sequence_id": sequence_id,
                            "frame_id": frame_id,
                            "track_id": track_id,
                            "timestamp": timestamp,
                            "x": x,
                            "y": y,
                            "z": z,
                            "action_id": action_id,
                            "length": 0.5,
                            "width": 0.5,
                            "height": 1.6 + 0.03 * track_index,
                            "heading": 0.05 * frame_id,
                            "occlusion": float((frame_id + track_index) % 7 == 0),
                        }
                    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


class NPZActivityDataset(Dataset):
    """In-memory PyTorch dataset for one prepared activity NPZ file."""

    REQUIRED_ARRAYS = (
        "x",
        "delta_t",
        "valid_mask",
        "target_mask",
        "y",
        "sequence_ids",
        "track_ids",
        "frame_ids",
        "target_frame_ids",
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=False) as payload:
            missing = [name for name in self.REQUIRED_ARRAYS if name not in payload]
            if missing:
                raise ValueError(f"Activity NPZ is missing arrays: {missing}")
            self.x = torch.from_numpy(payload["x"].astype(np.float32, copy=True))
            self.delta_t = torch.from_numpy(payload["delta_t"].astype(np.float32, copy=True))
            self.valid_mask = torch.from_numpy(payload["valid_mask"].astype(bool, copy=True))
            self.target_mask = torch.from_numpy(payload["target_mask"].astype(bool, copy=True))
            self.y = torch.from_numpy(payload["y"].astype(np.float32, copy=True))
            self.sequence_ids = payload["sequence_ids"].astype(str, copy=True)
            self.track_ids = payload["track_ids"].astype(str, copy=True)
            self.frame_ids = payload["frame_ids"].astype(np.int64, copy=True)
            self.target_frame_ids = payload["target_frame_ids"].astype(np.int64, copy=True)
        count = self.x.shape[0]
        if any(array.shape[0] != count for array in (self.delta_t, self.valid_mask, self.target_mask, self.y)):
            raise ValueError("Activity NPZ arrays disagree on window count")
        if (len(self.sequence_ids) != count or len(self.track_ids) != count
                or self.frame_ids.shape[0] != count
                or self.target_frame_ids.shape[0] != count):
            raise ValueError("Activity NPZ identifiers disagree on window count")
        if self.x.ndim != 3 or self.y.ndim != 3:
            raise ValueError("x and y must have shapes [N,T,D] and [N,T,C]")
        if self.delta_t.shape != self.valid_mask.shape or self.delta_t.shape != self.target_mask.shape:
            raise ValueError("delta_t, valid_mask, and target_mask must share shape [N,T]")
        if self.x.shape[:2] != self.delta_t.shape or self.y.shape[:2] != self.delta_t.shape:
            raise ValueError("Activity NPZ temporal dimensions do not align")
        if torch.any(self.target_mask & ~self.valid_mask):
            raise ValueError("target_mask must be a subset of valid_mask")
        if len(self) and torch.any(self.target_mask.sum(dim=1) != 1):
            raise ValueError("Every activity window must contain exactly one primary target")
        if torch.any(self.delta_t[self.valid_mask] <= 0):
            raise ValueError("Every valid transition must have a positive delta_t")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "x": self.x[index],
            "delta_t": self.delta_t[index],
            "valid_mask": self.valid_mask[index],
            "target_mask": self.target_mask[index],
            "y": self.y[index],
            "sequence_id": str(self.sequence_ids[index]),
            "track_id": str(self.track_ids[index]),
            "frame_ids": torch.from_numpy(self.frame_ids[index].copy()),
            "target_frame_ids": torch.from_numpy(self.target_frame_ids[index].copy()),
        }


def load_activity_metadata(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir)
    metadata_path = path if path.name == "metadata.json" else path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported activity metadata format: {payload.get('format_version')!r}"
        )
    return payload


__all__ = [
    "ActivityPreparationConfig",
    "FORMAT_VERSION",
    "NPZActivityDataset",
    "assign_sequence_splits",
    "build_activity_windows",
    "build_relational_observations",
    "generate_synthetic_canonical_csv",
    "load_activity_metadata",
    "load_canonical_activity_csv",
    "prepare_jrdb_activity_dataset",
    "relational_feature_names",
]
