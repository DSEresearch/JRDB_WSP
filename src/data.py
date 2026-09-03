from __future__ import annotations

import re
import json
import pickle
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import read_json


FRAME_COLUMNS = ["frame", "frame_id", "frameid", "frame_no", "frame_number", "fid"]
TIME_COLUMNS = ["timestamp", "time", "ts", "t", "seconds", "sec"]
AGENT_COLUMNS = [
    "agent_id",
    "track_id",
    "trackid",
    "ped_id",
    "pedid",
    "pedestrian_id",
    "person_id",
    "object_id",
    "obj_id",
    "id",
]
X_COLUMNS = ["x", "pos_x", "position_x", "center_x", "cx", "bbox_center_x", "x_center"]
Y_COLUMNS = ["y", "pos_y", "position_y", "center_y", "cy", "bbox_center_y", "y_center"]
BBOX_SETS = [
    ("x1", "y1", "x2", "y2"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("x_min", "y_min", "x_max", "y_max"),
    ("left", "top", "right", "bottom"),
    ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
    ("xtl", "ytl", "xbr", "ybr"),
]
FEATURE_COLUMNS = ["x", "y", "vx", "vy", "ax", "ay"]
FEATURE_SETS = {
    "motion": FEATURE_COLUMNS,
    "position": ["x", "y"],
}
SUPPORTED_SUFFIXES = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".pkl", ".pickle", ".parquet", ".xml"}
YOLO_LABEL_COLUMNS = ["label", "x", "y", "width", "height"]


@dataclass(frozen=True)
class WindowConfig:
    obs_len: int = 8
    pred_len: int = 12
    stride: int = 1
    fps: float = 2.5
    min_dt: float = 1e-4
    irregular_obs: bool = False
    max_obs_skip: int = 3
    irregular_samples: int = 2
    seed: int = 42


def _clean_column(name: object) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _first_present(columns: Iterable[str], candidates: list[str]) -> str | None:
    cols = set(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    return None


def _looks_canonical(df: pd.DataFrame) -> bool:
    columns = {_clean_column(c) for c in df.columns}
    has_agent = _first_present(columns, AGENT_COLUMNS) is not None
    has_time = _first_present(columns, FRAME_COLUMNS) is not None or _first_present(columns, TIME_COLUMNS) is not None
    has_xy = _first_present(columns, X_COLUMNS) is not None and _first_present(columns, Y_COLUMNS) is not None
    has_bbox = any(all(col in columns for col in bbox) for bbox in BBOX_SETS)
    return has_agent and has_time and (has_xy or has_bbox)


def _assign_headerless_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = df.shape[1]
    if n >= 6:
        names = ["frame", "agent_id", "x", "y", "vx", "vy"] + [f"extra_{i}" for i in range(n - 6)]
    elif n == 5:
        names = ["frame", "agent_id", "x", "y", "label"]
    elif n >= 4:
        names = ["frame", "agent_id", "x", "y"] + [f"extra_{i}" for i in range(n - 4)]
    else:
        names = [f"col_{i}" for i in range(n)]
    df.columns = names[:n]
    return df


def _is_probable_yolo_label_path(path: Path) -> bool:
    return path.suffix.lower() == ".txt" and "labels" in {part.lower() for part in path.parts}


def _read_yolo_label_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|,|\t", engine="python", comment="#", header=None)
    if df.empty or df.shape[1] < 5:
        return pd.DataFrame(columns=YOLO_LABEL_COLUMNS)
    df = df.iloc[:, :5].copy()
    df.columns = YOLO_LABEL_COLUMNS
    for col in YOLO_LABEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=YOLO_LABEL_COLUMNS).reset_index(drop=True)


def _frame_from_stem(path: Path, fallback: int) -> int:
    match = re.search(r"\d+", path.stem)
    return int(match.group(0)) if match else fallback


def _sequence_name_for_label_file(path: Path, data_root: Path) -> str:
    parts = path.relative_to(data_root).parts
    if "labels" in parts:
        idx = parts.index("labels")
        if idx + 1 < len(parts):
            return "/".join(parts[: idx + 2])
    return path.parent.relative_to(data_root).as_posix()


def _link_sequence_detections(
    detections: pd.DataFrame,
    source: str,
    fps: float,
    max_gap: int = 3,
    max_distance: float | None = None,
) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=["source", "agent_id", "frame", "timestamp", "x", "y"])

    coord_max = float(detections[["x", "y"]].max().max())
    threshold = max_distance if max_distance is not None else (0.08 if coord_max <= 2.0 else 80.0)
    active: dict[int, dict[str, float]] = {}
    next_track_id = 0
    rows = []

    for frame, frame_df in detections.sort_values(["frame"]).groupby("frame", sort=True):
        frame_int = int(frame)
        dets = frame_df[["label", "x", "y"]].to_dict("records")
        active_ids = [tid for tid, state in active.items() if frame_int - int(state["frame"]) <= max_gap]
        pairs = []
        for det_idx, det in enumerate(dets):
            for tid in active_ids:
                state = active[tid]
                if int(det["label"]) != int(state["label"]):
                    continue
                dist = float(np.hypot(float(det["x"]) - state["x"], float(det["y"]) - state["y"]))
                if dist <= threshold:
                    pairs.append((dist, det_idx, tid))
        assigned_dets: set[int] = set()
        assigned_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        for _, det_idx, tid in sorted(pairs, key=lambda item: item[0]):
            if det_idx in assigned_dets or tid in assigned_tracks:
                continue
            assigned_dets.add(det_idx)
            assigned_tracks.add(tid)
            assignments[det_idx] = tid

        for det_idx, det in enumerate(dets):
            tid = assignments.get(det_idx)
            if tid is None:
                tid = next_track_id
                next_track_id += 1
            active[tid] = {
                "frame": frame_int,
                "x": float(det["x"]),
                "y": float(det["y"]),
                "label": float(det["label"]),
            }
            rows.append(
                {
                    "source": source,
                    "agent_id": f"{source}::track_{tid}",
                    "frame": frame_int,
                    "timestamp": frame_int / float(fps),
                    "x": float(det["x"]),
                    "y": float(det["y"]),
                }
            )
    return pd.DataFrame(rows)


def load_yolo_label_trajectories(
    data_dir: str | Path,
    fps: float,
    max_track_gap: int = 10,
    max_link_distance: float | None = None,
) -> pd.DataFrame:
    data_root = Path(data_dir)
    files = sorted(path for path in data_root.rglob("*.txt") if _is_probable_yolo_label_path(path))
    if not files:
        return pd.DataFrame()

    per_sequence: dict[str, list[pd.DataFrame]] = {}
    for fallback_frame, path in enumerate(files):
        df = _read_yolo_label_file(path)
        if df.empty:
            continue
        df["frame"] = _frame_from_stem(path, fallback=fallback_frame)
        source = _sequence_name_for_label_file(path, data_root)
        per_sequence.setdefault(source, []).append(df)

    linked = []
    for source, chunks in per_sequence.items():
        detections = pd.concat(chunks, ignore_index=True)
        linked_seq = _link_sequence_detections(
            detections,
            source=source,
            fps=fps,
            max_gap=max_track_gap,
            max_distance=max_link_distance,
        )
        if not linked_seq.empty:
            linked.append(linked_seq)
    return pd.concat(linked, ignore_index=True) if linked else pd.DataFrame()


def _dataframe_from_array(arr: Any) -> pd.DataFrame:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 3:
        frames = []
        for seq_idx in range(arr.shape[0]):
            g = _assign_headerless_columns(pd.DataFrame(arr[seq_idx]))
            if "agent_id" not in g.columns:
                g["agent_id"] = seq_idx
            frames.append(g)
        return pd.concat(frames, ignore_index=True)
    return _assign_headerless_columns(pd.DataFrame(arr))


def _dataframe_from_pickle_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, np.ndarray):
        return _dataframe_from_array(payload)
    if isinstance(payload, dict):
        for key in ["data", "tracks", "trajectories", "trajectory", "annotations", "sequences"]:
            if key in payload:
                return _dataframe_from_pickle_payload(payload[key])
        try:
            return pd.DataFrame(payload)
        except ValueError:
            return pd.json_normalize(payload)
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in payload):
            return pd.json_normalize(payload)
        return _dataframe_from_array(payload)
    return pd.DataFrame()


def _read_json_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "track" in item and isinstance(item["track"], dict):
                    track = item["track"]
                    rows.append(
                        {
                            "frame": track.get("f", track.get("frame")),
                            "agent_id": track.get("p", track.get("pedestrian", track.get("id"))),
                            "x": track.get("x"),
                            "y": track.get("y"),
                        }
                    )
                elif isinstance(item, dict):
                    rows.append(item)
        return pd.json_normalize(rows)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _dataframe_from_pickle_payload(payload)


def _read_xml_tracks(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for track in root.findall(".//track"):
        agent_id = track.attrib.get("id", track.attrib.get("name", "unknown"))
        label = track.attrib.get("label", "")
        for box in track.findall(".//box"):
            if box.attrib.get("outside") == "1":
                continue
            row = {"agent_id": agent_id, "label": label, "frame": box.attrib.get("frame")}
            for key in ["xtl", "ytl", "xbr", "ybr"]:
                row[key] = box.attrib.get(key)
            rows.append(row)
    return pd.DataFrame(rows)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if _is_probable_yolo_label_path(path):
        df = _read_yolo_label_file(path)
        df["frame"] = _frame_from_stem(path, fallback=0)
        df["agent_id"] = np.arange(len(df))
    elif suffix == ".csv":
        df = pd.read_csv(path)
        if not _looks_canonical(df):
            retry = pd.read_csv(path, header=None)
            if _looks_canonical(_assign_headerless_columns(retry)):
                df = _assign_headerless_columns(retry)
    elif suffix in {".txt", ".tsv"}:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
        if not _looks_canonical(df):
            retry = pd.read_csv(path, sep=r"\s+|,|\t", engine="python", comment="#", header=None)
            if _looks_canonical(_assign_headerless_columns(retry)):
                df = _assign_headerless_columns(retry)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        df = _read_json_table(path)
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            df = _dataframe_from_pickle_payload(pickle.load(f))
    elif suffix == ".xml":
        df = _read_xml_tracks(path)
    else:
        raise ValueError(f"Unsupported annotation suffix: {suffix}")
    df.columns = [_clean_column(c) for c in df.columns]
    return df


def discover_annotation_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def canonicalize_table(path: Path, fps: float) -> pd.DataFrame | None:
    df = _read_table(path)
    if df.empty:
        return None

    agent_col = _first_present(df.columns, AGENT_COLUMNS)
    frame_col = _first_present(df.columns, FRAME_COLUMNS)
    time_col = _first_present(df.columns, TIME_COLUMNS)
    x_col = _first_present(df.columns, X_COLUMNS)
    y_col = _first_present(df.columns, Y_COLUMNS)

    if x_col is None or y_col is None:
        for x1, y1, x2, y2 in BBOX_SETS:
            if all(col in df.columns for col in [x1, y1, x2, y2]):
                df["x"] = (pd.to_numeric(df[x1], errors="coerce") + pd.to_numeric(df[x2], errors="coerce")) / 2.0
                df["y"] = (pd.to_numeric(df[y1], errors="coerce") + pd.to_numeric(df[y2], errors="coerce")) / 2.0
                x_col, y_col = "x", "y"
                break

    if agent_col is None or (frame_col is None and time_col is None) or x_col is None or y_col is None:
        return None

    out = pd.DataFrame()
    out["source"] = path.relative_to(path.parents[0] if len(path.parents) else path).as_posix()
    out["agent_id"] = df[agent_col].astype(str)
    if frame_col is not None:
        out["frame"] = pd.to_numeric(df[frame_col], errors="coerce")
    else:
        out["frame"] = np.nan
    if time_col is not None:
        out["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        out["timestamp"] = out["frame"] / float(fps)
    out["x"] = pd.to_numeric(df[x_col], errors="coerce")
    out["y"] = pd.to_numeric(df[y_col], errors="coerce")
    out = out.dropna(subset=["agent_id", "timestamp", "x", "y"]).copy()
    return out


def inspect_annotation_files(data_dir: str | Path, limit: int = 20) -> dict[str, Any]:
    data_root = Path(data_dir)
    all_files = sorted(path for path in data_root.rglob("*") if path.is_file())
    candidates = discover_annotation_files(data_root)
    report: dict[str, Any] = {
        "data_dir": str(data_root),
        "exists": data_root.exists(),
        "total_files": len(all_files),
        "suffix_counts": {},
        "candidate_files": len(candidates),
        "yolo_label_files": len([path for path in candidates if _is_probable_yolo_label_path(path)]),
        "samples": [],
    }
    for path in all_files:
        suffix = path.suffix.lower() or "<no_suffix>"
        report["suffix_counts"][suffix] = report["suffix_counts"].get(suffix, 0) + 1
    for path in candidates[:limit]:
        sample: dict[str, Any] = {"path": path.relative_to(data_root).as_posix(), "suffix": path.suffix.lower()}
        try:
            df = _read_table(path)
            sample["rows"] = int(len(df))
            sample["columns"] = list(map(str, df.columns[:30]))
            sample["probable_yolo_label"] = _is_probable_yolo_label_path(path)
            sample["readable"] = canonicalize_table(path, fps=2.5) is not None
        except Exception as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
            sample["readable"] = False
        report["samples"].append(sample)
    return report


def load_raw_trajectories(
    data_dir: str | Path,
    fps: float,
    max_track_gap: int = 10,
    max_link_distance: float | None = None,
) -> pd.DataFrame:
    rows = []
    data_root = Path(data_dir)
    rejected = []
    yolo_raw = load_yolo_label_trajectories(
        data_root,
        fps=fps,
        max_track_gap=max_track_gap,
        max_link_distance=max_link_distance,
    )
    if not yolo_raw.empty:
        return yolo_raw.sort_values(["source", "agent_id", "timestamp", "frame"]).drop_duplicates(
            subset=["source", "agent_id", "timestamp"], keep="first"
        ).reset_index(drop=True)
    for path in discover_annotation_files(data_root):
        try:
            df = canonicalize_table(path, fps=fps)
        except Exception as exc:
            rejected.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if df is not None and not df.empty:
            df["source"] = path.relative_to(data_root).as_posix()
            rows.append(df)
        else:
            rejected.append((path, "missing required frame/time, agent_id, and position/bbox columns"))
    if not rows:
        report = inspect_annotation_files(data_root, limit=12)
        sample_lines = []
        for sample in report["samples"]:
            detail = sample.get("error") or f"columns={sample.get('columns', [])}"
            sample_lines.append(f"  - {sample['path']}: {detail}")
        suffix_counts = ", ".join(f"{k}:{v}" for k, v in sorted(report["suffix_counts"].items())) or "none"
        raise RuntimeError(
            f"No readable trajectory annotation files found under {data_root}.\n"
            f"Folder exists: {report['exists']}; total files: {report['total_files']}; "
            f"supported candidates: {report['candidate_files']}; suffixes: {suffix_counts}\n"
            f"Sample candidate diagnostics:\n" + ("\n".join(sample_lines) if sample_lines else "  <no supported candidate files found>")
        )
    raw = pd.concat(rows, ignore_index=True)
    raw = raw.sort_values(["source", "agent_id", "timestamp", "frame"]).drop_duplicates(
        subset=["source", "agent_id", "timestamp"], keep="first"
    )
    return raw.reset_index(drop=True)


def add_motion_features(raw: pd.DataFrame, min_dt: float) -> pd.DataFrame:
    out = []
    for (_, _), group in raw.groupby(["source", "agent_id"], sort=False):
        g = group.sort_values(["timestamp", "frame"]).copy()
        dt = g["timestamp"].diff().to_numpy(dtype=np.float32)
        finite_dt = dt[np.isfinite(dt) & (dt > min_dt)]
        fallback = float(np.median(finite_dt)) if len(finite_dt) else 1.0
        dt[0] = fallback
        dt = np.maximum(dt, min_dt)
        xy = g[["x", "y"]].to_numpy(dtype=np.float32)
        vel = np.zeros_like(xy)
        acc = np.zeros_like(xy)
        vel[1:] = (xy[1:] - xy[:-1]) / dt[1:, None]
        vel[0] = vel[1] if len(vel) > 1 else 0.0
        acc[1:] = (vel[1:] - vel[:-1]) / dt[1:, None]
        acc[0] = acc[1] if len(acc) > 1 else 0.0
        g["delta_t"] = dt
        g["vx"], g["vy"] = vel[:, 0], vel[:, 1]
        g["ax"], g["ay"] = acc[:, 0], acc[:, 1]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _regular_obs_indices(start: int, obs_len: int) -> np.ndarray:
    return np.arange(start, start + obs_len, dtype=np.int64)


def _irregular_obs_indices(start: int, obs_len: int, max_obs_skip: int, rng: np.random.Generator) -> np.ndarray:
    skips = rng.integers(1, max_obs_skip + 1, size=obs_len - 1)
    return np.concatenate([[start], start + np.cumsum(skips)]).astype(np.int64)


def _window_delta_t(times: np.ndarray, obs_indices: np.ndarray, fallback_dt: float, min_dt: float) -> np.ndarray:
    obs_times = times[obs_indices].astype(np.float32)
    dt = np.empty(len(obs_indices), dtype=np.float32)
    dt[0] = fallback_dt
    if len(obs_indices) > 1:
        dt[1:] = obs_times[1:] - obs_times[:-1]
    return np.maximum(dt, min_dt)


def make_windows(features: pd.DataFrame, cfg: WindowConfig) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(cfg.seed)
    max_obs_span = 1 + (cfg.obs_len - 1) * max(1, cfg.max_obs_skip if cfg.irregular_obs else 1)
    required_len = max_obs_span + cfg.pred_len
    for (source, agent_id), group in features.groupby(["source", "agent_id"], sort=False):
        g = group.sort_values(["timestamp", "frame"]).reset_index(drop=True)
        if len(g) < required_len:
            continue
        feat = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        row_dt = g["delta_t"].to_numpy(dtype=np.float32)
        future_xy = g[["x", "y"]].to_numpy(dtype=np.float32)
        frames = g["frame"].to_numpy()
        times = g["timestamp"].to_numpy(dtype=np.float32)
        for start in range(0, len(g) - required_len + 1, cfg.stride):
            samples = cfg.irregular_samples if cfg.irregular_obs else 1
            for sample_id in range(samples):
                if cfg.irregular_obs:
                    obs_indices = _irregular_obs_indices(start, cfg.obs_len, cfg.max_obs_skip, rng)
                    if obs_indices[-1] + cfg.pred_len >= len(g):
                        continue
                else:
                    obs_indices = _regular_obs_indices(start, cfg.obs_len)
                pred_start = int(obs_indices[-1]) + 1
                pred_indices = np.arange(pred_start, pred_start + cfg.pred_len, dtype=np.int64)
                fallback_dt = float(row_dt[int(obs_indices[0])])
                row: dict[str, object] = {
                    "source": source,
                    "agent_id": agent_id,
                    "start_index": start,
                    "start_frame": frames[int(obs_indices[0])],
                    "start_time": float(times[int(obs_indices[0])]),
                    "irregular_sample_id": sample_id,
                    "obs_frame_gaps": ",".join(map(str, np.diff(frames[obs_indices]).astype(int).tolist())),
                }
                obs_feat = feat[obs_indices]
                obs_dt = _window_delta_t(times, obs_indices, fallback_dt=fallback_dt, min_dt=cfg.min_dt)
                pred_xy = future_xy[pred_indices]
                for i in range(cfg.obs_len):
                    row[f"obs_{i}_frame"] = float(frames[int(obs_indices[i])])
                    row[f"obs_{i}_time"] = float(times[int(obs_indices[i])])
                    for j, name in enumerate(FEATURE_COLUMNS):
                        row[f"obs_{i}_{name}"] = float(obs_feat[i, j])
                    row[f"obs_{i}_dt"] = float(obs_dt[i])
                for i in range(cfg.pred_len):
                    row[f"target_{i}_frame"] = float(frames[int(pred_indices[i])])
                    row[f"target_{i}_time"] = float(times[int(pred_indices[i])])
                    row[f"target_{i}_x"] = float(pred_xy[i, 0])
                    row[f"target_{i}_y"] = float(pred_xy[i, 1])
                rows.append(row)
    if not rows:
        raise RuntimeError("No fixed-length windows could be generated. Check obs_len, pred_len, and dataset columns.")
    return pd.DataFrame(rows)


def split_windows(
    windows: pd.DataFrame,
    seed: int,
    val_size: float,
    test_size: float,
    split_unit: str = "source_agent",
) -> dict[str, pd.DataFrame]:
    """Split windows without crossing the requested grouping boundary.

    ``source_agent`` preserves the original agent-disjoint protocol. ``source``
    provides the stricter scene/source-disjoint evaluation when at least three
    independent sources are available.
    """
    if split_unit not in {"source_agent", "source"}:
        raise ValueError("split_unit must be 'source_agent' or 'source'")
    if split_unit == "source":
        key_values = windows["source"].astype(str).drop_duplicates()
    else:
        keys = windows[["source", "agent_id"]].drop_duplicates().copy()
        key_values = keys["source"].astype(str) + "::" + keys["agent_id"].astype(str)
    if len(key_values) < 3:
        if split_unit == "source":
            raise ValueError("source-disjoint splitting requires at least three distinct sources")
        out = windows.copy()
        return {
            "train": out.reset_index(drop=True),
            "val": out.iloc[0:0].reset_index(drop=True),
            "test": out.iloc[0:0].reset_index(drop=True),
        }
    train_keys, test_keys = train_test_split(key_values, test_size=test_size, random_state=seed, shuffle=True)
    if len(train_keys) < 2:
        train_keys = key_values
        val_keys = key_values.iloc[0:0]
        test_keys = key_values.iloc[0:0]
    else:
        relative_val = val_size / max(1e-8, 1.0 - test_size)
        train_keys, val_keys = train_test_split(train_keys, test_size=relative_val, random_state=seed, shuffle=True)
    split_map = {key: "train" for key in train_keys}
    split_map.update({key: "val" for key in val_keys})
    split_map.update({key: "test" for key in test_keys})
    row_keys = (
        windows["source"].astype(str)
        if split_unit == "source"
        else windows["source"].astype(str) + "::" + windows["agent_id"].astype(str)
    )
    out = windows.copy()
    out["split"] = row_keys.map(split_map)
    return {name: out[out["split"] == name].drop(columns=["split"]).reset_index(drop=True) for name in ["train", "val", "test"]}


def fit_normalization(train_windows: pd.DataFrame, obs_len: int) -> dict[str, list[float]]:
    feature_cols = [f"obs_{i}_{name}" for i in range(obs_len) for name in FEATURE_COLUMNS]
    dt_cols = [f"obs_{i}_dt" for i in range(obs_len)]
    features = train_windows[feature_cols].to_numpy(dtype=np.float32).reshape(-1, len(FEATURE_COLUMNS))
    dt = train_windows[dt_cols].to_numpy(dtype=np.float32).reshape(-1, 1)
    return {
        "feature_mean": features.mean(axis=0).tolist(),
        "feature_std": np.maximum(features.std(axis=0), 1e-6).tolist(),
        "dt_mean": [float(dt.mean())],
        "dt_std": [float(max(dt.std(), 1e-6))],
        "feature_columns": FEATURE_COLUMNS,
    }


class TrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        normalization_path: str | Path,
        obs_len: int,
        pred_len: int,
        t_mode: str = "real",
        input_features: str = "motion",
        t_seed: int = 0,
        motion_dt_policy: str = "fixed",
    ):
        self.df = pd.read_csv(csv_path)
        self.norm = read_json(normalization_path)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.t_mode = t_mode
        self.t_seed = int(t_seed)
        if motion_dt_policy not in {"fixed", "recompute"}:
            raise ValueError("motion_dt_policy must be 'fixed' or 'recompute'")
        self.motion_dt_policy = motion_dt_policy
        self.input_features = input_features
        if input_features not in FEATURE_SETS:
            raise ValueError(f"Unknown input_features={input_features}. Valid: {sorted(FEATURE_SETS)}")
        self.selected_features = FEATURE_SETS[input_features]
        self.feature_indices = [FEATURE_COLUMNS.index(name) for name in self.selected_features]
        self.all_feature_cols = [f"obs_{i}_{name}" for i in range(obs_len) for name in FEATURE_COLUMNS]
        self.dt_cols = [f"obs_{i}_dt" for i in range(obs_len)]
        self.target_cols = [col for i in range(pred_len) for col in (f"target_{i}_x", f"target_{i}_y")]
        all_feature_mean = np.asarray(self.norm["feature_mean"], dtype=np.float32)
        all_feature_std = np.asarray(self.norm["feature_std"], dtype=np.float32)
        self.feature_mean = all_feature_mean[self.feature_indices]
        self.feature_std = all_feature_std[self.feature_indices]
        self.dt_mean = float(self.norm["dt_mean"][0])
        self.dt_std = float(self.norm["dt_std"][0])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        all_x = row[self.all_feature_cols].to_numpy(dtype=np.float32).reshape(self.obs_len, len(FEATURE_COLUMNS))
        dt_original = row[self.dt_cols].to_numpy(dtype=np.float32)
        dt = dt_original.copy()
        y = row[self.target_cols].to_numpy(dtype=np.float32).reshape(self.pred_len, 2)
        if self.t_mode == "shuffled":
            # A fixed per-window permutation makes the ablation reproducible
            # across epochs, worker counts, and evaluation runs.
            rng = np.random.default_rng(self.t_seed + int(idx))
            dt = dt[rng.permutation(self.obs_len)]
        elif self.t_mode == "constant":
            dt.fill(self.dt_mean)
        elif self.t_mode != "real":
            raise ValueError(f"Unknown t_mode: {self.t_mode}")

        if self.motion_dt_policy == "recompute" and self.input_features == "motion" and self.t_mode != "real":
            xy = all_x[:, :2]
            velocity = np.zeros_like(xy)
            acceleration = np.zeros_like(xy)
            if self.obs_len > 1:
                safe_dt = np.maximum(dt, 1e-6)
                velocity[1:] = (xy[1:] - xy[:-1]) / safe_dt[1:, None]
                velocity[0] = velocity[1]
                acceleration[1:] = (velocity[1:] - velocity[:-1]) / safe_dt[1:, None]
                acceleration[0] = acceleration[1]
            all_x[:, 2:4] = velocity
            all_x[:, 4:6] = acceleration

        x = all_x[:, self.feature_indices]
        x = (x - self.feature_mean) / self.feature_std
        dt_raw = dt.copy()
        dt = (dt - self.dt_mean) / self.dt_std
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "delta_t": torch.tensor(dt, dtype=torch.float32),
            "delta_t_raw": torch.tensor(dt_raw, dtype=torch.float32),
            "delta_t_original_raw": torch.tensor(dt_original, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
        }
