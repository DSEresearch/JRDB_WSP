from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.jrdb_activity_data import NPZActivityDataset, load_activity_metadata
from src.jrdb_activity_models import MODEL_NAMES, build_activity_model


DEFAULT_MODELS = ["wsp", "time_gru", "time_transformer"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train shared-input temporal models for JRDB-Act next-activity "
            "prediction under saved irregular-observation protocols."
        )
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train_file", default="train_seen.npz")
    parser.add_argument("--val_file", default="val_seen.npz")
    parser.add_argument(
        "--test_files", nargs="+", default=["test_*.npz"],
        help="File names or glob patterns, relative to --data_dir.",
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--condition_rank", type=int, default=16)
    parser.add_argument(
        "--parameter_match", action=argparse.BooleanOptionalAction, default=True,
        help="Select baseline widths closest to the WSP trainable-parameter count.",
    )
    parser.add_argument("--parameter_match_tolerance", type=float, default=0.10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=["fp32", "bf16"], default="fp32",
        help="Use BF16 autocast on CUDA; CPU execution remains FP32.",
    )
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument(
        "--time_interventions", nargs="+", default=["correct"],
        choices=["correct", "constant", "shuffled", "zero"],
        help=(
            "Secondary timing-input controls. These do not create irregular "
            "sampling; the prepared test files do."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker_mode", action="store_true",
        help="Suppress shared manifests for a concurrent model/seed worker.",
    )
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_fingerprint(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    excluded = {"output_dir", "resume", "models", "seeds"}
    argument_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key not in excluded
    }
    code_paths = [
        Path(__file__),
        PROJECT_ROOT / "src" / "jrdb_activity_data.py",
        PROJECT_ROOT / "src" / "jrdb_activity_models.py",
        PROJECT_ROOT / "src" / "wsp_review_models.py",
    ]
    payload = {
        "arguments": argument_payload,
        "metadata": metadata,
        "code_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in code_paths},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def expand_test_files(data_dir: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if not matches and (data_dir / pattern).is_file():
            matches = [data_dir / pattern]
        paths.update(path.resolve() for path in matches)
    selected = sorted(paths)
    if not selected:
        raise FileNotFoundError(f"No test files matched {patterns} in {data_dir}")
    return selected


def make_loader(path: Path, batch_size: int, shuffle: bool, num_workers: int,
                seed: int) -> tuple[NPZActivityDataset, DataLoader]:
    dataset = NPZActivityDataset(path)
    if len(dataset) == 0:
        raise ValueError(f"Prepared activity file contains no windows: {path}")
    if not bool(dataset.target_mask.any()):
        raise ValueError(f"Prepared activity file contains no primary targets: {path}")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
        drop_last=False,
    )
    return dataset, loader


def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               valid_mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    ).mean(dim=-1)
    weight = valid_mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def class_pos_weight(dataset: NPZActivityDataset) -> torch.Tensor:
    y = np.asarray(dataset.y, dtype=np.float64)
    mask = np.asarray(dataset.target_mask, dtype=bool)
    selected = y[mask]
    positive = selected.sum(axis=0)
    negative = selected.shape[0] - positive
    # A class absent from training receives neutral weight and is reported as
    # undefined, rather than creating an infinite loss coefficient.
    weight = np.ones_like(positive)
    present = positive > 0
    weight[present] = negative[present] / positive[present]
    return torch.from_numpy(np.clip(weight, 0.25, 50.0).astype(np.float32))


def feature_normalization(dataset: NPZActivityDataset) -> tuple[torch.Tensor, torch.Tensor]:
    selected = dataset.x[dataset.valid_mask]
    if selected.numel() == 0:
        raise ValueError("Training data contain no valid observations")
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def evaluable_class_indices(dataset: NPZActivityDataset) -> np.ndarray:
    targets = np.asarray(dataset.y, dtype=np.float32)[
        np.asarray(dataset.target_mask, dtype=bool)
    ]
    indexes = np.flatnonzero(targets.sum(axis=0) > 0).astype(np.int64)
    if indexes.size == 0:
        raise ValueError("No class has a positive primary target in training data")
    return indexes


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


_WIDTH_CACHE: dict[tuple[Any, ...], tuple[int, int, int]] = {}


def matched_width(model_name: str, input_dim: int, num_classes: int,
                  time_scale: float, max_length: int,
                  args: argparse.Namespace) -> tuple[int, int, int]:
    """Return width, model count, and WSP reference count.

    Matching trainable counts is a capacity control, not a claim that the
    architectures have identical compute or memory use. Those are measured
    separately from the trained models.
    """
    key = (
        model_name, input_dim, num_classes, time_scale, max_length,
        args.d_model, args.num_layers, args.condition_rank, args.parameter_match,
        args.parameter_match_tolerance,
    )
    if key in _WIDTH_CACHE:
        return _WIDTH_CACHE[key]
    reference = build_activity_model(
        "wsp", input_dim=input_dim, num_classes=num_classes,
        time_scale=time_scale, d_model=args.d_model,
        num_layers=args.num_layers, condition_rank=args.condition_rank,
        max_length=max_length,
    )
    reference_count = trainable_parameters(reference)
    if model_name == "wsp" or not args.parameter_match:
        model = build_activity_model(
            model_name, input_dim=input_dim, num_classes=num_classes,
            time_scale=time_scale, d_model=args.d_model,
            num_layers=args.num_layers, condition_rank=args.condition_rank,
            max_length=max_length,
        )
        result = (args.d_model, trainable_parameters(model), reference_count)
        _WIDTH_CACHE[key] = result
        return result
    candidates = sorted(set([args.d_model, *range(16, 513, 4)]))
    selected: tuple[float, int, int] | None = None
    for width in candidates:
        candidate = build_activity_model(
            model_name, input_dim=input_dim, num_classes=num_classes,
            time_scale=time_scale, d_model=width,
            num_layers=args.num_layers, condition_rank=args.condition_rank,
            max_length=max_length,
        )
        count = trainable_parameters(candidate)
        choice = (abs(count - reference_count) / reference_count, width, count)
        if selected is None or choice < selected:
            selected = choice
    assert selected is not None
    if selected[0] > args.parameter_match_tolerance:
        raise ValueError(
            f"Closest {model_name} parameter match differs by {selected[0]:.3f}, "
            f"above tolerance {args.parameter_match_tolerance:.3f}"
        )
    result = (selected[1], selected[2], reference_count)
    _WIDTH_CACHE[key] = result
    return result


def change_timing(delta_t: torch.Tensor, valid_mask: torch.Tensor, name: str,
                  seed: int) -> torch.Tensor:
    if name == "correct":
        return delta_t
    if name == "zero":
        return torch.zeros_like(delta_t)
    changed = delta_t.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for row in range(delta_t.shape[0]):
        indexes = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if indexes.numel() == 0:
            continue
        values = delta_t[row, indexes]
        if name == "constant":
            changed[row, indexes] = values.mean()
        elif name == "shuffled":
            order = torch.randperm(indexes.numel(), generator=generator)
            changed[row, indexes] = values[order.to(values.device)]
        else:
            raise ValueError(f"Unknown timing intervention: {name}")
    return changed


def valid_arrays(probabilities: np.ndarray, targets: np.ndarray,
                 valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = valid_mask.astype(bool)
    return probabilities[mask], targets[mask]


def choose_thresholds(probabilities: np.ndarray, targets: np.ndarray,
                      valid_mask: np.ndarray,
                      class_indices: np.ndarray) -> np.ndarray:
    prob, y = valid_arrays(probabilities, targets, valid_mask)
    thresholds = np.full(y.shape[-1], 0.5, dtype=np.float32)
    grid = np.linspace(0.10, 0.90, 17)
    for class_index in class_indices.tolist():
        if np.unique(y[:, class_index]).size < 2:
            continue
        scores = [
            f1_score(
                y[:, class_index], prob[:, class_index] >= value,
                zero_division=0,
            )
            for value in grid
        ]
        thresholds[class_index] = float(grid[int(np.argmax(scores))])
    return thresholds


def expected_calibration_error(prob: np.ndarray, target: np.ndarray,
                               bins: int = 15) -> float:
    flat_prob = prob.reshape(-1)
    flat_target = target.reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (flat_prob >= lower) & (
            flat_prob <= upper if upper == 1.0 else flat_prob < upper
        )
        if np.any(in_bin):
            result += float(np.mean(in_bin)) * abs(
                float(np.mean(flat_prob[in_bin]) - np.mean(flat_target[in_bin]))
            )
    return result


def classification_metrics(probabilities: np.ndarray, targets: np.ndarray,
                           valid_mask: np.ndarray,
                           thresholds: np.ndarray,
                           class_indices: np.ndarray) -> dict[str, float]:
    prob, y = valid_arrays(probabilities, targets, valid_mask)
    prob = prob[:, class_indices]
    y = y[:, class_indices]
    selected_thresholds = thresholds[class_indices]
    per_class_ap = [
        0.0 if y[:, index].sum() == 0 else float(
            average_precision_score(y[:, index], prob[:, index])
        )
        for index in range(y.shape[-1])
    ]
    predicted = prob >= selected_thresholds[None, :]
    return {
        "macro_ap": float(np.mean(per_class_ap)) if per_class_ap else float("nan"),
        "micro_ap": float(average_precision_score(y.reshape(-1), prob.reshape(-1))),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "brier": float(np.mean(np.square(prob - y))),
        "ece": expected_calibration_error(prob, y),
        "valid_transitions": int(y.shape[0]),
        "evaluable_classes": int(len(class_indices)),
    }


def collect_predictions(model: torch.nn.Module, loader: DataLoader,
                        device: torch.device, intervention: str,
                        seed: int, feature_mean: torch.Tensor,
                        feature_std: torch.Tensor,
                        precision: str) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "probability": [], "target": [], "valid_mask": [], "target_mask": [], "delta_t": [],
        "sequence_id": [], "track_id": [], "frame_ids": [], "target_frame_ids": [],
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            x = (x - feature_mean) / feature_std
            delta_t = batch["delta_t"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
            modified = change_timing(
                delta_t, valid_mask, intervention, seed + batch_index
            )
            with autocast_context(device, precision):
                logits = model(x, modified, valid_mask)
            collected["probability"].append(
                torch.sigmoid(logits.float()).cpu().numpy()
            )
            collected["target"].append(batch["y"].cpu().numpy())
            collected["valid_mask"].append(valid_mask.cpu().numpy())
            collected["target_mask"].append(batch["target_mask"].cpu().numpy())
            collected["delta_t"].append(delta_t.cpu().numpy())
            sequence = batch["sequence_id"]
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.cpu().numpy().astype(str)
            else:
                sequence = np.asarray(sequence, dtype=str)
            collected["sequence_id"].append(sequence)
            collected["track_id"].append(np.asarray(batch["track_id"], dtype=str))
            collected["frame_ids"].append(batch["frame_ids"].cpu().numpy())
            collected["target_frame_ids"].append(
                batch["target_frame_ids"].cpu().numpy()
            )
    return {
        key: np.concatenate(values, axis=0) for key, values in collected.items()
    }


def sequence_bootstrap(predictions: dict[str, np.ndarray], thresholds: np.ndarray,
                       repeats: int, seed: int,
                       class_indices: np.ndarray) -> dict[str, float]:
    if repeats <= 0:
        return {}
    sequence = predictions["sequence_id"].astype(str)
    unique = np.unique(sequence)
    if unique.size < 2:
        return {}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        selected = rng.choice(unique, size=unique.size, replace=True)
        indexes = np.concatenate([np.flatnonzero(sequence == item) for item in selected])
        metrics = classification_metrics(
            predictions["probability"][indexes], predictions["target"][indexes],
            predictions["target_mask"][indexes], thresholds, class_indices,
        )
        if math.isfinite(metrics["macro_ap"]):
            estimates.append(metrics["macro_ap"])
    if not estimates:
        return {}
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "macro_ap_sequence_bootstrap_low": float(lower),
        "macro_ap_sequence_bootstrap_high": float(upper),
        "bootstrap_sequences": int(unique.size),
        "bootstrap_repeats": int(len(estimates)),
    }


def train_one(model: torch.nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, pos_weight: torch.Tensor,
              device: torch.device, args: argparse.Namespace,
              checkpoint_path: Path, seed: int,
              feature_mean: torch.Tensor,
              feature_std: torch.Tensor,
              class_indices: np.ndarray) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    candidate_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.candidate"
    )
    pos_weight = pos_weight.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            x = (x - feature_mean) / feature_std
            delta_t = batch["delta_t"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
            target_mask = batch["target_mask"].to(device, non_blocking=True).bool()
            target = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                logits = model(x, delta_t, valid_mask)
                loss = masked_bce(logits, target, target_mask, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            valid = int(target_mask.sum().item())
            train_loss += float(loss.detach()) * valid
            train_weight += valid
        validation = collect_predictions(
            model, val_loader, device, "correct", seed, feature_mean, feature_std,
            args.precision,
        )
        threshold = np.full(validation["target"].shape[-1], 0.5, dtype=np.float32)
        metrics = classification_metrics(
            validation["probability"], validation["target"],
            validation["target_mask"], threshold, class_indices,
        )
        score = metrics["macro_ap"]
        if not math.isfinite(score):
            score = -float("inf")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(train_weight, 1),
            "val_macro_ap": metrics["macro_ap"],
            "val_macro_f1_at_0.5": metrics["macro_f1"],
        })
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            stale = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, candidate_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    if not candidate_path.exists():
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    checkpoint = torch.load(candidate_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_path, checkpoint_path)
    return model, history


def run_model_seed(model_name: str, seed: int, metadata: dict[str, Any],
                   train_dataset: NPZActivityDataset, train_loader: DataLoader,
                   val_loader: DataLoader, test_loaders: dict[str, DataLoader],
                   device: torch.device, args: argparse.Namespace,
                   fingerprint: str) -> dict[str, Any]:
    seed_everything(seed)
    input_dim = int(train_dataset.x.shape[-1])
    num_classes = int(train_dataset.y.shape[-1])
    observed_train_gaps = train_dataset.delta_t[train_dataset.valid_mask]
    empirical_time_scale = float(observed_train_gaps.max()) if observed_train_gaps.numel() else 1.0
    time_scale = float(metadata.get("time_scale", metadata.get("max_train_gap", empirical_time_scale)))
    if time_scale <= 0:
        raise ValueError("metadata time_scale must be positive")
    selected_width, selected_count, reference_count = matched_width(
        model_name, input_dim, num_classes, time_scale,
        int(train_dataset.x.shape[1]), args,
    )
    # Width selection constructs temporary modules, so reset the seed before
    # initializing the model that is actually optimized.
    seed_everything(seed)
    model = build_activity_model(
        model_name,
        input_dim=input_dim,
        num_classes=num_classes,
        time_scale=time_scale,
        d_model=selected_width,
        num_layers=args.num_layers,
        condition_rank=args.condition_rank,
        max_length=int(train_dataset.x.shape[1]),
    ).to(device)
    parameter_count = trainable_parameters(model)
    if parameter_count != selected_count:
        raise RuntimeError("Parameter count changed between width selection and construction")
    run_dir = args.output_dir / model_name / f"seed_{seed}"
    result_path = run_dir / "result.json"
    if args.resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != fingerprint:
            raise ValueError(
                f"Refusing stale resume for {model_name}/seed_{seed}: protocol fingerprint differs"
            )
        return existing
    start = time.perf_counter()
    feature_mean, feature_std = feature_normalization(train_dataset)
    class_indices = evaluable_class_indices(train_dataset)
    feature_mean = feature_mean.to(device)
    feature_std = feature_std.to(device)
    model, history = train_one(
        model, train_loader, val_loader,
        class_pos_weight(train_dataset), device, args,
        run_dir / "best.pt", seed, feature_mean, feature_std, class_indices,
    )
    validation = collect_predictions(
        model, val_loader, device, "correct", seed, feature_mean, feature_std,
        args.precision,
    )
    thresholds = choose_thresholds(
        validation["probability"], validation["target"],
        validation["target_mask"], class_indices,
    )
    validation_metrics = classification_metrics(
        validation["probability"], validation["target"],
        validation["target_mask"], thresholds, class_indices,
    )
    tests: dict[str, dict[str, Any]] = {}
    for condition, loader in test_loaders.items():
        for intervention in args.time_interventions:
            key = condition if intervention == "correct" else f"{condition}__time_{intervention}"
            predictions = collect_predictions(
                model, loader, device, intervention, seed, feature_mean, feature_std,
                args.precision,
            )
            metrics = classification_metrics(
                predictions["probability"], predictions["target"],
                predictions["target_mask"], thresholds, class_indices,
            )
            metrics.update(sequence_bootstrap(
                predictions, thresholds, args.bootstrap_repeats, seed, class_indices
            ))
            tests[key] = metrics
            np.savez_compressed(
                run_dir / f"predictions_{key}.npz",
                probability=predictions["probability"],
                target=predictions["target"],
                valid_mask=predictions["valid_mask"],
                target_mask=predictions["target_mask"],
                delta_t=predictions["delta_t"],
                sequence_id=predictions["sequence_id"],
                track_id=predictions["track_id"],
                frame_ids=predictions["frame_ids"],
                target_frame_ids=predictions["target_frame_ids"],
            )
    result = {
        "model": model_name,
        "source_kind": metadata.get("source_kind", "unknown"),
        "protocol_fingerprint": fingerprint,
        "seed": seed,
        "parameter_count": int(parameter_count),
        "d_model": int(selected_width),
        "wsp_reference_parameter_count": int(reference_count),
        "parameter_match_relative_error": float(
            abs(parameter_count - reference_count) / reference_count
        ),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "time_scale": time_scale,
        "feature_mean": feature_mean.detach().cpu().tolist(),
        "feature_std": feature_std.detach().cpu().tolist(),
        "device": str(device),
        "precision": args.precision if device.type == "cuda" else "fp32",
        "elapsed_seconds": time.perf_counter() - start,
        "thresholds": thresholds.tolist(),
        "evaluable_class_indices": class_indices.tolist(),
        "validation": validation_metrics,
        "tests": tests,
        "history": history,
    }
    write_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if not 0.0 <= args.parameter_match_tolerance <= 1.0:
        raise ValueError("parameter_match_tolerance must lie in [0, 1]")
    if args.bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be non-negative")
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available: {sorted(MODEL_NAMES)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_activity_metadata(args.data_dir)
    fingerprint = protocol_fingerprint(args, metadata)
    device = pick_device(args.device)
    test_paths = expand_test_files(args.data_dir, args.test_files)
    manifest = {
        "data_dir": str(args.data_dir.resolve()),
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_files": [path.name for path in test_paths],
        "models": args.models,
        "seeds": args.seeds,
        "time_interventions": args.time_interventions,
        "timing_semantics": (
            "x_t and known outgoing delta_t_t predict the activity state at "
            "the next retained observation; missing frames are removed during preprocessing"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "metadata": metadata,
        "protocol_fingerprint": fingerprint,
    }
    if not args.worker_mode:
        write_json(args.output_dir / "run_manifest.json", manifest)
    all_results = []
    for seed in args.seeds:
        _, val_loader = make_loader(
            args.data_dir / args.val_file, args.batch_size, False,
            args.num_workers, seed,
        )
        test_loaders = {
            path.stem.removeprefix("test_"): make_loader(
                path, args.batch_size, False, args.num_workers, seed
            )[1]
            for path in test_paths
        }
        for model_name in args.models:
            # Recreate the shuffled loader with the same generator seed so the
            # temporal cores receive identical epoch-order namespaces.
            train_dataset, train_loader = make_loader(
                args.data_dir / args.train_file, args.batch_size, True,
                args.num_workers, seed,
            )
            print(f"training model={model_name} seed={seed} device={device}", flush=True)
            result = run_model_seed(
                model_name, seed, metadata, train_dataset, train_loader,
                val_loader, test_loaders, device, args,
                fingerprint,
            )
            all_results.append(result)
            print(
                f"finished model={model_name} seed={seed} "
                f"val_macro_ap={result['validation']['macro_ap']:.4f}",
                flush=True,
            )
    if not args.worker_mode:
        write_json(args.output_dir / "all_results.json", all_results)
    return all_results


if __name__ == "__main__":
    main()
