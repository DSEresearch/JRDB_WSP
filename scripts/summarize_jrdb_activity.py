from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate JRDB-Act WSP experiment metrics across random seeds."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--primary_condition", default="unseen_gap")
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument(
        "--success_margin", type=float, default=0.02,
        help="Pre-registered practical macro-AP margin in absolute units.",
    )
    parser.add_argument("--primary_gap_min", type=float, default=4.0)
    parser.add_argument("--primary_gap_max", type=float, default=6.0)
    parser.add_argument("--minimum_seeds", type=int, default=3)
    parser.add_argument("--minimum_sequences", type=int, default=20)
    return parser.parse_args(argv)


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, float("nan"), float("nan")
    standard_error = float(np.std(array, ddof=1) / np.sqrt(array.size))
    # Seed-level normal interval is descriptive. Each run separately stores
    # the required video-sequence bootstrap interval.
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def macro_ap(probability: np.ndarray, target: np.ndarray,
             target_mask: np.ndarray,
             class_indices: np.ndarray) -> float:
    probability = probability[target_mask.astype(bool)]
    target = target[target_mask.astype(bool)]
    values = [
        0.0 if target[:, index].sum() == 0 else
        average_precision_score(target[:, index], probability[:, index])
        for index in class_indices.tolist()
    ]
    return float(np.mean(values)) if values else float("nan")


def load_predictions(input_dir: Path, model: str, seed: int,
                     condition: str) -> dict[str, np.ndarray]:
    path = input_dir / model / f"seed_{seed}" / f"predictions_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def verify_paired(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    for name in (
        "target", "target_mask", "sequence_id", "track_id", "frame_ids",
        "target_frame_ids", "delta_t",
    ):
        if name not in left or name not in right:
            raise ValueError(f"Prediction files lack required paired field {name!r}")
        if not np.array_equal(left[name], right[name]):
            raise ValueError(
                f"Models were not evaluated on identical ordered {name}; "
                "a paired comparison is invalid"
            )


def paired_sequence_bootstrap(input_dir: Path, wsp_results: list[dict],
                              baseline_results: list[dict], condition: str,
                              repeats: int, gap_min: float,
                              gap_max: float) -> dict[str, object]:
    wsp_by_seed = {int(result["seed"]): result for result in wsp_results}
    baseline_by_seed = {int(result["seed"]): result for result in baseline_results}
    seeds = sorted(set(wsp_by_seed).intersection(baseline_by_seed))
    if not seeds:
        raise ValueError("WSP and baseline have no common random seed")
    pairs = []
    class_indices = np.asarray(
        wsp_by_seed[seeds[0]]["evaluable_class_indices"], dtype=np.int64
    )
    reference_sequences: np.ndarray | None = None
    for seed in seeds:
        wsp = load_predictions(input_dir, "wsp", seed, condition)
        baseline_name = str(baseline_by_seed[seed]["model"])
        baseline = load_predictions(input_dir, baseline_name, seed, condition)
        verify_paired(wsp, baseline)
        if (wsp_by_seed[seed]["evaluable_class_indices"] != class_indices.tolist()
                or baseline_by_seed[seed]["evaluable_class_indices"] != class_indices.tolist()):
            raise ValueError("Models or seeds use different evaluable class sets")
        primary_mask = (
            wsp["target_mask"].astype(bool)
            & (wsp["delta_t"] >= gap_min)
            & (wsp["delta_t"] <= gap_max)
        )
        if not np.any(primary_mask):
            raise ValueError(
                f"No primary targets have delta_t in [{gap_min}, {gap_max}] seconds"
            )
        wsp["primary_mask"] = primary_mask
        baseline["primary_mask"] = primary_mask
        sequence = wsp["sequence_id"].astype(str)
        if reference_sequences is None:
            reference_sequences = sequence
        elif not np.array_equal(reference_sequences, sequence):
            raise ValueError("Prediction ordering differs across seeds")
        pairs.append((wsp, baseline))
    assert reference_sequences is not None
    unique_sequences = np.unique(reference_sequences)
    if unique_sequences.size < 2:
        raise ValueError("At least two independent sequences are required for bootstrap")

    point_differences = [
        macro_ap(wsp["probability"], wsp["target"], wsp["primary_mask"], class_indices)
        - macro_ap(
            baseline["probability"], baseline["target"],
            baseline["primary_mask"], class_indices,
        )
        for wsp, baseline in pairs
    ]
    rng = np.random.default_rng(2027)
    draws = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique_sequences, size=unique_sequences.size, replace=True)
        indexes = np.concatenate(
            [np.flatnonzero(reference_sequences == sequence) for sequence in sampled]
        )
        seed_differences = []
        for wsp, baseline in pairs:
            seed_differences.append(
                macro_ap(
                    wsp["probability"][indexes], wsp["target"][indexes],
                    wsp["primary_mask"][indexes], class_indices,
                )
                - macro_ap(
                    baseline["probability"][indexes], baseline["target"][indexes],
                    baseline["primary_mask"][indexes], class_indices,
                )
            )
        draws[repeat] = np.nanmean(seed_differences)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("Paired bootstrap produced no finite macro-AP differences")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "condition": condition,
        "primary_gap_seconds": [gap_min, gap_max],
        "evaluable_class_indices": class_indices.tolist(),
        "seeds": seeds,
        "sequence_count": int(unique_sequences.size),
        "wsp_minus_baseline_macro_ap": float(np.nanmean(point_differences)),
        "per_seed_differences": [float(value) for value in point_differences],
        "paired_sequence_bootstrap_low": float(low),
        "paired_sequence_bootstrap_high": float(high),
        "bootstrap_repeats": int(finite.size),
        "bootstrap_seed": 2027,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if not 0 <= args.primary_gap_min <= args.primary_gap_max:
        raise ValueError("primary gap bounds must be non-negative and ordered")
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap_repeats must be positive")
    if args.minimum_seeds <= 0 or args.minimum_sequences <= 0:
        raise ValueError("minimum seed and sequence counts must be positive")
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*/seed_*/result.json"))
    if not paths:
        raise FileNotFoundError(f"No */seed_*/result.json files under {args.input_dir}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fingerprints = {result.get("protocol_fingerprint") for result in results}
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError(
            "Results have missing or mixed protocol fingerprints; do not aggregate stale runs"
        )
    rows = []
    for result in results:
        for condition, metrics in result["tests"].items():
            row = {
                "model": result["model"],
                "seed": result["seed"],
                "condition": condition,
                "parameter_count": result["parameter_count"],
            }
            row.update(metrics)
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["condition", "model", "seed"])
    frame.to_csv(output_dir / "seed_metrics.csv", index=False)
    summary_rows = []
    metric_names = ["macro_ap", "macro_f1", "micro_ap", "micro_f1", "brier", "ece"]
    for (condition, model), group in frame.groupby(["condition", "model"], sort=True):
        summary: dict[str, object] = {
            "condition": condition,
            "model": model,
            "seeds": int(group["seed"].nunique()),
            "parameter_count_mean": float(group["parameter_count"].mean()),
        }
        for metric in metric_names:
            values = group[metric].dropna().astype(float).tolist()
            if not values:
                continue
            mean, low, high = confidence_interval(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_seed_ci_low"] = low
            summary[f"{metric}_seed_ci_high"] = high
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["condition", "model"])
    summary_frame.to_csv(output_dir / "summary_metrics.csv", index=False)
    primary = summary_frame[summary_frame["condition"] == args.primary_condition].copy()
    if primary.empty:
        available = sorted(summary_frame["condition"].unique().tolist())
        raise ValueError(
            f"Primary condition {args.primary_condition!r} is absent; available: {available}"
        )
    primary = primary.sort_values("macro_ap_mean", ascending=False)
    validation_means: dict[str, float] = {}
    for model in sorted({str(result["model"]) for result in results if result["model"] != "wsp"}):
        validation_means[model] = float(np.mean([
            result["validation"]["macro_ap"]
            for result in results if result["model"] == model
        ]))
    if not validation_means:
        raise ValueError("At least one non-WSP baseline is required")
    selected_baseline = max(validation_means, key=validation_means.get)
    paired = paired_sequence_bootstrap(
        args.input_dir,
        [result for result in results if result["model"] == "wsp"],
        [result for result in results if result["model"] == selected_baseline],
        args.primary_condition,
        args.bootstrap_repeats,
        args.primary_gap_min,
        args.primary_gap_max,
    )
    wsp_parameters = float(primary.loc[primary["model"] == "wsp", "parameter_count_mean"].iloc[0])
    baseline_parameters = float(
        primary.loc[primary["model"] == selected_baseline, "parameter_count_mean"].iloc[0]
    )
    paired.update({
        "baseline_selected_on_validation": selected_baseline,
        "validation_macro_ap_by_baseline": validation_means,
        "success_margin": args.success_margin,
        "parameter_count_relative_difference": abs(wsp_parameters - baseline_parameters) / wsp_parameters,
        "minimum_seeds": args.minimum_seeds,
        "minimum_sequences": args.minimum_sequences,
        "source_kind": sorted({str(result.get("source_kind", "unknown")) for result in results}),
    })
    synthetic = any("synthetic" in value for value in paired["source_kind"])
    paired["hypothesis_supported"] = bool(
        paired["wsp_minus_baseline_macro_ap"] >= args.success_margin
        and paired["paired_sequence_bootstrap_low"] > 0.0
        and paired["parameter_count_relative_difference"] <= 0.10
        and len(paired["seeds"]) >= args.minimum_seeds
        and paired["sequence_count"] >= args.minimum_sequences
        and not synthetic
    )
    paired["evidence_status"] = (
        "synthetic_smoke_test_only" if synthetic else
        "primary_criteria_met" if paired["hypothesis_supported"] else
        "primary_criteria_not_met"
    )
    (output_dir / "primary_comparison.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_lines = [
        "# JRDB-Act irregular-observation experiment summary",
        "",
        f"Primary prespecified condition: `{args.primary_condition}`.",
        "",
        "| Model | Seeds | Parameters | Macro AP | Macro F1 | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.to_dict(orient="records"):
        report_lines.append(
            "| {model} | {seeds} | {params:,.0f} | {ap:.4f} | {f1:.4f} | "
            "{brier:.4f} | {ece:.4f} |".format(
                model=row["model"], seeds=int(row["seeds"]),
                params=row["parameter_count_mean"], ap=row["macro_ap_mean"],
                f1=row["macro_f1_mean"], brier=row["brier_mean"],
                ece=row["ece_mean"],
            )
        )
    report_lines.extend([
        "",
        f"Validation-selected baseline: `{selected_baseline}`.",
        "",
        "Paired WSP-minus-baseline macro AP: "
        f"{paired['wsp_minus_baseline_macro_ap']:.4f} "
        f"(sequence-bootstrap 95% CI "
        f"[{paired['paired_sequence_bootstrap_low']:.4f}, "
        f"{paired['paired_sequence_bootstrap_high']:.4f}]).",
        "",
        "Pre-registered hypothesis result: "
        f"`{'supported' if paired['hypothesis_supported'] else 'not supported'}`.",
        f"Evidence status: `{paired['evidence_status']}`.",
        "",
        "Interpretation rule: support for the primary hypothesis requires WSP to "
        "improve macro average precision under held-out long gaps relative to the "
        "strongest baseline, with uncertainty assessed by resampling independent "
        "video sequences. Seed-level intervals in the CSV are descriptive; use the "
        "per-run sequence-bootstrap intervals for the primary inferential claim.",
        "",
    ])
    (output_dir / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(primary.to_string(index=False))
    return paired


if __name__ == "__main__":
    main()
