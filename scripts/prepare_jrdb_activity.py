from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.jrdb_activity_data import (  # noqa: E402
    ActivityPreparationConfig,
    generate_synthetic_canonical_csv,
    load_canonical_activity_csv,
    prepare_jrdb_activity_dataset,
)


CANONICAL_HELP = """
Canonical input contract
------------------------
This command intentionally does not parse raw JRDB/JRDB-Act JSON or perform
2D/3D IoU matching. Convert raw annotations to one long-form CSV before using
this script. Every row must contain:

  sequence_id, frame_id, track_id, timestamp, x, y, z, action_id

Use one row per action label. Repeat sequence/frame/track rows to represent a
multilabel activity state; all repeated rows must have identical state fields.
Optional numeric columns are length, width, height, heading (radians by
default), and occlusion. Aliases box_length/box_width/box_height, yaw, and
occluded are accepted. Coordinates and timestamps must share consistent
physical units throughout the file. Prefer the annotated JRDB-Act keyframes
and stable official 3D track identifiers. Do not include action labels or
ground-truth group identifiers among input features.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare target-centered 3D relational activity windows with "
            "controlled irregular observation schedules."
        ),
        epilog=CANONICAL_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        help=(
            "Canonical long-form CSV. Raw JRDB JSON/IoU conversion is not "
            "implemented; see the contract below."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--split_json",
        type=Path,
        help=(
            "Required for real JRDB data; sequence-disjoint split map. Accepts either "
            "{sequence_id: train|val|test} or {train: [...], val: [...], test: [...]}."
        ),
    )
    parser.add_argument("--k_neighbors", type=int, default=4)
    parser.add_argument("--window_length", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--train_seen_steps",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Observation-index gaps used to retain train/validation observations.",
    )
    parser.add_argument(
        "--unseen_gap_steps",
        nargs="+",
        type=int,
        default=[4, 5, 6],
        help="Disjoint observation-index gaps used only in test_unseen_gap.npz.",
    )
    parser.add_argument("--burst_min", type=int, default=2)
    parser.add_argument("--burst_max", type=int, default=4)
    parser.add_argument("--train_gap_min_seconds", type=float, default=1.0)
    parser.add_argument("--train_gap_max_seconds", type=float, default=3.0)
    parser.add_argument("--unseen_gap_min_seconds", type=float, default=4.0)
    parser.add_argument("--unseen_gap_max_seconds", type=float, default=6.0)
    parser.add_argument(
        "--include_partial_windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad final short windows and mark padding through valid_mask.",
    )
    parser.add_argument(
        "--heading_degrees",
        action="store_true",
        help="Interpret the optional heading/yaw column in degrees rather than radians.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the seven prepared NPZ files and metadata.json if present.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Generate a deterministic canonical CSV and prepare it. This is only "
            "for smoke tests and is not experimental evidence."
        ),
    )
    parser.add_argument("--synthetic_sequences", type=int, default=6)
    parser.add_argument("--synthetic_frames", type=int, default=24)
    parser.add_argument("--synthetic_tracks", type=int, default=4)
    parser.add_argument(
        "--expected_classes", type=int, default=0,
        help="Fail unless this many distinct action labels are present; use 26 for JRDB-Act.",
    )
    parser.add_argument(
        "--expected_interval_seconds", type=float, default=0.0,
        help=(
            "If positive, require every within-track timestamp gap to be an integer "
            "multiple of this base interval; use 1 for official action keyframes."
        ),
    )
    parser.add_argument("--interval_tolerance", type=float, default=1e-5)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.synthetic and args.input_csv is not None:
        raise ValueError("Use either --synthetic or --input_csv, not both")
    if not args.synthetic and args.input_csv is None:
        raise ValueError("--input_csv is required unless --synthetic is used")
    if not args.synthetic and args.split_json is None:
        raise ValueError(
            "--split_json is required for real data; random sequence assignment is "
            "allowed only in --synthetic smoke tests"
        )
    if args.input_csv is not None and not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if args.split_json is not None and not args.split_json.is_file():
        raise FileNotFoundError(args.split_json)
    if args.synthetic_sequences < 3 or args.synthetic_frames < 2 or args.synthetic_tracks < 1:
        raise ValueError(
            "synthetic settings require sequences>=3, frames>=2, and tracks>=1"
        )
    if args.expected_classes < 0:
        raise ValueError("--expected_classes must be non-negative")
    if args.expected_interval_seconds < 0 or args.interval_tolerance <= 0:
        raise ValueError("Expected interval must be non-negative and tolerance positive")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_csv = args.input_csv
    if args.synthetic:
        input_csv = args.output_dir / "synthetic_canonical.csv"
        if input_csv.exists() and not args.overwrite:
            raise FileExistsError(
                f"{input_csv} already exists; pass --overwrite or choose a new output directory"
            )
        generate_synthetic_canonical_csv(
            input_csv,
            seed=args.seed,
            sequences=args.synthetic_sequences,
            frames=args.synthetic_frames,
            tracks=args.synthetic_tracks,
        )
    assert input_csv is not None

    if args.expected_interval_seconds > 0 or args.expected_classes:
        entities, labels, _ = load_canonical_activity_csv(
            input_csv, heading_degrees=args.heading_degrees
        )
        if args.expected_classes and len(labels) != args.expected_classes:
            raise ValueError(
                f"Expected {args.expected_classes} classes, found {len(labels)}"
            )
    if args.expected_interval_seconds > 0:
        differences = (
            entities.groupby(["sequence_id", "track_id"], sort=False)["timestamp"]
            .diff()
            .dropna()
            .to_numpy()
        )
        ratios = differences / args.expected_interval_seconds
        if differences.size and not np.allclose(
            ratios, np.round(ratios),
            atol=args.interval_tolerance, rtol=0.0,
        ):
            raise ValueError(
                "Within-track timestamp gaps are not integer multiples of "
                f"{args.expected_interval_seconds} seconds"
            )

    config = ActivityPreparationConfig(
        k_neighbors=args.k_neighbors,
        window_length=args.window_length,
        window_stride=args.window_stride,
        seed=args.seed,
        train_seen_steps=tuple(args.train_seen_steps),
        unseen_gap_steps=tuple(args.unseen_gap_steps),
        train_gap_min_seconds=args.train_gap_min_seconds,
        train_gap_max_seconds=args.train_gap_max_seconds,
        unseen_gap_min_seconds=args.unseen_gap_min_seconds,
        unseen_gap_max_seconds=args.unseen_gap_max_seconds,
        burst_min=args.burst_min,
        burst_max=args.burst_max,
        include_partial_windows=args.include_partial_windows,
        heading_degrees=args.heading_degrees,
    )
    metadata = prepare_jrdb_activity_dataset(
        input_csv,
        args.output_dir,
        config=config,
        split_json=args.split_json,
        overwrite=args.overwrite,
        source_kind="synthetic_smoke" if args.synthetic else "jrdb_act_canonical",
    )
    print(f"Prepared activity data in {args.output_dir.resolve()}")
    for name, details in metadata["files"].items():
        print(
            f"  {name}: {details['windows']} windows, "
            f"{details['valid_transitions']} valid transitions"
        )
    return metadata


if __name__ == "__main__":
    main()
