from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jrdb_activity_models import MODEL_NAMES


DEFAULT_MODELS = list(MODEL_NAMES)
SCHEMA_VERSION = "jrdb_activity_multigpu_v1"
RUNNER_PATH = Path(__file__).with_name("run_jrdb_activity_experiment.py")

# These options are set once by the launcher for every independent child.  A
# duplicate in --runner_args could redirect output, expose another GPU, or make
# the child write shared files, so fail before starting any process.
CONTROLLED_RUNNER_OPTIONS = frozenset(
    {
        "--data_dir",
        "--output_dir",
        "--models",
        "--seeds",
        "--device",
        "--worker_mode",
        "--worker-mode",
        "--gpus",
        "--dry_run",
        "--dry-run",
        "--runner_args",
        "--runner-args",
    }
)


@dataclass(frozen=True)
class PlannedJob:
    model: str
    seed: int
    gpu_id: str
    command: tuple[str, ...]
    log_path: Path
    result_path: Path

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "gpu_id": self.gpu_id,
            "command": list(self.command),
            "environment": {"CUDA_VISIBLE_DEVICES": self.gpu_id},
            "log_path": str(self.log_path),
            "result_path": str(self.result_path),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Schedule independent JRDB activity (model, seed) jobs across one "
            "worker queue per visible GPU."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--runner_args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Arguments forwarded verbatim to each child runner. Place this option "
            "last; data/output/model/seed/device/worker options are forbidden."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def normalize_runner_args(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    # Support the conventional `--runner_args -- --epochs ...` spelling as
    # well as `--runner_args --epochs ...`.
    if values[:1] == ["--"]:
        values = values[1:]
    return values


def validate_runner_args(arguments: Sequence[str]) -> None:
    conflicts = []
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        if option in CONTROLLED_RUNNER_OPTIONS:
            conflicts.append(option)
    if conflicts:
        raise ValueError(
            "The launcher controls these options; remove them from --runner_args: "
            + ", ".join(sorted(set(conflicts)))
        )


def validate_args(args: argparse.Namespace) -> list[str]:
    runner_args = normalize_runner_args(args.runner_args)
    validate_runner_args(runner_args)
    if not args.gpus:
        raise ValueError("At least one GPU ID is required")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU IDs must be unique so each worker owns one queue")
    for gpu_id in args.gpus:
        if not gpu_id or any(character.isspace() for character in gpu_id) or "," in gpu_id:
            raise ValueError(
                "Each GPU ID must be non-empty, contain no whitespace, and expose one device"
            )
    if len(set(args.models)) != len(args.models):
        raise ValueError("Model names must not be repeated")
    unknown = sorted(set(args.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available: {sorted(MODEL_NAMES)}")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not be repeated")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative")
    return runner_args


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "gpu"


def plan_jobs(args: argparse.Namespace, runner_args: Sequence[str]) -> list[PlannedJob]:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    jobs: list[PlannedJob] = []
    job_index = 0
    for model in args.models:
        for seed in args.seeds:
            gpu_id = str(args.gpus[job_index % len(args.gpus)])
            command = (
                sys.executable,
                str(RUNNER_PATH),
                "--data_dir",
                str(data_dir),
                "--output_dir",
                str(output_dir),
                "--models",
                model,
                "--seeds",
                str(seed),
                "--device",
                "cuda",
                "--worker_mode",
                *runner_args,
            )
            jobs.append(
                PlannedJob(
                    model=model,
                    seed=int(seed),
                    gpu_id=gpu_id,
                    command=command,
                    log_path=(
                        output_dir
                        / "logs"
                        / f"{model}__seed_{seed}__gpu_{_safe_name(gpu_id)}.log"
                    ),
                    result_path=output_dir / model / f"seed_{seed}" / "result.json",
                )
            )
            job_index += 1
    return jobs


def _display_command(job: PlannedJob) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={job.gpu_id} "
        + subprocess.list2cmdline(list(job.command))
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_job(job: PlannedJob) -> int:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = job.gpu_id
    started = datetime.now(timezone.utc).isoformat()
    with job.log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"started_utc={started}\n")
        log.write(f"gpu_id={job.gpu_id}\n")
        log.write(f"command={_display_command(job)}\n")
        log.write("child_output_begin\n")
        try:
            completed = subprocess.run(
                list(job.command),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return_code = int(completed.returncode)
        except Exception as error:  # Preserve a per-job diagnostic before failing.
            log.write(f"launcher_exception={type(error).__name__}: {error}\n")
            return_code = -1
        log.write("child_output_end\n")
        log.write(f"return_code={return_code}\n")
        log.write(f"finished_utc={datetime.now(timezone.utc).isoformat()}\n")
    return return_code


def execute_jobs(jobs: Sequence[PlannedJob], gpu_ids: Sequence[str]) -> dict[tuple[str, int], int]:
    queues = {gpu_id: queue.Queue() for gpu_id in gpu_ids}
    for job in jobs:
        queues[job.gpu_id].put(job)

    return_codes: dict[tuple[str, int], int] = {}
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        work = queues[gpu_id]
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            try:
                print(
                    f"starting model={job.model} seed={job.seed} gpu={gpu_id} "
                    f"log={job.log_path}",
                    flush=True,
                )
                return_code = _run_job(job)
                with lock:
                    return_codes[(job.model, job.seed)] = return_code
                print(
                    f"finished model={job.model} seed={job.seed} gpu={gpu_id} "
                    f"return_code={return_code}",
                    flush=True,
                )
            finally:
                work.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu_id,),
            name=f"jrdb-gpu-{_safe_name(gpu_id)}",
            daemon=False,
        )
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return return_codes


def _base_manifest(
    args: argparse.Namespace,
    runner_args: Sequence[str],
    jobs: Sequence[PlannedJob],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "runner_path": str(RUNNER_PATH),
        "python": sys.executable,
        "gpus": [str(value) for value in args.gpus],
        "models": list(args.models),
        "seeds": [int(value) for value in args.seeds],
        "runner_args": list(runner_args),
        "jobs": [job.manifest_entry() for job in jobs],
    }


def consolidate_results(
    jobs: Sequence[PlannedJob],
) -> tuple[list[dict[str, Any]], str]:
    results: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for job in jobs:
        if not job.result_path.is_file():
            raise FileNotFoundError(
                f"Successful child did not write expected result: {job.result_path}"
            )
        result = json.loads(job.result_path.read_text(encoding="utf-8"))
        if result.get("model") != job.model or int(result.get("seed", -1)) != job.seed:
            raise ValueError(
                f"Result identity mismatch in {job.result_path}: expected "
                f"{job.model}/seed_{job.seed}"
            )
        fingerprint = result.get("protocol_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"Result lacks protocol_fingerprint: {job.result_path}")
        fingerprints.add(fingerprint)
        results.append(result)
    if len(fingerprints) != 1:
        raise ValueError(
            "Child results have different protocol fingerprints: "
            + ", ".join(sorted(fingerprints))
        )
    return results, next(iter(fingerprints))


def main(argv: Sequence[str] | None = None) -> list[Any]:
    args = parse_args(argv)
    runner_args = validate_args(args)
    jobs = plan_jobs(args, runner_args)
    if args.dry_run:
        for job in jobs:
            print(_display_command(job))
        return list(jobs)

    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    if not RUNNER_PATH.is_file():
        raise FileNotFoundError(RUNNER_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_codes = execute_jobs(jobs, [str(value) for value in args.gpus])
    failures = [
        job for job in jobs if return_codes.get((job.model, job.seed), -1) != 0
    ]
    manifest = _base_manifest(args, runner_args, jobs)
    manifest["return_codes"] = {
        f"{model}/seed_{seed}": code
        for (model, seed), code in sorted(return_codes.items())
    }
    if failures:
        manifest["status"] = "failed"
        manifest["failed_jobs"] = [job.manifest_entry() for job in failures]
        _atomic_json(args.output_dir / "multigpu_manifest.json", manifest)
        raise RuntimeError(
            "JRDB activity child jobs failed: "
            + ", ".join(
                f"{job.model}/seed_{job.seed} (log: {job.log_path})"
                for job in failures
            )
        )

    try:
        results, fingerprint = consolidate_results(jobs)
    except Exception as error:
        manifest["status"] = "failed_consolidation"
        manifest["consolidation_error"] = f"{type(error).__name__}: {error}"
        _atomic_json(args.output_dir / "multigpu_manifest.json", manifest)
        raise
    _atomic_json(args.output_dir / "all_results.json", results)
    manifest["status"] = "complete"
    manifest["protocol_fingerprint"] = fingerprint
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(args.output_dir / "multigpu_manifest.json", manifest)
    return results


if __name__ == "__main__":
    main()


__all__ = [
    "CONTROLLED_RUNNER_OPTIONS",
    "DEFAULT_MODELS",
    "PlannedJob",
    "consolidate_results",
    "execute_jobs",
    "main",
    "normalize_runner_args",
    "parse_args",
    "plan_jobs",
    "validate_args",
    "validate_runner_args",
]
