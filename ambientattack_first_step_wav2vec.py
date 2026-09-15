#!/usr/bin/env python3
"""Wav2Vec-only stage-1 environment screening for one speech sample.

The script fixes one speech sample and one hard-text ASR model, samples 50
environment sounds without replacement, and evaluates neutral environment
transforms (pitch=0, rate=1, tilt=0) over an SNR grid. For each environment it
locally refines the highest SNR that reaches the primary CER threshold. Stage 1
uses direct digital mixing and deliberately does not apply an RIR.

Primary candidates require CER >= 0.20. If fewer than five primary candidates
exist, candidates with 0.12 <= CER < 0.20 fill the remaining places. Candidates
below the fallback threshold are reported but are never selected.

Examples:

    python ambientattack_first_step_wav2vec.py --speech sample.wav
    python ambientattack_first_step_wav2vec.py --sample-count 2 --top-k 1 \
        --snr-min 0 --snr-max 10 --force-cpu
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from jiwer import cer, wer

if __package__:
    from . import asr_attack_test as attack_core
else:
    import asr_attack_test as attack_core


DEFAULT_OUTPUT_DIR = Path("ambientattack_first_step_test_results")
DEFAULT_SAMPLE_COUNT = 50
DEFAULT_TOP_K = 5
DEFAULT_RANDOM_SEED = secrets.randbelow(2**32)
DEFAULT_PRIMARY_MIN_CER = 0.20
DEFAULT_FALLBACK_MIN_CER = 0.12
DEFAULT_SNR_GRID_STEP_DB = 5.0
DEFAULT_SNR_TOLERANCE_DB = 0.25
ENVIRONMENT_EXTENSIONS = {".wav", ".flac", ".ogg"}
TARGET_ASR = "wav2vec"

RESULT_FIELDS = (
    "random_order",
    "overall_rank",
    "selection_rank",
    "selected",
    "environment",
    "environment_name",
    "copied_environment",
    "status",
    "threshold_used",
    "target_snr_db",
    "actual_snr_db",
    "attack_cer",
    "attack_wer",
    "max_observed_cer",
    "asr_queries",
    "transcript_raw",
    "transcript_normalized",
    "error",
)


@dataclass(frozen=True)
class SNRParameters:
    snr_db: float


@dataclass(frozen=True)
class SNREvaluation:
    query: int
    phase: str
    parameters: SNRParameters
    transcript_raw: str
    transcript_normalized: str
    attack_cer: float
    attack_wer: float
    actual_snr_db: float


@dataclass
class ScreeningResult:
    """Best threshold-boundary result for one environment sound."""

    random_order: int
    environment_path: Path
    status: str
    threshold_used: Optional[float]
    evaluation: Optional[SNREvaluation]
    max_observed_cer: float
    query_count: int
    error: str = ""

    @property
    def eligible(self) -> bool:
        return self.status in ("primary", "fallback")

    def rank_key(self) -> Tuple[Any, ...]:
        """Primary first, fallback second, then highest SNR only."""
        group_order = {
            "primary": 0,
            "fallback": 1,
            "failed": 2,
            "error": 3,
        }[self.status]
        if self.evaluation is None:
            return (
                group_order,
                str(self.environment_path),
            )
        if self.eligible:
            return (
                group_order,
                -self.evaluation.actual_snr_db,
                str(self.environment_path),
            )
        return (
            group_order,
            -self.max_observed_cer,
            str(self.environment_path),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample environment sounds and select the best candidates "
            "using the highest successful SNR and a CER fallback threshold."
        )
    )
    parser.add_argument(
        "--speech",
        type=Path,
        required=True,
        help="One speech sample.",
    )
    parser.add_argument(
        "--environment-pool",
        type=Path,
        required=True,
        help="Directory searched recursively for environment audio.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of environment sounds sampled without replacement. Default: 50.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of environments selected for stage 2. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Random environment-sampling seed. By default, a new random "
            "32-bit seed is generated for each run."
        ),
    )
    parser.add_argument(
        "--primary-min-cer",
        type=float,
        default=DEFAULT_PRIMARY_MIN_CER,
        help="Preferred stage-1 CER threshold. Default: 0.20.",
    )
    parser.add_argument(
        "--fallback-min-cer",
        type=float,
        default=DEFAULT_FALLBACK_MIN_CER,
        help=(
            "Fallback CER threshold used only when fewer than top-k primary "
            "candidates exist. Default: 0.12."
        ),
    )
    parser.add_argument("--snr-min", type=float, default=-10.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument(
        "--snr-grid-step",
        type=float,
        default=DEFAULT_SNR_GRID_STEP_DB,
        help=(
            "Global descending SNR-grid spacing in dB before local binary "
            "refinement. Default: 5.0."
        ),
    )
    parser.add_argument(
        "--snr-tolerance",
        type=float,
        default=DEFAULT_SNR_TOLERANCE_DB,
        help="Local SNR boundary-refinement tolerance in dB. Default: 0.25.",
    )

    parser.add_argument(
        "--asr",
        choices=(TARGET_ASR,),
        default=TARGET_ASR,
        help="Fixed target model. Default and only choice: wav2vec.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force PyTorch-based ASR models to run on CPU.",
    )
    parser.add_argument(
        "--wav2vec-model",
        type=Path,
        required=True,
        help="Path to the fairseq Wav2Vec CTC checkpoint.",
    )
    parser.add_argument(
        "--wav2vec-dictionary",
        type=Path,
        required=True,
        help="Path to the fairseq dict.ltr.txt file.",
    )
    parser.add_argument(
        "--wav2vec-chunk-seconds",
        type=float,
        default=attack_core.WAV2VEC_CHUNK_SECONDS,
    )
    parser.add_argument(
        "--wav2vec-stride-seconds",
        type=float,
        default=attack_core.WAV2VEC_STRIDE_SECONDS,
    )
    args = parser.parse_args()
    return args


def find_environment_files(directory: Path) -> List[Path]:
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in ENVIRONMENT_EXTENSIONS
    )


def validate_args(args: argparse.Namespace) -> List[Path]:
    if not args.speech.is_file():
        raise FileNotFoundError(f"Speech file does not exist: {args.speech}")
    if not args.environment_pool.is_dir():
        raise NotADirectoryError(
            f"Environment pool does not exist: {args.environment_pool}"
        )
    if args.sample_count < 1:
        raise ValueError("--sample-count must be at least 1")
    if args.top_k < 1 or args.top_k > args.sample_count:
        raise ValueError("--top-k must be in [1, --sample-count]")
    if not (
        math.isfinite(args.fallback_min_cer)
        and math.isfinite(args.primary_min_cer)
        and 0.0 <= args.fallback_min_cer < args.primary_min_cer
    ):
        raise ValueError(
            "CER thresholds must satisfy 0 <= fallback < primary"
        )
    for label, value in (
        ("--snr-min", args.snr_min),
        ("--snr-max", args.snr_max),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if args.snr_min > args.snr_max:
        raise ValueError("--snr-min must not exceed --snr-max")
    if not math.isfinite(args.snr_grid_step) or args.snr_grid_step <= 0.0:
        raise ValueError("--snr-grid-step must be finite and positive")
    if not math.isfinite(args.snr_tolerance) or args.snr_tolerance <= 0.0:
        raise ValueError("--snr-tolerance must be finite and positive")
    if (
        args.wav2vec_chunk_seconds < 0.0
        or args.wav2vec_stride_seconds < 0.0
    ):
        raise ValueError("wav2vec chunk and stride must be non-negative")
    if (
        args.wav2vec_chunk_seconds > 0.0
        and 2.0 * args.wav2vec_stride_seconds
        >= args.wav2vec_chunk_seconds
    ):
        raise ValueError("wav2vec stride must be less than half of chunk length")
    environment_files = find_environment_files(args.environment_pool)
    if len(environment_files) < args.sample_count:
        raise RuntimeError(
            f"Environment pool has {len(environment_files)} audio files, but "
            f"--sample-count is {args.sample_count}"
        )

    for path in (args.wav2vec_model, args.wav2vec_dictionary):
        if not path.is_file():
            raise FileNotFoundError(f"Wav2Vec file does not exist: {path}")
    if args.wav2vec_dictionary.name != "dict.ltr.txt":
        raise ValueError("--wav2vec-dictionary must be named dict.ltr.txt")
    return environment_files


def build_snr_grid(minimum: float, maximum: float, step: float) -> List[float]:
    """Return deterministic grid points from high SNR to low SNR."""
    values: List[float] = []
    value = float(maximum)
    while value >= minimum - 1e-10:
        values.append(round(max(value, minimum), 10))
        value -= step
    if not values or values[-1] > minimum + 1e-10:
        values.append(round(float(minimum), 10))
    return list(dict.fromkeys(values))


class SNRSearch:
    """Evaluate SNR candidates without auxiliary quality objectives."""

    def __init__(
        self,
        speech: np.ndarray,
        environment: np.ndarray,
        asr: Any,
        clean_text_raw: str,
    ) -> None:
        self.speech = np.asarray(speech, dtype=np.float32)
        loop = attack_core.make_crossfaded_environment_loop(environment)
        self.environment = attack_core.fit_to_length_with_sample_offset(
            loop,
            self.speech.size,
            0,
        )
        if attack_core.rms(self.environment) <= 1e-8:
            raise ValueError("Environment sound is silent after length matching")
        self.asr = asr
        self.clean_text_normalized = attack_core.normalize_text(clean_text_raw)
        self.query_count = 0
        self.cache: Dict[float, SNREvaluation] = {}

    def evaluate(self, snr_db: float, phase: str) -> SNREvaluation:
        key = round(float(snr_db), 10)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        environment_scale = attack_core.rms(self.speech) / (
            attack_core.rms(self.environment)
            * float(10.0 ** (key / 20.0))
        )
        environment_component = self.environment * environment_scale
        mixture = np.asarray(
            self.speech + environment_component,
            dtype=np.float32,
        )
        transcript_raw = self.asr.transcribe(mixture)
        transcript_normalized = attack_core.normalize_text(transcript_raw)
        self.query_count += 1
        evaluation = SNREvaluation(
            query=self.query_count,
            phase=phase,
            parameters=SNRParameters(snr_db=key),
            transcript_raw=transcript_raw,
            transcript_normalized=transcript_normalized,
            attack_cer=float(
                cer(self.clean_text_normalized, transcript_normalized)
            ),
            attack_wer=float(
                wer(self.clean_text_normalized, transcript_normalized)
            ),
            actual_snr_db=float(
                attack_core.calculate_snr_db(
                    self.speech,
                    environment_component,
                )
            ),
        )
        self.cache[key] = evaluation
        return evaluation


def evaluate_snr(
    attack: SNRSearch,
    snr_db: float,
    phase: str,
) -> SNREvaluation:
    return attack.evaluate(float(snr_db), phase)


def refine_highest_successful_snr(
    attack: SNRSearch,
    successful: SNREvaluation,
    higher_failure: SNREvaluation,
    threshold: float,
    tolerance: float,
    evaluations: List[SNREvaluation],
) -> SNREvaluation:
    """Bisect a local failure/success bracket in target-SNR space."""
    low_success = float(successful.parameters.snr_db)
    high_failure = float(higher_failure.parameters.snr_db)
    best = successful

    while high_failure - low_success > tolerance:
        midpoint = 0.5 * (low_success + high_failure)
        evaluation = evaluate_snr(attack, midpoint, "firststep_snr_refine")
        evaluations.append(evaluation)
        if evaluation.attack_cer >= threshold:
            low_success = midpoint
            best = evaluation
        else:
            high_failure = midpoint
    return best


def highest_threshold_evaluation(
    attack: SNRSearch,
    grid_evaluations: Sequence[SNREvaluation],
    threshold: float,
    tolerance: float,
    all_evaluations: List[SNREvaluation],
) -> Optional[SNREvaluation]:
    """Find the highest successful global SNR and locally refine its boundary."""
    successful_indices = [
        index
        for index, evaluation in enumerate(grid_evaluations)
        if evaluation.attack_cer >= threshold
    ]
    if not successful_indices:
        return None
    index = successful_indices[0]
    successful = grid_evaluations[index]
    if index == 0:
        return successful
    higher_failure = grid_evaluations[index - 1]
    return refine_highest_successful_snr(
        attack,
        successful,
        higher_failure,
        threshold,
        tolerance,
        all_evaluations,
    )


def screen_environment(
    random_order: int,
    environment_path: Path,
    speech: np.ndarray,
    asr: Any,
    clean_text_raw: str,
    snr_grid: Sequence[float],
    args: argparse.Namespace,
) -> ScreeningResult:
    attack: Optional[SNRSearch] = None
    try:
        environment = attack_core.load_audio(environment_path)
        attack = SNRSearch(
            speech,
            environment,
            asr,
            clean_text_raw,
        )
        evaluations = [
            evaluate_snr(attack, snr_db, "firststep_snr_grid")
            for snr_db in snr_grid
        ]
        grid_evaluations = list(evaluations)

        selected = highest_threshold_evaluation(
            attack,
            grid_evaluations,
            args.primary_min_cer,
            args.snr_tolerance,
            evaluations,
        )
        if selected is not None:
            status = "primary"
            threshold_used: Optional[float] = args.primary_min_cer
        else:
            selected = highest_threshold_evaluation(
                attack,
                grid_evaluations,
                args.fallback_min_cer,
                args.snr_tolerance,
                evaluations,
            )
            if selected is not None:
                status = "fallback"
                threshold_used = args.fallback_min_cer
            else:
                status = "failed"
                threshold_used = None
                selected = max(
                    evaluations,
                    key=lambda item: (
                        item.attack_cer,
                        item.actual_snr_db,
                    ),
                )

        max_observed_cer = max(item.attack_cer for item in evaluations)
        return ScreeningResult(
            random_order=random_order,
            environment_path=environment_path,
            status=status,
            threshold_used=threshold_used,
            evaluation=selected,
            max_observed_cer=float(max_observed_cer),
            query_count=attack.query_count,
        )
    except Exception as exc:
        return ScreeningResult(
            random_order=random_order,
            environment_path=environment_path,
            status="error",
            threshold_used=None,
            evaluation=None,
            max_observed_cer=0.0,
            query_count=0 if attack is None else attack.query_count,
            error=f"{type(exc).__name__}: {exc}",
        )


def result_row(
    result: ScreeningResult,
    overall_rank: Optional[int] = None,
    selection_rank: Optional[int] = None,
    copied_environment: Optional[Path] = None,
) -> Dict[str, Any]:
    evaluation = result.evaluation
    return {
        "random_order": result.random_order,
        "overall_rank": "" if overall_rank is None else overall_rank,
        "selection_rank": "" if selection_rank is None else selection_rank,
        "selected": selection_rank is not None,
        "environment": str(result.environment_path),
        "environment_name": result.environment_path.name,
        "copied_environment": (
            "" if copied_environment is None else str(copied_environment)
        ),
        "status": result.status,
        "threshold_used": (
            "" if result.threshold_used is None else f"{result.threshold_used:.8f}"
        ),
        "target_snr_db": (
            "" if evaluation is None else f"{evaluation.parameters.snr_db:.8f}"
        ),
        "actual_snr_db": (
            "" if evaluation is None else f"{evaluation.actual_snr_db:.8f}"
        ),
        "attack_cer": (
            "" if evaluation is None else f"{evaluation.attack_cer:.10f}"
        ),
        "attack_wer": (
            "" if evaluation is None else f"{evaluation.attack_wer:.10f}"
        ),
        "max_observed_cer": f"{result.max_observed_cer:.10f}",
        "asr_queries": result.query_count,
        "transcript_raw": "" if evaluation is None else evaluation.transcript_raw,
        "transcript_normalized": (
            "" if evaluation is None else evaluation.transcript_normalized
        ),
        "error": result.error,
    }


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.speech = args.speech.expanduser().resolve()
    args.environment_pool = args.environment_pool.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    environment_files = validate_args(args)

    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(
        len(environment_files),
        size=args.sample_count,
        replace=False,
    )
    sampled_environments = [environment_files[int(index)] for index in selected_indices]
    snr_grid = build_snr_grid(args.snr_min, args.snr_max, args.snr_grid_step)
    run_dir = (
        args.output_dir
        / f"{args.speech.stem}__{args.asr}__seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "screening_progress.csv"

    speech = attack_core.load_audio(args.speech)

    print("=" * 88)
    print(f"Speech             : {args.speech}")
    print(f"ASR                : {args.asr}")
    print(f"Environment pool   : {args.environment_pool}")
    print(f"Pool size          : {len(environment_files)}")
    print(f"Random sample      : {args.sample_count} (seed={args.seed})")
    print(f"Top K              : {args.top_k}")
    print(f"Primary CER        : {args.primary_min_cer:.4f}")
    print(f"Fallback CER       : {args.fallback_min_cer:.4f}")
    print(f"SNR grid           : {snr_grid}")
    print(f"SNR tolerance      : {args.snr_tolerance:.4f} dB")
    print(f"Output             : {run_dir}")
    print("=" * 88)

    asr = None
    results: List[ScreeningResult] = []
    total_queries = 0
    try:
        asr = attack_core.build_asr(args)
        clean_text_raw = asr.transcribe(speech)
        clean_text_normalized = attack_core.normalize_text(clean_text_raw)
        if not clean_text_normalized:
            raise RuntimeError("ASR returned an empty clean transcript")
        total_queries = 1
        print(f"Dry clean transcript: {clean_text_raw}")

        with progress_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for order, environment_path in enumerate(
                sampled_environments,
                start=1,
            ):
                result = screen_environment(
                    order,
                    environment_path,
                    speech,
                    asr,
                    clean_text_raw,
                    snr_grid,
                    args,
                )
                results.append(result)
                total_queries += result.query_count
                writer.writerow(result_row(result))
                handle.flush()

                evaluation = result.evaluation
                if evaluation is None:
                    metric_text = f"ERROR={result.error}"
                else:
                    metric_text = (
                        f"target_SNR={evaluation.parameters.snr_db:.3f} "
                        f"actual_SNR={evaluation.actual_snr_db:.3f} "
                        f"CER={evaluation.attack_cer:.4f}"
                    )
                print(
                    f"[{order:2d}/{len(sampled_environments)}] "
                    f"{result.status:8s} {environment_path.name} {metric_text}"
                )
    finally:
        gc.collect()
        if asr is not None:
            asr.close()

    ranked_results = sorted(results, key=lambda item: item.rank_key())
    eligible_results = [result for result in ranked_results if result.eligible]
    selected_results = eligible_results[: args.top_k]
    selected_ranks = {
        str(result.environment_path): rank
        for rank, result in enumerate(selected_results, start=1)
    }

    selected_audio_dir = run_dir / "selected_environment_audio"
    selected_audio_dir.mkdir(parents=True, exist_ok=True)
    copied_environment_paths: Dict[str, Path] = {}
    for rank, result in enumerate(selected_results, start=1):
        copied_path = (
            selected_audio_dir
            / f"rank_{rank:02d}__{result.environment_path.name}"
        )
        shutil.copy2(result.environment_path, copied_path)
        copied_environment_paths[str(result.environment_path)] = copied_path

    ranked_rows = []
    for overall_rank, result in enumerate(ranked_results, start=1):
        ranked_rows.append(
            result_row(
                result,
                overall_rank=overall_rank,
                selection_rank=selected_ranks.get(str(result.environment_path)),
                copied_environment=copied_environment_paths.get(
                    str(result.environment_path)
                ),
            )
        )
    selected_rows = [
        result_row(
            result,
            overall_rank=rank,
            selection_rank=rank,
            copied_environment=copied_environment_paths[
                str(result.environment_path)
            ],
        )
        for rank, result in enumerate(selected_results, start=1)
    ]

    ranked_path = run_dir / "all_environment_rankings.csv"
    selected_path = run_dir / f"selected_top{args.top_k}.csv"
    no_primary_path = run_dir / "no_primary_environments.csv"
    no_primary_or_fallback_path = (
        run_dir / "no_primary_or_fallback_environments.csv"
    )
    path_list = run_dir / "selected_environment_paths.txt"
    summary_path = run_dir / "selection_summary.json"
    write_rows(ranked_path, ranked_rows)
    write_rows(selected_path, selected_rows)
    write_rows(
        no_primary_path,
        [
            row
            for row in ranked_rows
            if row["status"] in ("fallback", "failed")
        ],
    )
    write_rows(
        no_primary_or_fallback_path,
        [row for row in ranked_rows if row["status"] == "failed"],
    )
    path_list.write_text(
        "".join(
            f"{copied_environment_paths[str(result.environment_path)]}\n"
            for result in selected_results
        ),
        encoding="utf-8",
    )

    status_counts = {
        status: sum(result.status == status for result in results)
        for status in ("primary", "fallback", "failed", "error")
    }
    summary = {
        "speech": str(args.speech),
        "asr": args.asr,
        "clean_transcript_audio_domain": "dry_original_speech",
        "clean_transcript_raw": clean_text_raw,
        "clean_transcript_normalized": clean_text_normalized,
        "environment_pool": str(args.environment_pool),
        "pool_size": len(environment_files),
        "sample_count": args.sample_count,
        "top_k_requested": args.top_k,
        "top_k_selected": len(selected_results),
        "random_seed": args.seed,
        "primary_min_cer": args.primary_min_cer,
        "fallback_min_cer": args.fallback_min_cer,
        "snr_min_db": args.snr_min,
        "snr_max_db": args.snr_max,
        "snr_grid_step_db": args.snr_grid_step,
        "snr_tolerance_db": args.snr_tolerance,
        "snr_grid_db": snr_grid,
        "selection_rule": (
            "primary before fallback; highest actual SNR within each status"
        ),
        "status_counts": status_counts,
        "no_primary_csv": str(no_primary_path),
        "no_primary_or_fallback_csv": str(no_primary_or_fallback_path),
        "total_asr_queries_including_clean": total_queries,
        "selected_environments": [
            {
                "rank": rank,
                "path": str(result.environment_path),
                "copied_path": str(
                    copied_environment_paths[str(result.environment_path)]
                ),
                "status": result.status,
                "threshold_used": result.threshold_used,
                "target_snr_db": result.evaluation.parameters.snr_db,
                "actual_snr_db": result.evaluation.actual_snr_db,
                "attack_cer": result.evaluation.attack_cer,
                "attack_wer": result.evaluation.attack_wer,
            }
            for rank, result in enumerate(selected_results, start=1)
            if result.evaluation is not None
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print(f"Primary candidates : {status_counts['primary']}")
    print(f"Fallback candidates: {status_counts['fallback']}")
    print(f"Failed candidates  : {status_counts['failed']}")
    print(f"Errors             : {status_counts['error']}")
    print(f"Selected           : {len(selected_results)}/{args.top_k}")
    print(f"Total ASR queries  : {total_queries}")
    for rank, result in enumerate(selected_results, start=1):
        assert result.evaluation is not None
        print(
            f"  #{rank} [{result.status}] {result.environment_path} "
            f"SNR={result.evaluation.actual_snr_db:.3f} dB "
            f"CER={result.evaluation.attack_cer:.4f}"
        )
    if len(selected_results) < args.top_k:
        print(
            "Warning: fewer eligible primary/fallback candidates were found; "
            "CER values below the fallback threshold were not selected."
        )
    print(f"Progress CSV       : {progress_path}")
    print(f"All rankings CSV   : {ranked_path}")
    print(f"Selected CSV       : {selected_path}")
    print(f"No primary CSV     : {no_primary_path}")
    print(f"No primary/fallback: {no_primary_or_fallback_path}")
    print(f"Selected audio     : {selected_audio_dir}")
    print(f"Selected paths     : {path_list}")
    print(f"Summary JSON       : {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
