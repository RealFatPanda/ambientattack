#!/usr/bin/env python3
"""Standalone Wav2Vec AmbientAttack second-stage test for one sample pair.

Only the environmental signal is optimized. Speech and environment playback
travel through separate fixed acoustic channels, their received signals are
then added, and the common microphone response is applied. ASR is evaluated on
both the digital mixture before the RIRs and the receiver-side mixture; both
must satisfy the CER threshold. Successful candidates are ranked by
receiver-side NISQA MOS.

This file contains the attack implementation and the fairseq Wav2Vec CTC
wrapper directly; it does not import other project Python modules.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import re
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, replace
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from jiwer import cer, wer
from scipy.signal import butter, fftconvolve, istft, resample_poly, sosfilt, stft


TARGET_ASR = "wav2vec"
TARGET_SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg"}

DEFAULT_OUTPUT_DIR = Path("ambientattack_seconde_step_test_results")
DEFAULT_DATA_TEST_ROOT = Path("data_test")

WAV2VEC_CHUNK_SECONDS = 20.0
WAV2VEC_STRIDE_SECONDS = 4.0
DEFAULT_SNR_SEARCH_MIN_CER = 0.10

EXPLICIT_NUMBER_RULES = (
    (r"\b0?7\s*[:.]\s*15\b", "seven fifteen"),
    (r"\b715\b", "seven fifteen"),
    (r"\b0?3\s*[:.]\s*30\b", "three thirty"),
    (r"\b330\b", "three thirty"),
    (r"\b50\s*%", "fifty percent"),
    (r"\b125\b", "one hundred twenty five"),
    (r"\b68\b", "sixty eight"),
    (r"\b50\b", "fifty"),
    (r"\b30\b", "thirty"),
    (r"\b15\b", "fifteen"),
    (r"\b7\b", "seven"),
    (r"\b3\b", "three"),
    (r"\b2\b", "two"),
)

DEFAULT_PHYSICAL_CHANNEL = "rir"
DEFAULT_RIR_NORMALIZATION = "energy"
DEFAULT_RIR_SEED = 1234
DEFAULT_MAX_RIR_SECONDS = 1.0
DEFAULT_RIR_TAIL_SECONDS = 0.5
DEFAULT_ROOM_RT60_S = 0.40
DEFAULT_SPEECH_DISTANCE_M = 1.0
DEFAULT_ENVIRONMENT_DISTANCE_M = 1.0
EARLY_REFLECTION_COUNT = 14

DEFAULT_MICROPHONE_BANDPASS = "butterworth"
MICROPHONE_BANDPASS_CHOICES = ("none", "butterworth")
MICROPHONE_FILTER_ORDER = 4
MICROPHONE_LOWCUT_HZ = 100.0
MICROPHONE_HIGHCUT_HZ = 7200.0

DEFAULT_PARAMETER_BOUNDS = {
    "pitch_steps": (-4.0, 4.0),
    "rate": (0.80, 1.20),
    "spectral_tilt_db_per_octave": (-6.0, 6.0),
}
FEATURE_OPTIMIZATION_ORDER = (
    "snr_db",
    "pitch_steps",
    "rate",
    "spectral_tilt_db_per_octave",
)
FEATURE_DISPLAY_NAMES = {
    "snr_db": "Loudness / SNR",
    "pitch_steps": "Pitch",
    "rate": "Rate",
    "spectral_tilt_db_per_octave": "Tilt",
}
DEFAULT_PATTERN_INITIAL_STEPS = {
    "snr_db": 5.0,
    "pitch_steps": 1.0,
    "rate": 0.05,
    "spectral_tilt_db_per_octave": 1.5,
}
DEFAULT_PATTERN_MIN_STEPS = {
    "snr_db": 0.25,
    "pitch_steps": 0.125,
    "rate": 0.005,
    "spectral_tilt_db_per_octave": 0.25,
}
ENVIRONMENT_LOOP_CROSSFADE_MS = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one environment sound for one speech sample against "
            "Wav2Vec using hard-text ASR feedback."
        )
    )
    parser.add_argument(
        "--data-test-root",
        type=Path,
        default=DEFAULT_DATA_TEST_ROOT,
        help="Paired test-data root. Default: ./data_test.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help=(
            "Matched subdirectory name, for example 01_sample. If no audio "
            "paths are given, the default is 01_sample."
        ),
    )
    parser.add_argument(
        "--speech",
        type=Path,
        default=None,
        help="Explicit speech path; otherwise resolved using --sample-id.",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=None,
        help="Explicit environment path; otherwise resolved using --sample-id.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="Optional ground-truth transcript used only for reporting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--speaker-name",
        default=None,
        help="Optional output label; defaults to the speech parent name.",
    )
    parser.add_argument(
        "--sentence-name",
        default=None,
        help="Optional output label; defaults to the speech filename stem.",
    )

    channel = parser.add_argument_group("physical channel")
    channel.add_argument(
        "--physical-channel",
        "--channel",
        dest="physical_channel",
        choices=("none", "rir"),
        default=DEFAULT_PHYSICAL_CHANNEL,
    )
    channel.add_argument(
        "--speech-rir",
        type=Path,
        default=None,
        help="Optional measured RIR from the speech loudspeaker to receiver.",
    )
    channel.add_argument(
        "--environment-rir",
        type=Path,
        default=None,
        help=(
            "Optional measured RIR from the environment loudspeaker to "
            "receiver."
        ),
    )
    channel.add_argument(
        "--rir-normalization",
        choices=("none", "peak", "energy"),
        default=DEFAULT_RIR_NORMALIZATION,
    )
    channel.add_argument("--rir-seed", type=int, default=DEFAULT_RIR_SEED)
    channel.add_argument(
        "--max-rir-seconds",
        type=float,
        default=DEFAULT_MAX_RIR_SECONDS,
    )
    channel.add_argument(
        "--rir-tail-seconds",
        type=float,
        default=DEFAULT_RIR_TAIL_SECONDS,
    )
    channel.add_argument(
        "--room-rt60",
        type=float,
        default=DEFAULT_ROOM_RT60_S,
    )
    channel.add_argument(
        "--speech-distance",
        type=float,
        default=DEFAULT_SPEECH_DISTANCE_M,
    )
    channel.add_argument(
        "--environment-distance",
        type=float,
        default=DEFAULT_ENVIRONMENT_DISTANCE_M,
    )
    channel.add_argument(
        "--microphone-bandpass",
        choices=MICROPHONE_BANDPASS_CHOICES,
        default=DEFAULT_MICROPHONE_BANDPASS,
    )

    search = parser.add_argument_group("coordinate pattern search")
    search.add_argument(
        "--rounds",
        "--steps",
        dest="rounds",
        type=int,
        default=10,
    )
    search.add_argument(
        "--snr-search-min-cer",
        type=float,
        default=DEFAULT_SNR_SEARCH_MIN_CER,
    )
    search.add_argument(
        "--pattern-initial-steps",
        nargs=4,
        type=float,
        default=[
            DEFAULT_PATTERN_INITIAL_STEPS[name]
            for name in FEATURE_OPTIMIZATION_ORDER
        ],
        metavar=("SNR_DB", "PITCH", "RATE", "TILT"),
    )
    search.add_argument(
        "--pattern-min-steps",
        nargs=4,
        type=float,
        default=[
            DEFAULT_PATTERN_MIN_STEPS[name]
            for name in FEATURE_OPTIMIZATION_ORDER
        ],
        metavar=("SNR_DB", "PITCH", "RATE", "TILT"),
    )
    search.add_argument("--pattern-step-growth", type=float, default=1.25)
    search.add_argument("--pattern-step-shrink", type=float, default=0.50)
    search.add_argument("--snr-min", type=float, default=-10.0)
    search.add_argument("--snr-max", type=float, default=20.0)

    objective = parser.add_argument_group("attack and quality objective")
    objective.add_argument("--min-attack-cer", type=float, default=0.1)

    wav2vec = parser.add_argument_group("Wav2Vec")
    wav2vec.add_argument(
        "--wav2vec-model",
        type=Path,
        required=True,
        help="Path to the fairseq Wav2Vec CTC checkpoint.",
    )
    wav2vec.add_argument(
        "--wav2vec-dictionary",
        type=Path,
        required=True,
        help="Path to the fairseq dict.ltr.txt file.",
    )
    wav2vec.add_argument(
        "--wav2vec-chunk-seconds",
        type=float,
        default=WAV2VEC_CHUNK_SECONDS,
    )
    wav2vec.add_argument(
        "--wav2vec-stride-seconds",
        type=float,
        default=WAV2VEC_STRIDE_SECONDS,
    )
    wav2vec.add_argument("--force-cpu", action="store_true")

    nisqa = parser.add_argument_group("NISQA")
    nisqa.add_argument(
        "--nisqa-root",
        type=Path,
        required=True,
        help="Path to the NISQA repository root.",
    )
    nisqa.add_argument(
        "--nisqa-model",
        type=Path,
        required=True,
        help="Path to the NISQA MOS checkpoint.",
    )
    nisqa.add_argument(
        "--nisqa-device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and configuration without loading the models.",
    )
    return parser.parse_args()


def _identity(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or Path(value).name != value:
        raise ValueError(f"Invalid {label} name: {value!r}")
    return value


def find_single_sample_file(
    directory: Path,
    extensions: Sequence[str],
    label: str,
    preferred_stem: Optional[str] = None,
) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} directory does not exist: {directory}")
    extension_set = {extension.lower() for extension in extensions}
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extension_set
    )
    if preferred_stem is not None:
        preferred = [path for path in candidates if path.stem == preferred_stem]
        if len(preferred) == 1:
            return preferred[0]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} file in {directory}, found "
            f"{len(candidates)}"
        )
    return candidates[0]


def resolve_test_sample_paths(args: argparse.Namespace) -> None:
    """Resolve one paired speech/environment sample from data_test."""
    args.data_test_root = args.data_test_root.expanduser().resolve()
    if args.sample_id is None and args.speech is None and args.environment is None:
        args.sample_id = "01_sample"

    if args.sample_id is not None:
        sample_id = _identity(args.sample_id, "sample")
        speech_dir = args.data_test_root / "speech" / sample_id
        environment_dir = args.data_test_root / "environment" / sample_id
        if args.speech is None:
            args.speech = find_single_sample_file(
                speech_dir,
                tuple(AUDIO_EXTENSIONS),
                "speech audio",
                preferred_stem=sample_id,
            )
        if args.environment is None:
            args.environment = find_single_sample_file(
                environment_dir,
                tuple(AUDIO_EXTENSIONS),
                "environment audio",
            )
        if args.reference_file is None:
            args.reference_file = find_single_sample_file(
                speech_dir,
                (".txt",),
                "reference transcript",
                preferred_stem=sample_id,
            )

    if args.speech is None or args.environment is None:
        raise ValueError(
            "Provide --sample-id, or provide both --speech and --environment"
        )
    if args.reference_file is None:
        automatic_reference = args.speech.with_suffix(".txt")
        if automatic_reference.is_file():
            args.reference_file = automatic_reference


def validate_args(args: argparse.Namespace) -> Tuple[str, str]:
    resolve_test_sample_paths(args)
    for name in (
        "speech",
        "environment",
        "output_dir",
        "wav2vec_model",
        "wav2vec_dictionary",
        "nisqa_root",
        "nisqa_model",
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
    if args.reference_file is not None:
        args.reference_file = args.reference_file.expanduser().resolve()
    for name in ("speech_rir", "environment_rir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())

    for label, path in (
        ("speech", args.speech),
        ("environment", args.environment),
        ("Wav2Vec checkpoint", args.wav2vec_model),
        ("Wav2Vec dictionary", args.wav2vec_dictionary),
        ("NISQA checkpoint", args.nisqa_model),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.nisqa_root.is_dir():
        raise NotADirectoryError(
            f"NISQA repository does not exist: {args.nisqa_root}"
        )
    if args.reference_file is not None and not args.reference_file.is_file():
        raise FileNotFoundError(
            f"Reference transcript does not exist: {args.reference_file}"
        )
    for label, path in (
        ("Speech RIR", args.speech_rir),
        ("Environment RIR", args.environment_rir),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.speech.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported speech format: {args.speech}")
    if args.environment.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported environment format: {args.environment}")
    if args.wav2vec_dictionary.name != "dict.ltr.txt":
        raise ValueError("Wav2Vec dictionary must be named dict.ltr.txt")

    for label, value in (
        ("--max-rir-seconds", args.max_rir_seconds),
        ("--room-rt60", args.room_rt60),
        ("--speech-distance", args.speech_distance),
        ("--environment-distance", args.environment_distance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    if not math.isfinite(args.rir_tail_seconds) or args.rir_tail_seconds < 0:
        raise ValueError("--rir-tail-seconds must be finite and non-negative")
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.snr_min >= args.snr_max:
        raise ValueError("--snr-min must be smaller than --snr-max")
    if args.min_attack_cer < 0 or args.snr_search_min_cer < 0:
        raise ValueError("CER thresholds must be non-negative")
    if any(value <= 0 for value in args.pattern_initial_steps):
        raise ValueError("--pattern-initial-steps must be positive")
    if any(value <= 0 for value in args.pattern_min_steps):
        raise ValueError("--pattern-min-steps must be positive")
    if any(
        minimum > initial
        for initial, minimum in zip(
            args.pattern_initial_steps,
            args.pattern_min_steps,
        )
    ):
        raise ValueError("Minimum pattern steps cannot exceed initial steps")
    if args.pattern_step_growth < 1:
        raise ValueError("--pattern-step-growth must be at least 1")
    if not 0 < args.pattern_step_shrink < 1:
        raise ValueError("--pattern-step-shrink must be in (0, 1)")
    if args.wav2vec_chunk_seconds < 0 or args.wav2vec_stride_seconds < 0:
        raise ValueError("Wav2Vec chunk and stride must be non-negative")
    if (
        args.wav2vec_chunk_seconds > 0
        and 2 * args.wav2vec_stride_seconds >= args.wav2vec_chunk_seconds
    ):
        raise ValueError(
            "Wav2Vec stride must be less than half of chunk duration"
        )

    raw_speaker = args.speech.parent.name
    speaker = (
        raw_speaker[:1].upper() + raw_speaker[1:]
        if args.speaker_name is None
        else args.speaker_name
    )
    sentence = (
        args.speech.stem
        if args.sentence_name is None
        else args.sentence_name
    )
    return _identity(speaker, "speaker"), _identity(sentence, "sentence")



@dataclass
class PhysicalChannelState:
    speech_rir: np.ndarray
    speech_rir_name: str
    environment_rir: np.ndarray
    environment_rir_name: str
    room_rt60_s: Optional[float]
    speech_distance_m: Optional[float]
    environment_distance_m: Optional[float]
    normalization: str
    tail_seconds: float


@dataclass(frozen=True)
class AttackParameters:
    snr_db: float
    pitch_steps: float
    rate: float
    spectral_tilt_db_per_octave: float


@dataclass
class AttackEvaluation:
    query: int
    phase: str
    step: int
    candidate: int
    parameters: AttackParameters
    transcript_raw: str
    transcript_normalized: str
    attack_success: bool
    attack_wer: float
    attack_cer: float
    pre_rir_transcript_raw: str
    pre_rir_transcript_normalized: str
    pre_rir_attack_success: bool
    pre_rir_attack_wer: float
    pre_rir_attack_cer: float
    nisqa_mos: Optional[float]
    actual_snr_db: float

    def rank_key(self) -> Tuple[float, ...]:
        """Enforce joint pre/receiver CER success, then maximize NISQA MOS."""
        if self.attack_success:
            # Successful candidates always have a MOS in the normal run.
            # Keep the None fallback so lightweight tests without NISQA can
            # still compare candidates deterministically.
            negative_mos = (
                float("inf")
                if self.nisqa_mos is None
                else -self.nisqa_mos
            )
            return (0.0, negative_mos)
        return (
            1.0,
            -min(self.attack_cer, self.pre_rir_attack_cer),
            -(self.attack_cer + self.pre_rir_attack_cer),
        )

    def csv_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "query": self.query,
            "phase": self.phase,
            "step": self.step,
            "candidate": self.candidate,
        }
        row.update(asdict(self.parameters))
        row.update(
            {
                "transcript_raw": self.transcript_raw,
                "transcript_normalized": self.transcript_normalized,
                "attack_success": self.attack_success,
                "attack_wer": self.attack_wer,
                "attack_cer": self.attack_cer,
                "pre_rir_transcript_raw": self.pre_rir_transcript_raw,
                "pre_rir_transcript_normalized": (
                    self.pre_rir_transcript_normalized
                ),
                "pre_rir_attack_success": self.pre_rir_attack_success,
                "pre_rir_attack_wer": self.pre_rir_attack_wer,
                "pre_rir_attack_cer": self.pre_rir_attack_cer,
                "nisqa_mos": (
                    "" if self.nisqa_mos is None else self.nisqa_mos
                ),
                "actual_snr_db": self.actual_snr_db,
            }
        )
        return row



class Wav2VecASR:
    """Load a fairseq CTC checkpoint and return greedy hard-text output."""

    def __init__(
        self,
        model_path: Path,
        dictionary_path: Optional[Path] = None,
        chunk_seconds: float = 20.0,
        stride_seconds: float = 4.0,
        force_cpu: bool = False,
        input_sample_rate: int = 16000,
    ) -> None:
        import torch
        from fairseq import checkpoint_utils

        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Fairseq checkpoint not found: {model_path}")
        if dictionary_path is None:
            dictionary_path = model_path.parent / "dict.ltr.txt"
        dictionary_path = dictionary_path.expanduser().resolve()
        if not dictionary_path.is_file():
            raise FileNotFoundError(
                f"Fairseq letter dictionary not found: {dictionary_path}"
            )
        if dictionary_path.name != "dict.ltr.txt":
            raise ValueError("The fairseq dictionary must be named dict.ltr.txt")
        if chunk_seconds < 0.0:
            raise ValueError("chunk_seconds must be non-negative")
        if stride_seconds < 0.0:
            raise ValueError("stride_seconds must be non-negative")
        if chunk_seconds > 0.0 and 2.0 * stride_seconds >= chunk_seconds:
            raise ValueError(
                "stride_seconds must be less than half of chunk_seconds"
            )

        self.torch = torch
        self.device = torch.device(
            "cpu"
            if force_cpu
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.input_sample_rate = int(input_sample_rate)
        self.chunk_seconds = float(chunk_seconds)
        self.stride_seconds = float(stride_seconds)

        print(f"[wav2vec] Loading {model_path} on {self.device}")
        overrides = {
            "task": "audio_finetuning",
            "data": str(dictionary_path.parent),
            "labels": "ltr",
        }
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [str(model_path)],
            arg_overrides=overrides,
        )
        self.model = models[0].eval().to(self.device)
        self.dictionary = task.target_dictionary
        self.blank = self.dictionary.bos()
        self.sample_rate = int(cfg.task.sample_rate)
        print(f"[wav2vec] Sample rate: {self.sample_rate} Hz")

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        if self.input_sample_rate == self.sample_rate:
            return audio
        from torchaudio.functional import resample

        waveform = self.torch.from_numpy(audio)
        return resample(
            waveform,
            self.input_sample_rate,
            self.sample_rate,
        ).numpy()

    def _decode_chunk(self, audio: np.ndarray) -> List[Tuple[str, float, float]]:
        source = self.torch.from_numpy(audio).to(self.device).unsqueeze(0)
        with self.torch.inference_mode():
            encoder_out = self.model(source=source, padding_mask=None)
            logits = self.model.get_logits(encoder_out)[:, 0]
        token_ids = logits.argmax(dim=-1).cpu().tolist()

        runs: List[Tuple[int, int, int]] = []
        start = 0
        for token, group in itertools.groupby(token_ids):
            length = sum(1 for _ in group)
            end = start + length
            if token != self.blank:
                runs.append((token, start, end))
            start = end

        seconds_per_frame = (
            len(audio) / self.sample_rate / max(len(token_ids), 1)
        )
        words: List[Tuple[str, float, float]] = []
        letters: List[str] = []
        word_start: Optional[float] = None
        word_end = 0.0
        for token, frame_start, frame_end in runs:
            symbol = self.dictionary[token]
            if symbol == "|":
                if letters:
                    words.append(
                        (
                            "".join(letters),
                            word_start or 0.0,
                            word_end,
                        )
                    )
                    letters = []
                    word_start = None
                continue
            if symbol.startswith("<"):
                continue
            if word_start is None:
                word_start = frame_start * seconds_per_frame
            word_end = frame_end * seconds_per_frame
            letters.append(symbol)
        if letters:
            words.append(("".join(letters), word_start or 0.0, word_end))
        return words

    def transcribe(self, audio: np.ndarray) -> str:
        """Return a greedy CTC transcript for a mono floating waveform."""
        audio = self._resample(np.asarray(audio, dtype=np.float32))
        duration = len(audio) / self.sample_rate
        if self.chunk_seconds <= 0.0 or duration <= self.chunk_seconds:
            words = self._decode_chunk(audio)
            return " ".join(word for word, _, _ in words).strip()

        chunk_samples = round(self.chunk_seconds * self.sample_rate)
        stride_samples = round(self.stride_seconds * self.sample_rate)
        step_samples = chunk_samples - 2 * stride_samples
        selected_words: List[str] = []
        chunk_start = 0
        while chunk_start < len(audio):
            chunk_end = min(chunk_start + chunk_samples, len(audio))
            words = self._decode_chunk(audio[chunk_start:chunk_end])
            left = 0.0 if chunk_start == 0 else self.stride_seconds
            chunk_duration = (chunk_end - chunk_start) / self.sample_rate
            right = (
                chunk_duration
                if chunk_end == len(audio)
                else chunk_duration - self.stride_seconds
            )
            for word, start_time, end_time in words:
                midpoint = 0.5 * (start_time + end_time)
                if left <= midpoint < right or (
                    chunk_end == len(audio) and midpoint <= right
                ):
                    selected_words.append(word)
            if chunk_end == len(audio):
                break
            chunk_start += step_samples
        return " ".join(selected_words).strip()

    def close(self) -> None:
        """Release model memory."""
        if self.model is not None:
            model = self.model
            self.model = None
            del model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def expand_explicit_numbers(text: str) -> str:
    """Spell numeric forms according to the fixed Qwen TTS prompts."""
    for pattern, replacement in EXPLICIT_NUMBER_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    """Apply identical number and text normalization to all ASR text."""
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = text.replace("’", "'")
    text = expand_explicit_numbers(text)
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported audio shape for {path}: {audio.shape}")
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    if sample_rate != target_sr:
        divisor = gcd(int(sample_rate), int(target_sr))
        audio = resample_poly(
            audio,
            target_sr // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Audio contains NaN or Inf: {path}")
    return np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


def write_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError(f"Cannot write audio containing NaN or Inf: {path}")
    # Preserve evaluated samples when peak control is disabled. PCM16 cannot
    # represent values outside [-1, 1], so use IEEE-float WAV rather than
    # silently hard-clipping an out-of-range waveform at save time.
    subtype = (
        "FLOAT"
        if output.size and float(np.max(np.abs(output))) > 1.0
        else "PCM_16"
    )
    sf.write(
        str(path),
        output,
        TARGET_SAMPLE_RATE,
        subtype=subtype,
    )


def rms(audio: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + eps))


def calculate_snr_db(
    speech_component: np.ndarray,
    environment_component: np.ndarray,
) -> float:
    speech_power = float(np.mean(np.square(speech_component, dtype=np.float64)))
    environment_power = float(
        np.mean(np.square(environment_component, dtype=np.float64))
    )
    return 10.0 * math.log10(
        max(speech_power, 1e-20) / max(environment_power, 1e-20)
    )


def normalize_rir(rir: np.ndarray, mode: str) -> np.ndarray:
    rir = np.asarray(rir, dtype=np.float32)
    if rir.size == 0:
        raise ValueError("RIR is empty")
    if not np.all(np.isfinite(rir)):
        raise ValueError("RIR contains NaN or Inf")
    if mode == "none":
        denominator = 1.0
    elif mode == "peak":
        denominator = float(np.max(np.abs(rir)))
    elif mode == "energy":
        denominator = float(
            np.sqrt(np.sum(np.square(rir, dtype=np.float64)))
        )
    else:
        raise ValueError(f"Unknown RIR normalization: {mode}")
    if denominator <= 1e-12:
        raise ValueError("RIR is silent")
    return np.asarray(rir / denominator, dtype=np.float32)


def load_rir(
    path: Path,
    normalization: str,
    max_seconds: float,
) -> np.ndarray:
    rir, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if rir.ndim == 2:
        rir = rir.mean(axis=1, dtype=np.float32)
    elif rir.ndim != 1:
        raise ValueError(f"Unsupported RIR shape for {path}: {rir.shape}")
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        rir = resample_poly(
            rir,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    max_samples = max(1, int(round(max_seconds * TARGET_SAMPLE_RATE)))
    return normalize_rir(rir[:max_samples], normalization)


def generate_synthetic_rir(
    rt60_s: float,
    distance_m: float,
    max_seconds: float,
    normalization: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a deterministic lightweight RIR with early reflections."""
    speed_of_sound = 343.0
    direct_delay = max(
        0,
        int(round(distance_m / speed_of_sound * TARGET_SAMPLE_RATE)),
    )
    total_length = max(
        direct_delay + 2,
        int(round((rt60_s + 0.10) * TARGET_SAMPLE_RATE)),
    )
    total_length = min(
        total_length,
        max(1, int(round(max_seconds * TARGET_SAMPLE_RATE))),
    )
    time = np.arange(total_length, dtype=np.float64) / TARGET_SAMPLE_RATE
    decay = np.power(10.0, -3.0 * time / max(rt60_s, 1e-3))
    direct_gain = 1.0 / max(distance_m, 0.25)
    rir = (
        rng.normal(0.0, 1.0, total_length)
        * decay
        * 0.004
        * direct_gain
    )
    if direct_delay < total_length:
        rir[direct_delay] += direct_gain

    for _ in range(EARLY_REFLECTION_COUNT):
        maximum_delay = min(0.080, rt60_s * 0.50)
        if maximum_delay <= 0.003:
            break
        delay_s = float(rng.uniform(0.003, maximum_delay))
        index = direct_delay + int(round(delay_s * TARGET_SAMPLE_RATE))
        if index >= total_length:
            continue
        gain = (
            direct_gain
            * float(rng.uniform(0.05, 0.35))
            * float(10.0 ** (-3.0 * delay_s / max(rt60_s, 1e-3)))
        )
        rir[index] += gain * (-1.0 if rng.random() < 0.5 else 1.0)
    return normalize_rir(np.asarray(rir, dtype=np.float32), normalization)


def build_physical_channel(
    args: argparse.Namespace,
) -> Optional[PhysicalChannelState]:
    if args.physical_channel == "none":
        return None

    rng = np.random.default_rng(args.rir_seed)

    def source_rir(
        path: Optional[Path],
        source_name: str,
        distance_m: float,
    ) -> Tuple[np.ndarray, str, Optional[float]]:
        if path is not None:
            return (
                load_rir(
                    path,
                    args.rir_normalization,
                    args.max_rir_seconds,
                ),
                str(path.resolve()),
                None,
            )
        return (
            generate_synthetic_rir(
                args.room_rt60,
                distance_m,
                args.max_rir_seconds,
                args.rir_normalization,
                rng,
            ),
            f"synthetic_{source_name}_rir",
            float(distance_m),
        )

    speech_rir, speech_rir_name, speech_distance = source_rir(
        args.speech_rir,
        "speech",
        args.speech_distance,
    )
    environment_rir, environment_rir_name, environment_distance = source_rir(
        args.environment_rir,
        "environment",
        args.environment_distance,
    )

    return PhysicalChannelState(
        speech_rir=speech_rir,
        speech_rir_name=speech_rir_name,
        environment_rir=environment_rir,
        environment_rir_name=environment_rir_name,
        room_rt60_s=(
            None
            if args.speech_rir is not None
            and args.environment_rir is not None
            else float(args.room_rt60)
        ),
        speech_distance_m=speech_distance,
        environment_distance_m=environment_distance,
        normalization=args.rir_normalization,
        tail_seconds=float(args.rir_tail_seconds),
    )


def apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    target_length: int,
) -> np.ndarray:
    received = fftconvolve(audio, rir, mode="full")
    if received.size < target_length:
        received = np.pad(received, (0, target_length - received.size))
    return np.asarray(received[:target_length], dtype=np.float32)


def build_microphone_bandpass(mode: str) -> Optional[np.ndarray]:
    """Build the one fixed causal receiver filter used by every candidate."""
    if mode == "none":
        return None
    if mode != "butterworth":
        raise ValueError(f"Unknown microphone band-pass mode: {mode}")
    return np.asarray(
        butter(
            MICROPHONE_FILTER_ORDER,
            [MICROPHONE_LOWCUT_HZ, MICROPHONE_HIGHCUT_HZ],
            btype="bandpass",
            fs=TARGET_SAMPLE_RATE,
            output="sos",
        ),
        dtype=np.float64,
    )


def apply_microphone_bandpass(
    audio: np.ndarray,
    sos: Optional[np.ndarray],
) -> np.ndarray:
    """Apply the fixed microphone response, or copy when it is disabled."""
    source = np.asarray(audio, dtype=np.float32)
    if sos is None:
        return source.copy()
    return np.asarray(sosfilt(sos, source), dtype=np.float32)


def microphone_bandpass_metadata(mode: str) -> Dict[str, Any]:
    enabled = mode == "butterworth"
    return {
        "microphone_bandpass": mode,
        "microphone_bandpass_enabled": enabled,
        "microphone_filter_type": (
            "causal_butterworth_sos" if enabled else None
        ),
        "microphone_filter_order": (
            MICROPHONE_FILTER_ORDER if enabled else None
        ),
        "microphone_lowcut_hz": MICROPHONE_LOWCUT_HZ if enabled else None,
        "microphone_highcut_hz": MICROPHONE_HIGHCUT_HZ if enabled else None,
    }


def channel_metadata(
    state: Optional[PhysicalChannelState],
) -> Dict[str, Any]:
    if state is None:
        return {
            "physical_channel": "none",
            "topology": "speech_and_environment_bypass_then_receiver_mix",
        }
    return {
        "physical_channel": "rir",
        "topology": (
            "speech_rir_and_environment_rir_then_receiver_mix_then_"
            "microphone_then_asr"
        ),
        "speech_rir": state.speech_rir_name,
        "speech_rir_samples": int(state.speech_rir.size),
        "environment_rir": state.environment_rir_name,
        "environment_rir_samples": int(state.environment_rir.size),
        "room_rt60_s": state.room_rt60_s,
        "speech_distance_m": state.speech_distance_m,
        "environment_distance_m": state.environment_distance_m,
        "rir_normalization": state.normalization,
        "rir_tail_seconds": state.tail_seconds,
        "speech_rir_fixed_during_optimization": True,
        "environment_rir_fixed_during_optimization": True,
    }


def time_stretch_audio(audio: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 1e-8:
        return audio.copy()
    import librosa

    return np.asarray(
        librosa.effects.time_stretch(audio, rate=float(rate)),
        dtype=np.float32,
    )


def pitch_shift_audio(audio: np.ndarray, steps: float) -> np.ndarray:
    if abs(steps) < 1e-8:
        return audio.copy()
    import librosa

    return np.asarray(
        librosa.effects.pitch_shift(
            audio,
            sr=TARGET_SAMPLE_RATE,
            n_steps=float(steps),
        ),
        dtype=np.float32,
    )


def apply_spectral_tilt(
    audio: np.ndarray,
    db_per_octave: float,
    reference_hz: float = 1000.0,
) -> np.ndarray:
    if abs(db_per_octave) < 1e-8 or audio.size < 64:
        return audio.copy()
    nperseg = min(1024, max(64, int(2 ** math.floor(math.log2(audio.size)))))
    noverlap = int(nperseg * 0.75)
    frequencies, _, spectrum = stft(
        audio,
        fs=TARGET_SAMPLE_RATE,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    safe_frequencies = np.maximum(frequencies, 20.0)
    gain_db = db_per_octave * np.log2(safe_frequencies / reference_hz)
    gain_db = np.clip(gain_db, -24.0, 24.0)
    spectrum *= np.power(10.0, gain_db / 20.0)[:, None]
    _, output = istft(
        spectrum,
        fs=TARGET_SAMPLE_RATE,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    if output.size < audio.size:
        output = np.pad(output, (0, audio.size - output.size))
    return np.asarray(output[: audio.size], dtype=np.float32)


def fit_to_length_with_sample_offset(
    audio: np.ndarray,
    target_length: int,
    offset_samples: int,
) -> np.ndarray:
    if audio.size == 0:
        raise ValueError("Cannot align an empty environment waveform")
    offset = int(offset_samples) % audio.size
    indices = (offset + np.arange(target_length, dtype=np.int64)) % audio.size
    return np.asarray(audio[indices], dtype=np.float32)


def make_crossfaded_environment_loop(audio: np.ndarray) -> np.ndarray:
    """Return a periodic environment loop with a smooth end/start join.

    The end and beginning are overlapped rather than concatenated. Repeating
    the returned waveform therefore does not introduce a hard sample jump at
    every environment boundary.
    """
    if audio.size < 4:
        return np.asarray(audio, dtype=np.float32).copy()

    requested_samples = int(
        round(ENVIRONMENT_LOOP_CROSSFADE_MS * TARGET_SAMPLE_RATE / 1000.0)
    )
    crossfade_samples = min(
        max(1, requested_samples),
        max(1, audio.size // 4),
    )

    # The periodic boundary after the overlap is between audio[k - 1] and
    # audio[k]. Move k within a small neighbourhood of the requested duration
    # to avoid placing that boundary on an existing sharp transient.
    search_start = max(1, crossfade_samples // 2)
    search_stop = min(
        max(search_start + 1, audio.size // 4),
        crossfade_samples + max(2, crossfade_samples // 2),
    )
    candidate_offsets = np.arange(search_start, search_stop, dtype=np.int64)
    boundary_changes = np.abs(
        np.asarray(audio[candidate_offsets], dtype=np.float64)
        - np.asarray(audio[candidate_offsets - 1], dtype=np.float64)
    )
    crossfade_samples = int(
        candidate_offsets[int(np.argmin(boundary_changes))]
    )

    phase = (
        np.arange(1, crossfade_samples + 1, dtype=np.float64)
        / (crossfade_samples + 1)
    )
    fade_in = 0.5 - 0.5 * np.cos(np.pi * phase)
    overlap = (
        np.asarray(audio[-crossfade_samples:], dtype=np.float64)
        * (1.0 - fade_in)
        + np.asarray(audio[:crossfade_samples], dtype=np.float64) * fade_in
    )
    middle = np.asarray(
        audio[crossfade_samples:-crossfade_samples],
        dtype=np.float64,
    )
    return np.asarray(np.concatenate((middle, overlap)), dtype=np.float32)


def transform_environment(
    speech: np.ndarray,
    environment: np.ndarray,
    parameters: AttackParameters,
) -> np.ndarray:
    """Apply only global rate, pitch, and spectral-tilt features."""
    output = time_stretch_audio(environment, parameters.rate)
    output = pitch_shift_audio(output, parameters.pitch_steps)
    output = apply_spectral_tilt(
        output,
        parameters.spectral_tilt_db_per_octave,
    )
    output = make_crossfaded_environment_loop(output)
    output = fit_to_length_with_sample_offset(output, speech.size, 0)
    if rms(output) <= 1e-8:
        raise ValueError("Environment transform produced a silent waveform")
    return output


class NISQAScorer:
    """Optional reusable wrapper around the local NISQA repository."""

    def __init__(
        self,
        root: Path,
        model_path: Path,
        initial_audio_path: Path,
        device: str,
    ) -> None:
        root_string = str(root.resolve())
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            from nisqa import NISQA_lib as nisqa_lib
            from nisqa.NISQA_model import nisqaModel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "NISQA dependencies are incomplete "
                f"(missing {exc.name!r}). Stage 2 requires NISQA for final "
                "candidate ranking; install the dependencies required by "
                "the local NISQA repository."
            ) from exc

        self.nisqa_lib = nisqa_lib
        args = {
            "mode": "predict_file",
            "pretrained_model": str(model_path.resolve()),
            "deg": str(initial_audio_path.resolve()),
            "output_dir": None,
            "tr_bs_val": 1,
            "tr_num_workers": 0,
            "tr_device": device,
            "ms_channel": None,
        }
        print(f"[NISQA] Loading {model_path} on {device}")
        self.runner = nisqaModel(args)

    def score_file(self, audio_path: Path) -> float:
        self.runner.args["deg"] = str(audio_path.resolve())
        self.runner._loadDatasetsFile()
        if self.runner.args["dim"]:
            prediction, _ = self.nisqa_lib.predict_dim(
                self.runner.model,
                self.runner.ds_val,
                1,
                self.runner.dev,
                num_workers=0,
            )
        else:
            prediction, _ = self.nisqa_lib.predict_mos(
                self.runner.model,
                self.runner.ds_val,
                1,
                self.runner.dev,
                num_workers=0,
            )
        return float(prediction[0, 0])


class SingleSampleAttack:
    def __init__(
        self,
        speech: np.ndarray,
        environment: np.ndarray,
        asr: Any,
        clean_text_raw: str,
        received_length: int,
        physical_channel: Optional[PhysicalChannelState],
        microphone_sos: Optional[np.ndarray],
        args: argparse.Namespace,
        nisqa_scorer: Optional[NISQAScorer],
        temp_dir: Path,
    ) -> None:
        self.speech = speech
        self.environment = environment
        self.asr = asr
        self.received_length = int(received_length)
        if self.received_length < 1:
            raise ValueError("Receiver output length must be positive")
        self.physical_channel = physical_channel
        self.microphone_sos = microphone_sos
        self.clean_text_raw = clean_text_raw
        self.clean_text_normalized = normalize_text(clean_text_raw)
        self.args = args
        self.nisqa_scorer = nisqa_scorer
        self.temp_audio_path = temp_dir / "nisqa_candidate.wav"
        self.query_count = 0
        self.receiver_query_count = 0
        self.pre_rir_query_count = 0
        self.evaluation_count = 0
        self.evaluation_cache: Dict[AttackParameters, AttackEvaluation] = {}

        if not self.clean_text_normalized:
            raise ValueError("Target ASR returned an empty clean transcript")

        if self.physical_channel is None:
            acoustic_speech = np.asarray(self.speech, dtype=np.float32).copy()
        else:
            acoustic_speech = apply_rir(
                self.speech,
                self.physical_channel.speech_rir,
                self.received_length,
            )
        self.acoustic_speech = acoustic_speech
        self.received_speech = apply_microphone_bandpass(
            self.acoustic_speech,
            self.microphone_sos,
        )
        if rms(self.received_speech) <= 1e-8:
            raise ValueError("Speech is silent after its physical channel")

    def render(
        self,
        parameters: AttackParameters,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[str, Any],
    ]:
        transformed_playback = transform_environment(
            self.speech,
            self.environment,
            parameters,
        )

        if self.physical_channel is None:
            acoustic_environment_unscaled = transformed_playback.copy()
        else:
            acoustic_environment_unscaled = apply_rir(
                transformed_playback,
                self.physical_channel.environment_rir,
                self.received_length,
            )
        received_environment_unscaled = apply_microphone_bandpass(
            acoustic_environment_unscaled,
            self.microphone_sos,
        )
        if rms(received_environment_unscaled) <= 1e-8:
            raise ValueError("Environment is silent after its physical channel")

        environment_scale = rms(self.received_speech) / (
            rms(received_environment_unscaled)
            * float(10.0 ** (parameters.snr_db / 20.0))
        )
        playback_environment = np.asarray(
            transformed_playback * environment_scale,
            dtype=np.float32,
        )
        pre_rir_mixture = np.asarray(
            self.speech + playback_environment,
            dtype=np.float32,
        )
        received_environment = np.asarray(
            received_environment_unscaled * environment_scale,
            dtype=np.float32,
        )
        acoustic_environment = np.asarray(
            acoustic_environment_unscaled * environment_scale,
            dtype=np.float32,
        )
        acoustic_mixture = np.asarray(
            self.acoustic_speech + acoustic_environment,
            dtype=np.float32,
        )
        received_mixture = apply_microphone_bandpass(
            acoustic_mixture,
            self.microphone_sos,
        )
        actual_snr_db = calculate_snr_db(
            self.received_speech,
            received_environment,
        )
        details: Dict[str, Any] = {
            "target_snr_db": float(parameters.snr_db),
            "actual_snr_db": actual_snr_db,
            "environment_scale": float(environment_scale),
            "speech_gain": 1.0,
            **channel_metadata(self.physical_channel),
            **microphone_bandpass_metadata(
                self.args.microphone_bandpass
            ),
            "channel_order": (
                "speech_rir_and_environment_rir_then_receiver_mix_then_"
                "microphone"
            ),
            "snr_domain": "receiver_components_after_rir_and_microphone",
            "asr_audio_domain": "receiver_side_mixture",
            "nisqa_audio_domain": "receiver_side_mixture",
            "playback_snr_db": calculate_snr_db(
                self.speech,
                playback_environment,
            ),
            "pre_rir_mixture_peak": float(
                np.max(np.abs(pre_rir_mixture))
            ),
            "received_speech_rms": rms(self.received_speech),
            "received_environment_rms": rms(received_environment),
            "received_mixture_peak": float(
                np.max(np.abs(received_mixture))
            ),
        }
        return (
            received_mixture,
            pre_rir_mixture,
            playback_environment,
            received_environment,
            details,
        )

    def evaluate(
        self,
        parameters: AttackParameters,
        phase: str,
        step: int,
        candidate: int,
    ) -> AttackEvaluation:
        (
            received_mixture,
            pre_rir_mixture,
            _,
            _,
            details,
        ) = self.render(parameters)
        transcript_raw = self.asr.transcribe(received_mixture)
        self.query_count += 1
        self.receiver_query_count += 1
        transcript_normalized = normalize_text(transcript_raw)

        pre_rir_transcript_raw = self.asr.transcribe(pre_rir_mixture)
        self.query_count += 1
        self.pre_rir_query_count += 1
        self.evaluation_count += 1
        pre_rir_transcript_normalized = normalize_text(
            pre_rir_transcript_raw
        )

        attack_wer = float(
            wer(self.clean_text_normalized, transcript_normalized)
        )
        attack_cer = float(
            cer(self.clean_text_normalized, transcript_normalized)
        )
        receiver_attack_success = (
            transcript_normalized != self.clean_text_normalized
            and attack_cer > self.args.min_attack_cer
        )
        pre_rir_attack_wer = float(
            wer(
                self.clean_text_normalized,
                pre_rir_transcript_normalized,
            )
        )
        pre_rir_attack_cer = float(
            cer(
                self.clean_text_normalized,
                pre_rir_transcript_normalized,
            )
        )
        pre_rir_attack_success = (
            pre_rir_transcript_normalized != self.clean_text_normalized
            and pre_rir_attack_cer > self.args.min_attack_cer
        )
        attack_success = receiver_attack_success and pre_rir_attack_success

        nisqa_mos: Optional[float] = None
        if attack_success and self.nisqa_scorer is not None:
            write_audio(self.temp_audio_path, received_mixture)
            nisqa_mos = self.nisqa_scorer.score_file(self.temp_audio_path)

        evaluation = AttackEvaluation(
            query=self.evaluation_count,
            phase=phase,
            step=step,
            candidate=candidate,
            parameters=parameters,
            transcript_raw=transcript_raw,
            transcript_normalized=transcript_normalized,
            attack_success=attack_success,
            attack_wer=attack_wer,
            attack_cer=attack_cer,
            pre_rir_transcript_raw=pre_rir_transcript_raw,
            pre_rir_transcript_normalized=pre_rir_transcript_normalized,
            pre_rir_attack_success=pre_rir_attack_success,
            pre_rir_attack_wer=pre_rir_attack_wer,
            pre_rir_attack_cer=pre_rir_attack_cer,
            nisqa_mos=nisqa_mos,
            actual_snr_db=details["actual_snr_db"],
        )
        self.evaluation_cache[parameters] = evaluation
        return evaluation

    def cached_evaluation(
        self,
        parameters: AttackParameters,
    ) -> Optional[AttackEvaluation]:
        return self.evaluation_cache.get(parameters)


def neutral_parameters(snr_db: float) -> AttackParameters:
    return AttackParameters(
        snr_db=float(snr_db),
        pitch_steps=0.0,
        rate=1.0,
        spectral_tilt_db_per_octave=0.0,
    )


def is_better(
    candidate: AttackEvaluation,
    incumbent: Optional[AttackEvaluation],
) -> bool:
    return incumbent is None or candidate.rank_key() < incumbent.rank_key()


def parameter_bounds(args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
    return {
        "snr_db": (float(args.snr_min), float(args.snr_max)),
        **DEFAULT_PARAMETER_BOUNDS,
    }


def feature_rank_key(
    evaluation: AttackEvaluation,
    feature_name: str,
    attack: SingleSampleAttack,
) -> Tuple[float, ...]:
    """Use the separate SNR CER target only during the loudness update."""
    if feature_name != "snr_db":
        return evaluation.rank_key()

    loudness_target_met = (
        evaluation.transcript_normalized != attack.clean_text_normalized
        and evaluation.attack_cer > attack.args.snr_search_min_cer
        and evaluation.pre_rir_transcript_normalized
        != attack.clean_text_normalized
        and evaluation.pre_rir_attack_cer
        > attack.args.snr_search_min_cer
    )
    if loudness_target_met:
        if evaluation.nisqa_mos is not None:
            return (
                0.0,
                0.0,
                -evaluation.nisqa_mos,
            )
        # This can occur when --snr-search-min-cer is lower than the global
        # --min-attack-cer. Continue toward the global success threshold
        # before NISQA becomes available.
        return (
            0.0,
            1.0,
            -min(evaluation.attack_cer, evaluation.pre_rir_attack_cer),
            -(evaluation.attack_cer + evaluation.pre_rir_attack_cer),
        )
    return (
        1.0,
        1.0,
        -min(evaluation.attack_cer, evaluation.pre_rir_attack_cer),
        -(evaluation.attack_cer + evaluation.pre_rir_attack_cer),
    )


def build_pattern_candidates(
    current_value: float,
    step_size: float,
    bounds: Tuple[float, float],
) -> List[Tuple[str, float]]:
    """Build deterministic x+-step and x+-2*step coordinate candidates."""
    lower, upper = bounds
    candidates: List[Tuple[str, float]] = []
    seen = {round(float(current_value), 12)}

    def add(source: str, value: float) -> None:
        clipped = float(np.clip(value, lower, upper))
        key = round(clipped, 12)
        if key not in seen:
            seen.add(key)
            candidates.append((source, float(key)))

    for multiplier in (-2.0, -1.0, 1.0, 2.0):
        label = f"local_{multiplier:+.0f}"
        add(label, current_value + multiplier * step_size)

    return candidates


def optimize_single_feature(
    attack: SingleSampleAttack,
    current: AttackEvaluation,
    feature_name: str,
    bounds: Tuple[float, float],
    step_size: float,
    minimum_step: float,
    round_index: int,
    args: argparse.Namespace,
    history: List[AttackEvaluation],
) -> Tuple[AttackEvaluation, float]:
    """Run one deterministic multi-scale coordinate-pattern update."""
    display_name = FEATURE_DISPLAY_NAMES[feature_name]
    current_value = float(getattr(current.parameters, feature_name))
    candidates = build_pattern_candidates(
        current_value,
        step_size,
        bounds,
    )
    print(
        "\n================ "
        f"Round {round_index}: {display_name} / Pattern search "
        "================"
    )
    print(
        f"center={current_value:+.6f} step={step_size:.6f} "
        "local_scales=(-2,-1,+1,+2)"
    )
    if feature_name == "snr_db":
        print(
            "Loudness target: both pre-RIR and receiver transcripts changed "
            "and both "
            f"CER > {args.snr_search_min_cer:.4f}"
        )

    evaluated = [current]
    for candidate_index, (source, value) in enumerate(candidates, start=1):
        parameters = replace(
            current.parameters,
            **{feature_name: float(value)},
        )
        evaluation = attack.cached_evaluation(parameters)
        if evaluation is None:
            evaluation = attack.evaluate(
                parameters,
                phase=f"pattern_{feature_name}_{source}",
                step=round_index,
                candidate=candidate_index,
            )
            history.append(evaluation)
            cache_label = ""
        else:
            cache_label = " [cached]"
        evaluated.append(evaluation)
        print(
            f"  source={source:>10s} value={value:+.6f}{cache_label}"
        )
        print_evaluation(evaluation, prefix="  candidate")

    rank = lambda item: feature_rank_key(item, feature_name, attack)
    feature_best = min(evaluated, key=rank)
    improved = rank(feature_best) < rank(current)
    if improved:
        updated_step = min(
            step_size * args.pattern_step_growth,
            0.5 * (bounds[1] - bounds[0]),
        )
    else:
        feature_best = current
        updated_step = max(
            minimum_step,
            step_size * args.pattern_step_shrink,
        )

    action = "grow" if improved else "shrink"
    print(
        f"[{display_name}] improved={str(improved).lower()} "
        f"step: {step_size:.6f} -> {updated_step:.6f} ({action})"
    )
    print_evaluation(feature_best, prefix="  feature_best")
    return feature_best, float(updated_step)


def run_sequential_feature_optimization(
    attack: SingleSampleAttack,
    args: argparse.Namespace,
    history: List[AttackEvaluation],
) -> AttackEvaluation:
    """Optimize SNR, Pitch, Rate, and Tilt sequentially each round."""
    bounds_by_feature = parameter_bounds(args)
    step_sizes = dict(zip(FEATURE_OPTIMIZATION_ORDER, args.pattern_initial_steps))
    minimum_steps = dict(zip(FEATURE_OPTIMIZATION_ORDER, args.pattern_min_steps))

    neutral = neutral_parameters(0.5 * (args.snr_min + args.snr_max))
    current = attack.evaluate(
        neutral,
        phase="initial",
        step=0,
        candidate=0,
    )
    history.append(current)
    global_best = current
    print("\n================ Initial candidate ================")
    print_evaluation(current, prefix="initial")

    for round_index in range(1, args.rounds + 1):
        for feature_name in FEATURE_OPTIMIZATION_ORDER:
            feature_history_start = len(history)
            current, step_sizes[feature_name] = optimize_single_feature(
                attack,
                current,
                feature_name,
                bounds_by_feature[feature_name],
                float(step_sizes[feature_name]),
                float(minimum_steps[feature_name]),
                round_index,
                args,
                history,
            )
            for evaluation in history[feature_history_start:]:
                if is_better(evaluation, global_best):
                    global_best = evaluation
            if is_better(current, global_best):
                global_best = current

        current = global_best
        print(
            "\n---------------- "
            f"Round {round_index}/{args.rounds} complete "
            "----------------"
        )
        print_evaluation(global_best, prefix="global_best")

    return global_best



def print_evaluation(
    evaluation: AttackEvaluation,
    prefix: str = "candidate",
) -> None:
    nisqa_text = (
        ""
        if evaluation.nisqa_mos is None
        else f" NISQA={evaluation.nisqa_mos:.3f}"
    )
    print(
        f"{prefix} q={evaluation.query:4d} "
        f"success={str(evaluation.attack_success):5s} "
        f"receiverCER={evaluation.attack_cer:.4f} "
        f"preRIRCER={evaluation.pre_rir_attack_cer:.4f} "
        f"receiverWER={evaluation.attack_wer:.4f} "
        f"preRIRWER={evaluation.pre_rir_attack_wer:.4f} "
        f"SNR={evaluation.actual_snr_db:.2f} "
        f"pitch={evaluation.parameters.pitch_steps:+.3f} "
        f"rate={evaluation.parameters.rate:.3f} "
        f"tilt={evaluation.parameters.spectral_tilt_db_per_octave:+.3f} "
        f"{nisqa_text}"
    )
    print(f"  receiver transcript: {evaluation.transcript_raw}")
    print(f"  pre-RIR transcript : {evaluation.pre_rir_transcript_raw}")


def read_reference(path: Optional[Path]) -> str:
    if path is None:
        return ""
    if not path.is_file():
        raise FileNotFoundError(f"Reference file does not exist: {path}")
    return " ".join(path.read_text(encoding="utf-8").split())


def write_history(path: Path, history: Sequence[AttackEvaluation]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [evaluation.csv_row() for evaluation in history]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def safe_filename_component(value: str) -> str:
    """Make a stable readable filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "environment"





def save_single_result(
    result_dir: Path,
    args: argparse.Namespace,
    attack: SingleSampleAttack,
    best: AttackEvaluation,
    history: Sequence[AttackEvaluation],
    received_mixture: np.ndarray,
    pre_rir_mixture: np.ndarray,
    playback_environment: np.ndarray,
    received_environment: np.ndarray,
    render_details: Dict[str, Any],
    clean_nisqa_mos: float,
    pre_rir_nisqa_mos: float,
    received_nisqa_mos: float,
    ground_truth: str,
    physical_channel: Optional[PhysicalChannelState],
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    mixture_path = result_dir / "adversarial_received_mixture.wav"
    pre_rir_path = result_dir / "mixture_before_rir.wav"
    environment_path = result_dir / "optimized_environment_playback.wav"
    received_speech_path = result_dir / "received_speech_component.wav"
    received_environment_path = (
        result_dir / "received_environment_component.wav"
    )
    history_path = result_dir / "optimization_history.csv"
    result_path = result_dir / "result.json"

    write_audio(mixture_path, received_mixture)
    write_audio(pre_rir_path, pre_rir_mixture)
    write_audio(environment_path, playback_environment)
    write_audio(received_speech_path, attack.received_speech)
    write_audio(received_environment_path, received_environment)
    write_history(history_path, history)

    speech_rir_path: Optional[Path] = None
    environment_rir_path: Optional[Path] = None
    if physical_channel is not None:
        speech_rir_path = result_dir / "speech_rir_used.wav"
        environment_rir_path = result_dir / "environment_rir_used.wav"
        write_audio(speech_rir_path, physical_channel.speech_rir)
        write_audio(environment_rir_path, physical_channel.environment_rir)

    ground_truth_normalized = normalize_text(ground_truth)
    report = {
        "target_asr": TARGET_ASR,
        "data_test_root": str(args.data_test_root),
        "sample_id": args.sample_id,
        "speech_file": str(args.speech),
        "environment_file": str(args.environment),
        "reference_file": (
            None if args.reference_file is None else str(args.reference_file)
        ),
        "attack_success_definition": (
            "both_pre_rir_and_receiver_transcripts_changed_and_both_cer_"
            "values_strictly_above_min_attack_cer"
        ),
        "joint_attack_success": best.attack_success,
        "min_attack_cer": args.min_attack_cer,
        "snr_search_min_cer": args.snr_search_min_cer,
        "clean_transcript_audio_domain": "dry_original_speech",
        "clean_transcript_raw": attack.clean_text_raw,
        "clean_transcript_normalized": attack.clean_text_normalized,
        "attack_audio_domain": "receiver_side_mixture",
        "attack_transcript_raw": best.transcript_raw,
        "attack_transcript_normalized": best.transcript_normalized,
        "receiver_attack_success": (
            best.transcript_normalized != attack.clean_text_normalized
            and best.attack_cer > args.min_attack_cer
        ),
        "attack_cer": best.attack_cer,
        "attack_wer": best.attack_wer,
        "pre_rir_audio_domain": "digital_speech_environment_mixture",
        "pre_rir_transcript_raw": best.pre_rir_transcript_raw,
        "pre_rir_transcript_normalized": best.pre_rir_transcript_normalized,
        "pre_rir_attack_success": best.pre_rir_attack_success,
        "pre_rir_attack_cer": best.pre_rir_attack_cer,
        "pre_rir_attack_wer": best.pre_rir_attack_wer,
        "ground_truth_raw": ground_truth,
        "ground_truth_normalized": ground_truth_normalized,
        "ground_truth_clean_wer": (
            None
            if not ground_truth_normalized
            else float(
                wer(ground_truth_normalized, attack.clean_text_normalized)
            )
        ),
        "ground_truth_attack_cer": (
            None
            if not ground_truth_normalized
            else float(cer(ground_truth_normalized, best.transcript_normalized))
        ),
        "ground_truth_attack_wer": (
            None
            if not ground_truth_normalized
            else float(wer(ground_truth_normalized, best.transcript_normalized))
        ),
        "ground_truth_pre_rir_cer": (
            None
            if not ground_truth_normalized
            else float(
                cer(
                    ground_truth_normalized,
                    best.pre_rir_transcript_normalized,
                )
            )
        ),
        "ground_truth_pre_rir_wer": (
            None
            if not ground_truth_normalized
            else float(
                wer(
                    ground_truth_normalized,
                    best.pre_rir_transcript_normalized,
                )
            )
        ),
        "clean_nisqa_mos": clean_nisqa_mos,
        "clean_nisqa_audio_domain": "dry_original_speech",
        "pre_rir_nisqa_mos": pre_rir_nisqa_mos,
        "pre_rir_nisqa_audio_domain": (
            "digital_speech_environment_mixture_before_rir"
        ),
        "pre_rir_nisqa_change_from_clean": (
            pre_rir_nisqa_mos - clean_nisqa_mos
        ),
        "received_nisqa_mos": received_nisqa_mos,
        "received_nisqa_audio_domain": "receiver_side_mixture",
        "nisqa_change_from_clean": received_nisqa_mos - clean_nisqa_mos,
        "best_parameters": asdict(best.parameters),
        "actual_snr_db": best.actual_snr_db,
        "selection_rule": (
            "joint_pre_rir_and_receiver_success_then_max_receiver_nisqa; "
            "failures_maximize_min_cer_then_cer_sum"
        ),
        "optimization_rounds": args.rounds,
        "feature_optimization_order": list(FEATURE_OPTIMIZATION_ORDER),
        "pattern_initial_steps": dict(
            zip(FEATURE_OPTIMIZATION_ORDER, args.pattern_initial_steps)
        ),
        "pattern_min_steps": dict(
            zip(FEATURE_OPTIMIZATION_ORDER, args.pattern_min_steps)
        ),
        "optimization_candidate_evaluations": attack.evaluation_count,
        "optimization_receiver_asr_queries": attack.receiver_query_count,
        "optimization_pre_rir_asr_queries": attack.pre_rir_query_count,
        "total_asr_queries_including_clean": 1 + attack.query_count,
        "physical_channel": channel_metadata(physical_channel),
        "microphone": microphone_bandpass_metadata(
            args.microphone_bandpass
        ),
        "render_details": render_details,
        "outputs": {
            "adversarial_received_mixture": str(mixture_path.resolve()),
            "mixture_before_rir": str(pre_rir_path.resolve()),
            "optimized_environment_playback": str(environment_path.resolve()),
            "received_speech_component": str(received_speech_path.resolve()),
            "received_environment_component": str(
                received_environment_path.resolve()
            ),
            "speech_rir_used": (
                None
                if speech_rir_path is None
                else str(speech_rir_path.resolve())
            ),
            "environment_rir_used": (
                None
                if environment_rir_path is None
                else str(environment_rir_path.resolve())
            ),
            "optimization_history": str(history_path.resolve()),
        },
        "best_evaluation": best.csv_row(),
    }
    result_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


def main() -> None:
    args = parse_args()
    speaker_name, sentence_name = validate_args(args)
    result_dir = (
        args.output_dir / TARGET_ASR / speaker_name / sentence_name
    )

    speech = load_audio(args.speech)
    environment = load_audio(args.environment)
    ground_truth = read_reference(args.reference_file)
    physical_channel = build_physical_channel(args)
    microphone_sos = build_microphone_bandpass(args.microphone_bandpass)
    if physical_channel is None:
        received_length = speech.size
    else:
        tail_samples = int(
            round(args.rir_tail_seconds * TARGET_SAMPLE_RATE)
        )
        received_length = speech.size + tail_samples

    print("\n[Standalone Wav2Vec AmbientAttack]")
    print(f"  sample ID   : {args.sample_id or 'explicit_paths'}")
    print(f"  speech      : {args.speech}")
    print(f"  environment : {args.environment}")
    print(f"  transcript  : {args.reference_file}")
    print(f"  output      : {result_dir}")
    print(f"  rounds      : {args.rounds}")
    print(
        "  CER target  : both pre-RIR and receiver CER > "
        f"{args.min_attack_cer}"
    )
    print(
        "  topology    : speech channel + environment channel -> "
        "receiver mix -> microphone"
    )
    print(f"  channel     : {channel_metadata(physical_channel)}")
    print(
        "  microphone  : "
        f"{microphone_bandpass_metadata(args.microphone_bandpass)}"
    )

    if args.dry_run:
        print("  validation  : OK (models were not loaded)")
        return

    asr: Optional[Wav2VecASR] = None
    with tempfile.TemporaryDirectory(
        prefix="ambientattack_wav2vec_"
    ) as temp_name:
        temp_dir = Path(temp_name)
        try:
            asr = Wav2VecASR(
                args.wav2vec_model,
                args.wav2vec_dictionary,
                args.wav2vec_chunk_seconds,
                args.wav2vec_stride_seconds,
                args.force_cpu,
                TARGET_SAMPLE_RATE,
            )
            print("\n[ASR] Querying dry speech for the clean reference")
            clean_text_raw = asr.transcribe(speech)
            print(f"[ASR] Clean transcript: {clean_text_raw}")

            nisqa_scorer = NISQAScorer(
                args.nisqa_root,
                args.nisqa_model,
                args.speech,
                args.nisqa_device,
            )
            clean_path = temp_dir / "clean_dry_speech.wav"
            write_audio(clean_path, speech)
            clean_nisqa_mos = nisqa_scorer.score_file(clean_path)
            if not math.isfinite(clean_nisqa_mos):
                raise RuntimeError("NISQA returned a non-finite clean MOS")
            print(f"[NISQA] Clean dry MOS: {clean_nisqa_mos:.4f}")

            history: List[AttackEvaluation] = []
            attack = SingleSampleAttack(
                speech=speech,
                environment=environment,
                asr=asr,
                clean_text_raw=clean_text_raw,
                received_length=received_length,
                physical_channel=physical_channel,
                microphone_sos=microphone_sos,
                args=args,
                nisqa_scorer=nisqa_scorer,
                temp_dir=temp_dir,
            )
            best = run_sequential_feature_optimization(
                attack,
                args,
                history,
            )
            (
                received_mixture,
                pre_rir_mixture,
                playback_environment,
                received_environment,
                render_details,
            ) = attack.render(best.parameters)

            # Score the exact receiver-side waveform even if the attack fails.
            result_dir.mkdir(parents=True, exist_ok=True)
            final_mixture_path = (
                result_dir / "adversarial_received_mixture.wav"
            )
            final_pre_rir_path = result_dir / "mixture_before_rir.wav"
            write_audio(final_mixture_path, received_mixture)
            write_audio(final_pre_rir_path, pre_rir_mixture)
            received_nisqa_mos = nisqa_scorer.score_file(final_mixture_path)
            pre_rir_nisqa_mos = nisqa_scorer.score_file(final_pre_rir_path)
            if not (
                math.isfinite(received_nisqa_mos)
                and math.isfinite(pre_rir_nisqa_mos)
            ):
                raise RuntimeError("NISQA returned a non-finite final MOS")

            result_path = save_single_result(
                result_dir=result_dir,
                args=args,
                attack=attack,
                best=best,
                history=history,
                received_mixture=received_mixture,
                pre_rir_mixture=pre_rir_mixture,
                playback_environment=playback_environment,
                received_environment=received_environment,
                render_details=render_details,
                clean_nisqa_mos=clean_nisqa_mos,
                pre_rir_nisqa_mos=pre_rir_nisqa_mos,
                received_nisqa_mos=received_nisqa_mos,
                ground_truth=ground_truth,
                physical_channel=physical_channel,
            )

            print("\n[Final result]")
            print_evaluation(best, prefix="best")
            print(f"  clean MOS    : {clean_nisqa_mos:.4f}")
            print(f"  pre-RIR MOS  : {pre_rir_nisqa_mos:.4f}")
            print(f"  receiver MOS : {received_nisqa_mos:.4f}")
            print(f"  result JSON  : {result_path}")
            print(f"  pre-RIR WAV  : {final_pre_rir_path}")
            print(f"  receiver WAV : {final_mixture_path}")
        finally:
            if asr is not None:
                asr.close()
            gc.collect()


if __name__ == "__main__":
    main()
