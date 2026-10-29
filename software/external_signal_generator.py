"""
external_signal_generator.py
============================

Generates an INDEPENDENT SYNTHETIC TEST SET for the Analog Modulation
Classifier - signals that were never part of ``dataset/`` and never used to
fit, tune or select any model.

What is reused (not re-implemented)
-----------------------------------
All modulation mathematics, the AWGN model, parameter ranges and signal
validation come straight from ``signal_generator.py``:
``AMGenerator``/``FMGenerator``/``DSBSCGenerator``/``SSBGenerator`` (SSB =
upper sideband via ``scipy.signal.hilbert``), ``ParameterSampler``,
``NoiseGenerator`` and ``SignalValidator``.  Only the *sampling seed*, the
SNR selection, the output location and the metadata are new.

Independence guarantees
-----------------------
* Its own random streams, spawned from ``numpy.random.SeedSequence(seed)``
  (default seed 20261020).  The training seed (42) is rejected.
* Every signal gets freshly sampled frequencies, amplitudes, phases,
  duration and a fresh noise realisation.
* Output goes ONLY under ``external_test_signals/<run_id>/`` (never into
  ``dataset/``, ``trained_models/`` or ``results/``); nothing is copied from
  ``dataset/``.
* Files are saved flat as ``EXT-<run_id>-<index>.npy`` in shuffled order; the
  ground-truth label lives only in ``metadata.csv`` (the predictor never sees
  it, and neither file name nor folder encodes the class).

Honest limitation
-----------------
The signals share the training data's generation assumptions (same equations,
parameter ranges, AWGN).  This is an independent *synthetic* test set - not
proof of real-world generalisation.

CLI
---
    python external_signal_generator.py --per-class 250 --snr 30 20 10 5
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from signal_generator import (
    Config,
    GeneratorFactory,
    NoiseGenerator,
    ParameterSampler,
    SignalGenerationError,
    SignalParameters,
    SignalValidationError,
    SignalValidator,
)

logger = logging.getLogger("external_signal_generator")

# ==============================================================================
# CONSTANTS
# ==============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent
EXTERNAL_SIGNALS_ROOT: Path = PROJECT_ROOT / "external_test_signals"

DEFAULT_SEED: int = 20261020
TRAINING_SEED: int = 42  # seed used by signal_generator.py for the training set

MODULATION_CLASSES: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")
SUPPORTED_SNR_DB: Tuple[float, ...] = (30.0, 20.0, 10.0, 5.0)  # same set as the training data
SNR_LIMITS_DB: Tuple[float, float] = (-10.0, 50.0)
MAX_SIGNALS_PER_CLASS: int = 10_000
MIN_SAMPLES: int = 256

SIGNALS_SUBDIR: str = "signals"
METADATA_FILENAME: str = "metadata.csv"
SUMMARY_FILENAME: str = "generation_summary.json"
PROTECTED_DIRNAMES: Tuple[str, ...] = ("dataset", "trained_models", "results")

METADATA_COLUMNS: Tuple[str, ...] = (
    "signal_id", "filename", "modulation", "carrier_frequency", "message_frequency",
    "carrier_amplitude", "message_amplitude", "modulation_index", "carrier_phase",
    "message_phase", "duration", "n_samples", "sampling_frequency", "snr_db",
    "random_state", "run_id", "generated_at",
)

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


# ==============================================================================
# EXCEPTIONS / DATA CLASSES
# ==============================================================================

class ExternalGenerationError(ValueError):
    """Raised for invalid configuration, parameters or generation failures."""


class GenerationCancelled(Exception):
    """Raised when the caller requests cancellation."""

    is_cancellation = True


def _assert_not_protected(path: Path) -> None:
    """Refuses output locations inside dataset/, trained_models/ or results/."""
    resolved = path.resolve()
    for name in PROTECTED_DIRNAMES:
        protected = (PROJECT_ROOT / name).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ExternalGenerationError(
                f"Refusing to write independent test signals inside protected folder '{name}/'."
            )


@dataclass
class ExternalGenerationConfig:
    """User-facing settings for one generation run."""

    signals_per_class: int = 100
    snr_values_db: Tuple[float, ...] = SUPPORTED_SNR_DB
    random_state: int = DEFAULT_SEED
    output_root: Path = field(default_factory=lambda: EXTERNAL_SIGNALS_ROOT)
    modulation_types: Tuple[str, ...] = MODULATION_CLASSES

    def validate(self) -> None:
        """Raises :class:`ExternalGenerationError` if anything is invalid."""
        if not isinstance(self.signals_per_class, int) or not 1 <= self.signals_per_class <= MAX_SIGNALS_PER_CLASS:
            raise ExternalGenerationError(
                f"signals_per_class must be an integer in [1, {MAX_SIGNALS_PER_CLASS}]."
            )
        if not self.snr_values_db:
            raise ExternalGenerationError("At least one SNR value is required.")
        low, high = SNR_LIMITS_DB
        for snr in self.snr_values_db:
            if not np.isfinite(snr) or not low <= float(snr) <= high:
                raise ExternalGenerationError(f"SNR {snr!r} dB is outside the allowed range [{low}, {high}] dB.")
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ExternalGenerationError("random_state must be a non-negative integer.")
        if self.random_state == TRAINING_SEED:
            raise ExternalGenerationError(
                f"Seed {TRAINING_SEED} was used to generate the training dataset; choose a different seed "
                f"(default {DEFAULT_SEED})."
            )
        unknown = [m for m in self.modulation_types if m not in MODULATION_CLASSES]
        if unknown or not self.modulation_types:
            raise ExternalGenerationError(f"Unsupported modulation type(s): {unknown or 'none given'}.")
        _assert_not_protected(Path(self.output_root))


@dataclass
class GenerationResult:
    """Outcome of a successful generation run."""

    run_id: str
    run_dir: Path
    metadata_path: Path
    summary_path: Path
    n_generated: int
    class_counts: Dict[str, int]
    snr_counts: Dict[str, int]


@dataclass(frozen=True)
class RunInfo:
    """Lightweight description of a previously generated test set."""

    run_id: str
    run_dir: Path
    n_signals: int
    seed: Optional[int]
    snr_values_db: Tuple[float, ...]
    class_counts: Dict[str, int]
    created_at: str


# ==============================================================================
# PARAMETER VALIDATION
# ==============================================================================

def validate_signal_parameters(params: SignalParameters, base: Config) -> None:
    """Rejects physically invalid or aliasing parameter sets.

    Checks positivity, Nyquist compliance (including the FM Carson-rule band
    edge ``fc + (beta + 1) * fm``), positive lower band edge, modulation-index
    limits, phase ranges, minimum length and SNR range.
    """
    fs = float(params.sampling_frequency)
    if not np.isclose(fs, base.sampling_frequency):
        raise ExternalGenerationError(f"Sampling frequency {fs} Hz differs from the project value {base.sampling_frequency} Hz.")
    if params.modulation not in MODULATION_CLASSES:
        raise ExternalGenerationError(f"Unknown modulation '{params.modulation}'.")

    positives = {
        "carrier_frequency": params.carrier_frequency, "message_frequency": params.message_frequency,
        "carrier_amplitude": params.carrier_amplitude, "message_amplitude": params.message_amplitude,
        "duration": params.duration,
    }
    for name, value in positives.items():
        if not np.isfinite(value) or value <= 0:
            raise ExternalGenerationError(f"{name} must be positive and finite (got {value!r}).")

    nyquist = fs / 2.0
    fc, fm = params.carrier_frequency, params.message_frequency
    if fc <= fm:
        raise ExternalGenerationError("Carrier frequency must exceed the message frequency.")
    if params.modulation == "FM":
        if not 0.0 < params.modulation_index <= 10.0:
            raise ExternalGenerationError("FM modulation index (beta) must be in (0, 10].")
        half_band = (params.modulation_index + 1.0) * fm
    else:
        if params.modulation == "AM" and not 0.0 < params.modulation_index <= 1.0:
            raise ExternalGenerationError("AM modulation index (mu) must be in (0, 1].")
        half_band = fm
    if fc + half_band >= nyquist:
        raise ExternalGenerationError("Signal bandwidth reaches the Nyquist frequency (aliasing).")
    if fc - half_band <= 0.0:
        raise ExternalGenerationError("Signal band extends to non-positive frequency.")

    for name in ("carrier_phase", "message_phase"):
        value = getattr(params, name)
        if not np.isfinite(value) or not 0.0 <= value <= 2.0 * np.pi + 1e-9:
            raise ExternalGenerationError(f"{name} must lie in [0, 2*pi].")
    if int(round(params.duration * fs)) < MIN_SAMPLES:
        raise ExternalGenerationError(f"Signal would have fewer than {MIN_SAMPLES} samples.")
    low, high = SNR_LIMITS_DB
    if not np.isfinite(params.snr_db) or not low <= params.snr_db <= high:
        raise ExternalGenerationError("SNR outside the allowed range.")


# ==============================================================================
# GENERATION
# ==============================================================================

def _generate_one(sampler: ParameterSampler, factory: GeneratorFactory, base: Config,
                  modulation: str, snr_db: float, max_attempts: int = 25) -> Tuple[np.ndarray, SignalParameters]:
    """Samples parameters until a valid signal is produced (bounded retries)."""
    last_error: Optional[Exception] = None
    for _ in range(max_attempts):
        params = replace(sampler.sample(modulation), snr_db=float(snr_db))
        try:
            validate_signal_parameters(params, base)
            signal = factory.get(modulation).generate(params)
        except (ExternalGenerationError, SignalGenerationError, SignalValidationError) as exc:
            last_error = exc
            continue
        return signal, params
    raise ExternalGenerationError(
        f"Could not generate a valid {modulation} signal after {max_attempts} attempts: {last_error}"
    )


def _unique_run_dir(root: Path) -> Tuple[str, Path]:
    base_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id, counter = base_id, 2
    while (root / run_id).exists():
        run_id = f"{base_id}_{counter}"
        counter += 1
    return run_id, root / run_id


def generate_external_test_set(config: Optional[ExternalGenerationConfig] = None,
                               progress_cb: Optional[ProgressCallback] = None,
                               should_cancel: Optional[CancelCallback] = None) -> GenerationResult:
    """Generates a balanced independent test set and saves it to disk.

    Args:
        config: Generation settings (defaults to 100 signals/class, the four
            training SNR values, seed 20261020).
        progress_cb: Optional ``(done, total, message)`` callback.
        should_cancel: Optional callable returning True to abort; a partial run
            folder is then deleted and :class:`GenerationCancelled` is raised.

    Returns:
        A :class:`GenerationResult` describing the new run.
    """
    cfg = config or ExternalGenerationConfig()
    cfg.validate()

    base = Config(random_seed=cfg.random_state)  # used ONLY for parameter ranges; nothing is written via it
    params_seq, noise_seq, order_seq = np.random.SeedSequence(cfg.random_state).spawn(3)
    rng_params = np.random.default_rng(params_seq)
    rng_noise = np.random.default_rng(noise_seq)
    rng_order = np.random.default_rng(order_seq)

    sampler = ParameterSampler(base, rng_params)
    factory = GeneratorFactory(NoiseGenerator(rng_noise), SignalValidator(expected_dtype=base.float_dtype))

    # Balanced jobs: every class gets the same count, SNR values cycled evenly, then shuffled.
    jobs: List[Tuple[str, float]] = []
    for modulation in cfg.modulation_types:
        cycle = [float(cfg.snr_values_db[i % len(cfg.snr_values_db)]) for i in range(cfg.signals_per_class)]
        jobs.extend((modulation, snr) for snr in rng_order.permutation(cycle))
    jobs = [jobs[i] for i in rng_order.permutation(len(jobs))]
    total = len(jobs)

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = _unique_run_dir(output_root)
    signals_dir = run_dir / SIGNALS_SUBDIR
    signals_dir.mkdir(parents=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    records: List[Dict[str, Any]] = []
    try:
        for index, (modulation, snr) in enumerate(jobs, start=1):
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled("Generation cancelled by the user.")
            signal, params = _generate_one(sampler, factory, base, modulation, snr)
            signal_id = f"EXT-{run_id}-{index:05d}"
            filename = f"{signal_id}.npy"
            np.save(signals_dir / filename, signal)
            records.append({
                "signal_id": signal_id, "filename": filename, "modulation": modulation,
                "carrier_frequency": params.carrier_frequency, "message_frequency": params.message_frequency,
                "carrier_amplitude": params.carrier_amplitude, "message_amplitude": params.message_amplitude,
                "modulation_index": params.modulation_index, "carrier_phase": params.carrier_phase,
                "message_phase": params.message_phase, "duration": params.duration,
                "n_samples": int(signal.size), "sampling_frequency": params.sampling_frequency,
                "snr_db": params.snr_db, "random_state": cfg.random_state, "run_id": run_id,
                "generated_at": generated_at,
            })
            if progress_cb is not None:
                progress_cb(index, total, f"Generating signal {index} of {total}")
    except BaseException:
        # Remove only the run folder created by THIS call.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    metadata = pd.DataFrame(records, columns=list(METADATA_COLUMNS))
    metadata_path = run_dir / METADATA_FILENAME
    metadata.to_csv(metadata_path, index=False)

    class_counts = {k: int(v) for k, v in metadata["modulation"].value_counts().items()}
    snr_counts = {str(k): int(v) for k, v in metadata["snr_db"].value_counts().sort_index().items()}
    summary = {
        "run_id": run_id,
        "created_at": generated_at,
        "random_state": cfg.random_state,
        "training_seed_excluded": TRAINING_SEED,
        "rng_streams": "numpy SeedSequence(random_state).spawn(3): parameters, noise, ordering",
        "sampling_frequency": base.sampling_frequency,
        "signals_per_class": cfg.signals_per_class,
        "snr_values_db": [float(s) for s in cfg.snr_values_db],
        "total_signals": int(len(metadata)),
        "class_counts": class_counts,
        "snr_counts": snr_counts,
        "parameter_ranges": {
            "carrier_freq_range": list(base.carrier_freq_range),
            "message_freq_range": list(base.message_freq_range),
            "carrier_amplitude_range": list(base.carrier_amplitude_range),
            "message_amplitude_range": list(base.message_amplitude_range),
            "modulation_index_range": list(base.modulation_index_range),
            "duration_range": list(base.duration_range),
        },
        "generator": "signal_generator.py modulators (AM, FM, DSB_SC, SSB upper sideband via Hilbert transform)",
        "independence_note": (
            "Independent SYNTHETIC test set: new seed, newly sampled parameters, phases and noise, never used "
            "for training/tuning/model selection. Shares the training data's generation assumptions, so it is "
            "not evidence of real-world generalisation."
        ),
    }
    summary_path = run_dir / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Generated %d independent signals in '%s'.", len(metadata), run_dir)
    return GenerationResult(run_id, run_dir, metadata_path, summary_path, len(metadata), class_counts, snr_counts)


# ==============================================================================
# RUN DISCOVERY / LOADING (used by the GUI and the evaluator)
# ==============================================================================

def load_generation_summary(run_dir: Path) -> Dict[str, Any]:
    """Returns ``generation_summary.json`` for a run (empty dict if absent/unreadable)."""
    path = Path(run_dir) / SUMMARY_FILENAME
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_test_runs(root: Path = EXTERNAL_SIGNALS_ROOT) -> List[RunInfo]:
    """Lists generated test sets (newest first) that contain a metadata file."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs: List[RunInfo] = []
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        if not (run_dir / METADATA_FILENAME).is_file():
            continue
        summary = load_generation_summary(run_dir)
        runs.append(RunInfo(
            run_id=run_dir.name,
            run_dir=run_dir,
            n_signals=int(summary.get("total_signals", 0)),
            seed=summary.get("random_state"),
            snr_values_db=tuple(float(s) for s in summary.get("snr_values_db", [])),
            class_counts={str(k): int(v) for k, v in summary.get("class_counts", {}).items()},
            created_at=str(summary.get("created_at", "")),
        ))
    return runs


def load_run_metadata(run_dir: Path) -> pd.DataFrame:
    """Loads and validates a run's ``metadata.csv`` (ground truth lives here)."""
    path = Path(run_dir) / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    frame = pd.read_csv(path, dtype={"signal_id": str, "filename": str, "modulation": str})
    required = {"signal_id", "filename", "modulation", "snr_db"}
    missing = required - set(frame.columns)
    if missing:
        raise ExternalGenerationError(f"{path.name} is missing required column(s): {sorted(missing)}")
    unknown = sorted(set(frame["modulation"]) - set(MODULATION_CLASSES))
    if unknown:
        raise ExternalGenerationError(f"{path.name} contains unknown class label(s): {unknown}")
    if frame.empty:
        raise ExternalGenerationError(f"{path.name} contains no signals.")
    return frame


def run_signal_path(run_dir: Path, filename: str) -> Path:
    """Path of a signal file inside a run (rejects path traversal)."""
    if filename != Path(filename).name:
        raise ExternalGenerationError(f"Invalid signal filename: {filename!r}")
    return Path(run_dir) / SIGNALS_SUBDIR / filename


# ==============================================================================
# CLI
# ==============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an independent synthetic test set.")
    parser.add_argument("--per-class", type=int, default=100, help="signals per class (default 100)")
    parser.add_argument("--snr", type=float, nargs="+", default=list(SUPPORTED_SNR_DB), help="SNR value(s) in dB")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    args = parser.parse_args(argv)

    cfg = ExternalGenerationConfig(signals_per_class=args.per_class, snr_values_db=tuple(args.snr),
                                   random_state=args.seed)
    try:
        result = generate_external_test_set(
            cfg, progress_cb=lambda d, t, m: print(f"\r{m}", end="", flush=True) if d % 25 == 0 or d == t else None)
    except ExternalGenerationError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    print(f"\nRun id      : {result.run_id}")
    print(f"Signals     : {result.n_generated}  {result.class_counts}")
    print(f"SNR counts  : {result.snr_counts}")
    print(f"Folder      : {result.run_dir}")
    print(f"Ground truth: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())