"""
feature_extraction.py
======================

Analog Modulation Classifier Using Machine Learning
-----------------------------------------------------

This module is responsible EXCLUSIVELY for:

    1. Loading raw ``.npy`` signals produced by ``signal_generator.py``.
    2. Preprocessing them using the existing, unmodified
       ``preprocessing.py`` pipeline (``SignalPreprocessor`` /
       ``PreprocessingConfig``).
    3. Extracting exactly 12 numerical DSP features from each
       preprocessed signal.
    4. Assembling a labeled feature dataframe for the entire dataset.
    5. Persisting the result as ``dataset/features.csv``.

It does NOT train, evaluate, or predict with any machine learning
model. Those responsibilities belong to separate downstream modules
(``train_models.py``, ``evaluate_models.py``, ``predict.py``).

Pipeline stage handled here::

    dataset/<MODULATION>/*.npy   (produced by signal_generator.py)
                |
                v
        SignalPreprocessor.process()   (preprocessing.py, reused as-is)
                |
                v
        processed 1-D float64 NumPy array (variable length per signal)
                |
                v
        FeatureExtractor.extract_features()
                |
                v
        12 numerical DSP features + Label
                |
                v
        dataset/features.csv

Compatibility notes (derived from signal_generator.py / preprocessing.py):
    * Sampling frequency: 20,000 Hz.
    * Signal durations vary between 0.05 s and 0.20 s, so signal
      lengths are NOT constant across the dataset. Every feature
      function below operates correctly on arbitrary valid signal
      lengths -- no resampling, cropping, or padding is ever performed.
    * Dataset layout: dataset/{AM,FM,DSB_SC,SSB}/<MODULATION>_####.npy
      plus dataset/metadata.csv, dataset/run_summary.json, and
      dataset/plots/, none of which are modified by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from tqdm import tqdm

from preprocessing import PreprocessingConfig, SignalPreprocessor, SignalValidationError

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================


def _build_logger(name: str = "feature_extraction") -> logging.Logger:
    """Builds and returns a configured module-level logger.

    Args:
        name: Name to assign to the logger instance.

    Returns:
        A fully configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


logger = _build_logger()


# ==============================================================================
# CONSTANTS
# ==============================================================================

#: Canonical, deterministic ordering of the 12 DSP feature column names.
FEATURE_NAMES: Tuple[str, ...] = (
    "RMS",
    "Energy",
    "FFT Peak",
    "Centroid",
    "Bandwidth",
    "Entropy",
    "Instantaneous Frequency",
    "Zero Crossing Rate",
    "Std",
    "Kurtosis",
    "Skewness",
    "Crest Factor",
)

#: Final dataframe column ordering, including the label column.
DATAFRAME_COLUMNS: Tuple[str, ...] = FEATURE_NAMES + ("Label",)

#: Modulation classes produced by signal_generator.py (also the class
#: sub-directory names under dataset/).
MODULATION_CLASSES: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")

#: Default output filename for the extracted feature dataframe.
FEATURES_FILENAME = "features.csv"


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================


class FeatureExtractionError(Exception):
    """Raised when a feature (or set of features) cannot be computed for a
    given signal."""


class DatasetDiscoveryError(Exception):
    """Raised when the dataset directory structure cannot be discovered or
    is missing expected contents."""


# ==============================================================================
# INDIVIDUAL FEATURE FUNCTIONS
# ==============================================================================
#
# All 12 functions below operate on a single preprocessed 1-D float64
# signal (and, where relevant, the known sampling frequency). None of
# them resample, crop, or pad the input -- they work directly with
# whatever number of samples the preprocessed signal contains.


def calculate_rms(signal: np.ndarray) -> float:
    """Calculates the Root Mean Square (RMS) value of a signal.

    Formula::

        RMS = sqrt(mean(x[n]^2))

    Args:
        signal: 1-D preprocessed float64 signal.

    Returns:
        The RMS value as a Python float.
    """
    return float(np.sqrt(np.mean(np.square(signal))))


def calculate_energy(signal: np.ndarray) -> float:
    """Calculates the total discrete-time signal energy.

    Formula (real-valued signal)::

        E = sum(x[n]^2)

    This is NOT average power and is NOT normalized by signal length --
    signals in this dataset have variable durations, so energy is
    computed directly from the raw sample values.

    Args:
        signal: 1-D preprocessed float64 signal.

    Returns:
        The total signal energy as a Python float.
    """
    return float(np.sum(np.square(signal)))


def _one_sided_spectrum(signal: np.ndarray, sampling_frequency: float) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the one-sided FFT frequency axis and magnitude spectrum.

    Uses ``numpy.fft.rfft`` / ``numpy.fft.rfftfreq`` since all signals in
    this project are real-valued, avoiding redundant negative-frequency
    computation.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        A tuple ``(freqs, magnitude)`` where ``freqs`` are the one-sided
        FFT bin frequencies (Hz) and ``magnitude`` is the corresponding
        non-negative magnitude spectrum ``|X(f)|``.
    """
    n = signal.shape[0]
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_frequency)
    magnitude = np.abs(spectrum)
    return freqs, magnitude


def calculate_fft_peak_frequency(signal: np.ndarray, sampling_frequency: float) -> float:
    """Finds the frequency (Hz) at which the FFT magnitude spectrum peaks.

    Uses the one-sided spectrum (``numpy.fft.rfft`` / ``rfftfreq``) so
    that only non-negative frequencies are considered -- the negative-
    frequency mirror is never returned as the peak.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        The frequency (Hz) corresponding to the maximum magnitude bin.
    """
    freqs, magnitude = _one_sided_spectrum(signal, sampling_frequency)
    peak_index = int(np.argmax(magnitude))
    return float(freqs[peak_index])


def calculate_spectral_centroid(signal: np.ndarray, sampling_frequency: float) -> float:
    """Calculates the spectral centroid (the "center of mass" of the spectrum).

    Formula (magnitude-weighted, one-sided spectrum)::

        Centroid = sum(f[k] * M[k]) / sum(M[k])

    If the total magnitude is zero (degenerate case), 0.0 Hz is
    returned instead of NaN.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        The spectral centroid in Hz.
    """
    freqs, magnitude = _one_sided_spectrum(signal, sampling_frequency)
    total_magnitude = np.sum(magnitude)

    if total_magnitude <= 0.0 or not np.isfinite(total_magnitude):
        return 0.0

    centroid = np.sum(freqs * magnitude) / total_magnitude
    return float(centroid)


def calculate_bandwidth(signal: np.ndarray, sampling_frequency: float) -> float:
    """Calculates the spectral bandwidth as the second central spectral moment.

    Uses the one-sided power spectrum ``P[k] = M[k]^2`` (where ``M[k]``
    is the one-sided FFT magnitude spectrum) together with the
    magnitude-weighted spectral centroid from
    :func:`calculate_spectral_centroid`.

    Formula::

        Bandwidth = sqrt( sum(P[k] * (f[k] - centroid)^2) / sum(P[k]) )

    This is the standard power-weighted spectral spread around the
    centroid, NOT a simple ``max_frequency - min_frequency`` range and
    NOT a percentage-bandwidth approximation.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        The spectral bandwidth in Hz.
    """
    freqs, magnitude = _one_sided_spectrum(signal, sampling_frequency)
    power = np.square(magnitude)
    total_power = np.sum(power)

    if total_power <= 0.0 or not np.isfinite(total_power):
        return 0.0

    centroid = calculate_spectral_centroid(signal, sampling_frequency)
    variance = np.sum(power * np.square(freqs - centroid)) / total_power
    return float(np.sqrt(max(variance, 0.0)))


def calculate_spectral_entropy(signal: np.ndarray, sampling_frequency: float) -> float:
    """Calculates the Shannon spectral entropy (in bits) of a signal.

    The one-sided power spectrum ``P[k] = |X(f_k)|^2`` is normalized
    into a probability-like distribution::

        p[k] = P[k] / sum(P[k])

    and the standard Shannon entropy is computed::

        H = -sum(p[k] * log2(p[k]))

    Zero-probability bins are excluded from the summation to avoid
    ``log2(0)`` producing NaN. No additional normalization by
    ``log2(N)`` is applied -- this is the raw Shannon entropy in bits.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        The Shannon spectral entropy in bits.
    """
    _, magnitude = _one_sided_spectrum(signal, sampling_frequency)
    power = np.square(magnitude)
    total_power = np.sum(power)

    if total_power <= 0.0 or not np.isfinite(total_power):
        return 0.0

    probabilities = power / total_power
    nonzero_probabilities = probabilities[probabilities > 0.0]

    entropy = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))
    return float(entropy)


def calculate_instantaneous_frequency(signal: np.ndarray, sampling_frequency: float) -> float:
    """Calculates the mean instantaneous frequency via the analytic signal.

    Follows the standard Hilbert-transform-based procedure:

        1. Compute the analytic signal ``z(t) = hilbert(x(t))``.
        2. Compute the unwrapped instantaneous phase
           ``phase(t) = unwrap(angle(z(t)))``.
        3. Differentiate the phase numerically (``numpy.gradient``) and
           scale by ``fs / (2*pi)`` to obtain the instantaneous
           frequency sequence ``f_inst(t)``.
        4. Reduce the sequence to a single scalar feature by taking its
           mean.

    This is distinct from :func:`calculate_fft_peak_frequency` -- it
    reflects the time-varying phase evolution of the signal rather than
    the dominant frequency bin of a global FFT.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        The mean instantaneous frequency in Hz.
    """
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # d(phase)/dt via numerical gradient, using the sample spacing (1/fs)
    # so the result is already in radians/second.
    dt = 1.0 / sampling_frequency
    phase_derivative = np.gradient(instantaneous_phase, dt)

    instantaneous_frequency = phase_derivative / (2.0 * np.pi)

    finite_values = instantaneous_frequency[np.isfinite(instantaneous_frequency)]
    if finite_values.size == 0:
        return 0.0

    return float(np.mean(finite_values))


def calculate_zero_crossing_rate(signal: np.ndarray, sampling_frequency: float) -> float:
    """Calculates the zero crossing rate (crossings per second).

    A zero crossing is counted whenever adjacent samples lie on
    opposite sides of zero. The sign of a sample is determined using
    ``numpy.signbit`` (True for negative values, False for zero and
    positive values), which treats exact zero as non-negative. This
    avoids double-counting a signal that briefly touches exactly zero
    as two separate crossings.

    Definition::

        ZCR = (number of sign changes between adjacent samples) / duration

    where ``duration = N / sampling_frequency`` and ``N`` is the number
    of samples. The rate (not the raw count) is returned so that
    signals of different lengths/durations remain comparable.

    Args:
        signal: 1-D preprocessed float64 signal.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        The zero crossing rate in crossings per second.
    """
    n = signal.shape[0]
    if n < 2:
        return 0.0

    sign_bits = np.signbit(signal)
    crossings = np.count_nonzero(np.diff(sign_bits))

    duration = n / sampling_frequency
    if duration <= 0.0:
        return 0.0

    return float(crossings / duration)


def calculate_standard_deviation(signal: np.ndarray) -> float:
    """Calculates the standard deviation of the signal amplitude.

    Uses ``numpy.std`` (population standard deviation, ``ddof=0``).

    Args:
        signal: 1-D preprocessed float64 signal.

    Returns:
        The standard deviation as a Python float.
    """
    return float(np.std(signal))


def calculate_kurtosis(signal: np.ndarray) -> float:
    """Calculates the statistical (excess) kurtosis of the signal.

    Uses ``scipy.stats.kurtosis`` with ``fisher=True`` (so that
    Gaussian-distributed data has approximately zero excess kurtosis)
    and ``bias=False`` (bias-corrected estimator). Falls back to the
    biased estimator for very short signals where the bias-corrected
    formula is undefined.

    Args:
        signal: 1-D preprocessed float64 signal.

    Returns:
        The excess kurtosis as a Python float.
    """
    if signal.shape[0] < 4:
        # The bias-corrected kurtosis estimator is undefined for very
        # short signals; fall back to the biased estimator rather than
        # letting NaN propagate into the dataset.
        value = scipy_kurtosis(signal, fisher=True, bias=True)
    else:
        value = scipy_kurtosis(signal, fisher=True, bias=False)

    if not np.isfinite(value):
        value = scipy_kurtosis(signal, fisher=True, bias=True)

    return float(value)


def calculate_skewness(signal: np.ndarray) -> float:
    """Calculates the statistical skewness of the signal.

    Uses ``scipy.stats.skew`` with ``bias=False`` (bias-corrected
    estimator), falling back to the biased estimator for very short or
    numerically degenerate (e.g. constant) signals so that NaN never
    silently enters the final feature dataframe.

    Args:
        signal: 1-D preprocessed float64 signal.

    Returns:
        The skewness as a Python float.
    """
    if signal.shape[0] < 3:
        value = scipy_skew(signal, bias=True)
    else:
        value = scipy_skew(signal, bias=False)

    if not np.isfinite(value):
        value = scipy_skew(signal, bias=True)

    if not np.isfinite(value):
        value = 0.0

    return float(value)


def calculate_crest_factor(signal: np.ndarray) -> float:
    """Calculates the crest factor of the signal.

    Formula::

        Crest Factor = max(abs(x[n])) / RMS

    Args:
        signal: 1-D preprocessed float64 signal.

    Returns:
        The crest factor as a Python float. Returns 0.0 in the
        degenerate case of zero RMS instead of raising a
        divide-by-zero error.
    """
    rms = calculate_rms(signal)
    if rms <= 0.0:
        return 0.0

    peak_amplitude = float(np.max(np.abs(signal)))
    return float(peak_amplitude / rms)


# ==============================================================================
# FEATURE EXTRACTOR
# ==============================================================================


class FeatureExtractor:
    """Extracts the 12 required DSP features from preprocessed signals.

    Wraps the existing, unmodified :class:`preprocessing.SignalPreprocessor`
    so that every feature calculation operates on the SAME preprocessed
    signal version -- no feature is computed from a separately
    preprocessed or raw variant of the signal.
    """

    def __init__(
        self,
        sampling_frequency: float = 20000.0,
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> None:
        """Initializes the feature extractor.

        Args:
            sampling_frequency: Sampling frequency (Hz) shared by every
                signal in the dataset. Must match the value used by
                ``signal_generator.py`` (20,000 Hz by default).
            preprocessing_config: Optional :class:`PreprocessingConfig`
                to control DC removal, filtering, and normalization. If
                omitted, the default configuration from
                ``preprocessing.py`` is used (remove_dc=True,
                normalize=True, apply_filter=False).
        """
        self.sampling_frequency = float(sampling_frequency)
        self.preprocessing_config = preprocessing_config or PreprocessingConfig(
            sampling_frequency=self.sampling_frequency
        )
        self.preprocessor = SignalPreprocessor(self.preprocessing_config)

        logger.info(
            "FeatureExtractor initialized (fs=%.1f Hz, feature_count=%d).",
            self.sampling_frequency,
            len(FEATURE_NAMES),
        )

    def preprocess(self, raw_signal: np.ndarray) -> np.ndarray:
        """Preprocesses a raw signal using the shared ``SignalPreprocessor``.

        Args:
            raw_signal: The raw signal array (as loaded from ``.npy``).

        Returns:
            The preprocessed 1-D float64 signal.
        """
        return self.preprocessor.process(raw_signal)

    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extracts all 12 DSP features from an ALREADY-PREPROCESSED signal.

        All 12 feature functions are evaluated against the exact same
        input array, guaranteeing internal consistency.

        Args:
            signal: 1-D preprocessed float64 signal (i.e. the output of
                :meth:`preprocess`).

        Returns:
            An ordered dictionary mapping each of the 12 canonical
            feature names (see :data:`FEATURE_NAMES`) to its computed
            float value.

        Raises:
            FeatureExtractionError: If any feature cannot be computed or
                yields a non-finite value.
        """
        fs = self.sampling_frequency

        try:
            features = {
                "RMS": calculate_rms(signal),
                "Energy": calculate_energy(signal),
                "FFT Peak": calculate_fft_peak_frequency(signal, fs),
                "Centroid": calculate_spectral_centroid(signal, fs),
                "Bandwidth": calculate_bandwidth(signal, fs),
                "Entropy": calculate_spectral_entropy(signal, fs),
                "Instantaneous Frequency": calculate_instantaneous_frequency(signal, fs),
                "Zero Crossing Rate": calculate_zero_crossing_rate(signal, fs),
                "Std": calculate_standard_deviation(signal),
                "Kurtosis": calculate_kurtosis(signal),
                "Skewness": calculate_skewness(signal),
                "Crest Factor": calculate_crest_factor(signal),
            }
        except Exception as exc:  # pragma: no cover - defensive re-raise
            raise FeatureExtractionError(f"Failed to compute features: {exc}") from exc

        self._validate_features(features)
        return features

    def extract_from_raw(self, raw_signal: np.ndarray) -> Dict[str, float]:
        """Convenience method: preprocesses a raw signal and extracts features.

        Args:
            raw_signal: The raw signal array (as loaded from ``.npy``).

        Returns:
            The 12-feature dictionary, as returned by
            :meth:`extract_features`.
        """
        processed = self.preprocess(raw_signal)
        return self.extract_features(processed)

    @staticmethod
    def _validate_features(features: Dict[str, float]) -> None:
        """Ensures every extracted feature is present and finite.

        Args:
            features: The feature dictionary to validate.

        Raises:
            FeatureExtractionError: If a feature is missing or
                non-finite (NaN or +/-Inf).
        """
        for name in FEATURE_NAMES:
            if name not in features:
                raise FeatureExtractionError(f"Missing required feature '{name}'.")

            value = features[name]
            if not np.isfinite(value):
                raise FeatureExtractionError(
                    f"Feature '{name}' produced a non-finite value: {value!r}."
                )


# ==============================================================================
# DATASET DISCOVERY
# ==============================================================================


@dataclass
class SignalFileRecord:
    """A discovered raw signal file paired with its ground-truth label.

    Attributes:
        path: Full filesystem path to the ``.npy`` signal file.
        label: Modulation class label derived from the parent directory
            (one of ``MODULATION_CLASSES``).
    """

    path: Path
    label: str


def discover_signal_files(dataset_dir: Path) -> List[SignalFileRecord]:
    """Discovers all raw signal ``.npy`` files belonging to known classes.

    Only files located directly inside one of the known class
    sub-directories (``AM``, ``FM``, ``DSB_SC``, ``SSB``) are
    considered. This deliberately excludes ``dataset/plots/`` and any
    other non-signal content (e.g. a stray ``dataset/features.csv``)
    even though a naive recursive ``*.npy`` glob over the whole dataset
    root would otherwise be safe here too, since ``plots/`` only ever
    contains ``.png`` files.

    Args:
        dataset_dir: Root dataset directory (e.g. ``Path("dataset")``).

    Returns:
        A sorted list of :class:`SignalFileRecord` entries.

    Raises:
        DatasetDiscoveryError: If ``dataset_dir`` does not exist, or if
            no signal files are found in any known class directory.
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise DatasetDiscoveryError(f"Dataset directory not found: '{dataset_dir}'.")

    records: List[SignalFileRecord] = []

    for modulation in MODULATION_CLASSES:
        class_dir = dataset_dir / modulation
        if not class_dir.exists():
            logger.warning("Expected class directory '%s' not found; skipping.", class_dir)
            continue

        npy_files = sorted(class_dir.glob("*.npy"))
        logger.info("Discovered %d signal file(s) in '%s'.", len(npy_files), class_dir)

        records.extend(SignalFileRecord(path=f, label=modulation) for f in npy_files)

    if not records:
        raise DatasetDiscoveryError(
            f"No '.npy' signal files found under any of {MODULATION_CLASSES} "
            f"in '{dataset_dir}'."
        )

    logger.info("Total signal files discovered: %d.", len(records))
    return records


# ==============================================================================
# DATASET-LEVEL FEATURE EXTRACTION
# ==============================================================================


def extract_dataset_features(
    dataset_dir: Path,
    extractor: FeatureExtractor,
) -> Tuple[pd.DataFrame, int, int, List[str]]:
    """Extracts DSP features for every signal file in the dataset.

    Each signal is loaded, preprocessed (via ``preprocessing.py``, reused
    unmodified), and passed through all 12 feature functions. Signals
    that fail to load, preprocess, or yield a non-finite feature are
    logged and skipped -- they do NOT halt processing of the remaining
    dataset, and they are never inserted as NaN rows.

    Args:
        dataset_dir: Root dataset directory containing the class
            sub-directories.
        extractor: A configured :class:`FeatureExtractor` instance.

    Returns:
        A tuple ``(dataframe, success_count, failure_count, failed_files)``.

    Raises:
        DatasetDiscoveryError: If no signal files can be found.
    """
    records = discover_signal_files(dataset_dir)

    rows: List[Dict[str, object]] = []
    failed_files: List[str] = []

    progress = tqdm(records, desc="Extracting features", unit="signal")
    for record in progress:
        try:
            raw_signal = np.load(record.path)
            processed_signal = extractor.preprocess(raw_signal)
            features = extractor.extract_features(processed_signal)
        except (SignalValidationError, FeatureExtractionError, OSError, ValueError) as exc:
            logger.error("Failed to process '%s': %s", record.path, exc)
            failed_files.append(str(record.path))
            continue

        row = {name: features[name] for name in FEATURE_NAMES}
        row["Label"] = record.label
        rows.append(row)

        progress.set_postfix(ok=len(rows), failed=len(failed_files))

    dataframe = pd.DataFrame(rows, columns=list(DATAFRAME_COLUMNS))

    success_count = len(rows)
    failure_count = len(failed_files)

    logger.info(
        "Feature extraction complete: %d succeeded, %d failed (out of %d files).",
        success_count,
        failure_count,
        len(records),
    )

    return dataframe, success_count, failure_count, failed_files


def save_features(dataframe: pd.DataFrame, dataset_dir: Path) -> Path:
    """Saves the extracted feature dataframe as ``dataset/features.csv``.

    Args:
        dataframe: The feature dataframe to save. Must contain exactly
            the 12 feature columns plus ``Label``.
        dataset_dir: Root dataset directory to save the CSV into.

    Returns:
        The path the CSV file was written to.

    Raises:
        FeatureExtractionError: If the dataframe contains NaN or
            infinite values, or does not have the expected columns.
    """
    if list(dataframe.columns) != list(DATAFRAME_COLUMNS):
        raise FeatureExtractionError(
            f"Unexpected dataframe columns: {list(dataframe.columns)}; "
            f"expected {list(DATAFRAME_COLUMNS)}."
        )

    numeric_columns = dataframe[list(FEATURE_NAMES)]
    if numeric_columns.isnull().values.any():
        raise FeatureExtractionError("Feature dataframe contains NaN values; refusing to save.")

    if not np.all(np.isfinite(numeric_columns.to_numpy(dtype=np.float64))):
        raise FeatureExtractionError("Feature dataframe contains non-finite values; refusing to save.")

    output_path = dataset_dir / FEATURES_FILENAME
    dataframe.to_csv(output_path, index=False)

    logger.info("Saved %d feature rows (%d columns) to '%s'.", len(dataframe), len(dataframe.columns), output_path)
    return output_path


# ==============================================================================
# SAFETY TEST (SINGLE SIGNAL) — RUN BEFORE FULL DATASET EXTRACTION
# ==============================================================================


def run_single_signal_sanity_check(dataset_dir: Path, extractor: FeatureExtractor) -> bool:
    """Runs a safe sanity check on a single AM signal before bulk processing.

    Loads one AM ``.npy`` file (without modifying it), preprocesses it,
    extracts all 12 features, and verifies that exactly 12 finite
    numerical values are returned under the correct feature names.

    Args:
        dataset_dir: Root dataset directory.
        extractor: A configured :class:`FeatureExtractor` instance.

    Returns:
        True if the sanity check passed, False otherwise.
    """
    am_dir = dataset_dir / "AM"
    if not am_dir.exists():
        logger.warning("No 'AM' directory found under '%s'; skipping sanity check.", dataset_dir)
        return False

    sample_files = sorted(am_dir.glob("*.npy"))
    if not sample_files:
        logger.warning("No '.npy' files found in '%s'; skipping sanity check.", am_dir)
        return False

    sample_path = sample_files[0]
    logger.info("Running single-signal sanity check on '%s'.", sample_path)

    try:
        raw_signal = np.load(sample_path)
        processed_signal = extractor.preprocess(raw_signal)
        features = extractor.extract_features(processed_signal)
    except (SignalValidationError, FeatureExtractionError, OSError, ValueError) as exc:
        logger.error("Sanity check FAILED for '%s': %s", sample_path, exc)
        return False

    if set(features.keys()) != set(FEATURE_NAMES):
        logger.error(
            "Sanity check FAILED: feature name mismatch. Expected %s, got %s.",
            FEATURE_NAMES,
            tuple(features.keys()),
        )
        return False

    if len(features) != 12:
        logger.error("Sanity check FAILED: expected exactly 12 features, got %d.", len(features))
        return False

    if not all(np.isfinite(v) for v in features.values()):
        logger.error("Sanity check FAILED: non-finite feature value(s) detected: %s", features)
        return False

    logger.info("Sanity check PASSED for '%s'. Features: %s", sample_path, features)
    return True


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main() -> None:
    """Standalone entry point for full-dataset feature extraction.

    Steps performed:
        1. Verify that ``dataset/`` exists.
        2. Run a small sanity check on a single AM signal.
        3. Extract features for the entire dataset.
        4. Save the resulting dataframe to ``dataset/features.csv``.
        5. Log a concise summary.

    Does NOT regenerate the signal dataset, and does NOT modify raw
    ``.npy`` files or ``metadata.csv``.
    """
    dataset_dir = Path("dataset")

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        logger.error("Dataset directory '%s' does not exist. Run signal_generator.py first.", dataset_dir)
        return

    extractor = FeatureExtractor(
        sampling_frequency=20000.0,
        preprocessing_config=PreprocessingConfig(sampling_frequency=20000.0),
    )

    sanity_passed = run_single_signal_sanity_check(dataset_dir, extractor)
    if not sanity_passed:
        logger.error("Single-signal sanity check failed; aborting full dataset extraction.")
        return

    try:
        dataframe, success_count, failure_count, failed_files = extract_dataset_features(
            dataset_dir, extractor
        )
    except DatasetDiscoveryError as exc:
        logger.error("Dataset discovery failed: %s", exc)
        return

    if dataframe.empty:
        logger.error("No signals were successfully processed; features.csv will not be written.")
        return

    output_path = save_features(dataframe, dataset_dir)

    logger.info(
        "Summary: %d signals succeeded, %d failed, dataframe shape=%s, output='%s'.",
        success_count,
        failure_count,
        dataframe.shape,
        output_path,
    )

    if failed_files:
        logger.warning("Failed files (%d): %s", len(failed_files), failed_files)


if __name__ == "__main__":
    main()