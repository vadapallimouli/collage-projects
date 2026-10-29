"""
preprocessing.py
=================

Analog Modulation Classifier Using Machine Learning
-----------------------------------------------------

This module is responsible EXCLUSIVELY for signal preprocessing (DSP-level
cleanup) of the analog-modulated signals produced by ``signal_generator.py``.

It does NOT perform feature extraction, model training, evaluation, or
prediction -- those responsibilities belong to separate downstream modules
(``feature_extraction.py``, ``train_models.py``, ``predict.py``).

Pipeline stage handled here::

    dataset/<MODULATION>/*.npy   (produced by signal_generator.py)
                |
                v
        SignalPreprocessor.process()
                |
                v
        processed 1-D float64 NumPy array (same length as input)
                |
                v
        feature_extraction.py   (NOT implemented here)

Compatibility notes (derived from signal_generator.py):
    * Sampling frequency: 20,000 Hz.
    * Signal durations vary between 0.05 s and 0.20 s, so signal lengths
      are NOT constant across the dataset -- preprocessing must never
      resample, crop, or pad a signal.
    * Carrier frequencies range from ~1,000 Hz to 5,000 Hz.
    * Message frequencies range from ~50 Hz to 500 Hz.
    * Dataset layout: dataset/{AM,FM,DSB_SC,SSB}/<MODULATION>_####.npy
      plus dataset/metadata.csv and dataset/run_summary.json, neither of
      which is read, written, or modified by this module.

The same ``SignalPreprocessor`` / ``PreprocessingConfig`` pair is meant to
be reused, unmodified, by both the training pipeline and the later
unknown-signal prediction pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

FilterCutoff = Union[float, Tuple[float, float]]


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def _build_logger(name: str = "preprocessing") -> logging.Logger:
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
# CUSTOM EXCEPTIONS
# ==============================================================================

class SignalValidationError(Exception):
    """Raised when a signal fails one or more validation checks."""


class PreprocessingConfigError(Exception):
    """Raised when a :class:`PreprocessingConfig` fails validation."""


class FilterDesignError(Exception):
    """Raised when a Butterworth filter cannot be designed for the given
    configuration (e.g. invalid or non-normalizable cutoff frequencies)."""


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for the signal preprocessing pipeline.

    Attributes:
        sampling_frequency: Sampling frequency in Hz. Must match the
            sampling frequency used by ``signal_generator.py`` (20,000 Hz
            by default) so that filter cutoffs are interpreted correctly.
        remove_dc: Whether to remove the DC offset (mean) from the signal.
        normalize: Whether to normalize the signal to unit maximum
            absolute amplitude.
        apply_filter: Whether to apply an optional Butterworth filter.
            Disabled by default so that modulation information (carrier
            frequencies up to ~5,000 Hz, sidebands, etc.) is never
            accidentally destroyed.
        filter_type: One of ``"lowpass"``, ``"highpass"``, ``"bandpass"``.
        cutoff_frequency: Cutoff frequency (Hz) for lowpass/highpass
            filters (a single float), or a ``(low, high)`` tuple in Hz
            for bandpass filters. Defaults to a wide bandpass window
            (20 Hz - 8,000 Hz) chosen to sit safely below the Nyquist
            frequency (10,000 Hz) while comfortably spanning the
            generator's carrier range (1,000 Hz - 5,000 Hz) and message
            range (50 Hz - 500 Hz), including SSB/DSB-SC sidebands.
        filter_order: Butterworth filter order (per stage, before SOS
            conversion doubles the effective order via zero-phase
            filtering).
    """

    sampling_frequency: float = 20000.0
    remove_dc: bool = True
    normalize: bool = True
    apply_filter: bool = False
    filter_type: str = "bandpass"
    cutoff_frequency: FilterCutoff = field(default_factory=lambda: (20.0, 8000.0))
    filter_order: int = 4

    _VALID_FILTER_TYPES = ("lowpass", "highpass", "bandpass")

    def __post_init__(self) -> None:
        """Validates the configuration immediately upon construction.

        Raises:
            PreprocessingConfigError: If any field holds an invalid value.
        """
        if not isinstance(self.sampling_frequency, (int, float)) or self.sampling_frequency <= 0:
            raise PreprocessingConfigError(
                f"sampling_frequency must be a positive number, got {self.sampling_frequency!r}."
            )

        if not isinstance(self.remove_dc, bool):
            raise PreprocessingConfigError("remove_dc must be a boolean.")

        if not isinstance(self.normalize, bool):
            raise PreprocessingConfigError("normalize must be a boolean.")

        if not isinstance(self.apply_filter, bool):
            raise PreprocessingConfigError("apply_filter must be a boolean.")

        if self.filter_type not in self._VALID_FILTER_TYPES:
            raise PreprocessingConfigError(
                f"filter_type must be one of {self._VALID_FILTER_TYPES}, "
                f"got '{self.filter_type}'."
            )

        if not isinstance(self.filter_order, int) or self.filter_order <= 0:
            raise PreprocessingConfigError(
                f"filter_order must be a positive integer, got {self.filter_order!r}."
            )

        self._validate_cutoff_frequency()

    def _validate_cutoff_frequency(self) -> None:
        """Validates ``cutoff_frequency`` against ``filter_type`` and Nyquist.

        Raises:
            PreprocessingConfigError: If the cutoff frequency shape does
                not match ``filter_type``, or any cutoff value is not a
                strictly positive value below the Nyquist frequency.
        """
        nyquist = self.sampling_frequency / 2.0

        if self.filter_type in ("lowpass", "highpass"):
            if not isinstance(self.cutoff_frequency, (int, float)):
                raise PreprocessingConfigError(
                    f"cutoff_frequency for filter_type='{self.filter_type}' must be a "
                    f"single number, got {self.cutoff_frequency!r}."
                )
            cutoffs = (float(self.cutoff_frequency),)
        else:  # bandpass
            if (
                not isinstance(self.cutoff_frequency, tuple)
                or len(self.cutoff_frequency) != 2
            ):
                raise PreprocessingConfigError(
                    "cutoff_frequency for filter_type='bandpass' must be a "
                    f"(low, high) tuple, got {self.cutoff_frequency!r}."
                )
            low, high = self.cutoff_frequency
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise PreprocessingConfigError(
                    f"bandpass cutoff_frequency values must be numeric, got {self.cutoff_frequency!r}."
                )
            if low >= high:
                raise PreprocessingConfigError(
                    f"bandpass low cutoff ({low}) must be strictly less than high cutoff ({high})."
                )
            cutoffs = (float(low), float(high))

        for value in cutoffs:
            if value <= 0.0:
                raise PreprocessingConfigError(
                    f"cutoff_frequency values must be strictly positive, got {value}."
                )
            if value >= nyquist:
                raise PreprocessingConfigError(
                    f"cutoff_frequency value {value} Hz must be strictly below the "
                    f"Nyquist frequency ({nyquist} Hz) for sampling_frequency="
                    f"{self.sampling_frequency} Hz."
                )


# ==============================================================================
# SIGNAL VALIDATOR
# ==============================================================================

class SignalValidator:
    """Performs sanity/validation checks on signals passing through the
    preprocessing pipeline.

    Every check raises :class:`SignalValidationError` with an informative
    message on failure; corrupted or malformed signals are never allowed
    to silently continue through the pipeline.
    """

    def validate(self, signal: np.ndarray) -> None:
        """Runs the full validation suite on a signal.

        Args:
            signal: The array to validate.

        Raises:
            SignalValidationError: If any validation check fails.
        """
        self._check_is_array(signal)
        self._check_not_empty(signal)
        self._check_no_nan(signal)
        self._check_no_inf(signal)
        self._check_non_zero_energy(signal)

    @staticmethod
    def _check_is_array(signal: np.ndarray) -> None:
        if not isinstance(signal, np.ndarray):
            raise SignalValidationError(
                f"Expected a numpy.ndarray, received {type(signal)!r}."
            )
        if signal.ndim != 1:
            raise SignalValidationError(
                f"Expected a 1-D signal, received array with shape {signal.shape}."
            )

    @staticmethod
    def _check_not_empty(signal: np.ndarray) -> None:
        if signal.shape[0] == 0:
            raise SignalValidationError("Signal is empty (zero samples).")

    @staticmethod
    def _check_no_nan(signal: np.ndarray) -> None:
        if np.isnan(signal).any():
            raise SignalValidationError("Signal contains NaN value(s).")

    @staticmethod
    def _check_no_inf(signal: np.ndarray) -> None:
        if np.isinf(signal).any():
            raise SignalValidationError("Signal contains infinite value(s).")

    @staticmethod
    def _check_non_zero_energy(signal: np.ndarray) -> None:
        energy = np.sum(np.square(signal))
        if energy <= 0.0:
            raise SignalValidationError("Signal has zero energy.")


# ==============================================================================
# SIGNAL PREPROCESSOR
# ==============================================================================

class SignalPreprocessor:
    """Applies DC removal, optional Butterworth filtering, and amplitude
    normalization to 1-D analog modulation signals.

    This class is designed to be instantiated once (with a
    :class:`PreprocessingConfig`) and reused across:
        * offline batch preprocessing of the training dataset, and
        * online preprocessing of a single unknown signal at prediction
          time.

    The output signal always has the same number of samples as the input
    signal -- no resampling, cropping, or padding is ever performed.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """Initializes the preprocessor.

        Args:
            config: Preprocessing configuration. If omitted, a default
                :class:`PreprocessingConfig` (fs=20000 Hz, DC removal and
                normalization enabled, filtering disabled) is used.
        """
        self.config = config or PreprocessingConfig()
        self.validator = SignalValidator()
        self._sos = None  # Lazily (re)designed filter, cached per config.

        logger.info(
            "SignalPreprocessor initialized (fs=%.1f Hz, remove_dc=%s, "
            "normalize=%s, apply_filter=%s, filter_type=%s, cutoff=%s, order=%d).",
            self.config.sampling_frequency,
            self.config.remove_dc,
            self.config.normalize,
            self.config.apply_filter,
            self.config.filter_type,
            self.config.cutoff_frequency,
            self.config.filter_order,
        )

    # -------------------------------------------------------------------- #
    # Core DSP operations
    # -------------------------------------------------------------------- #

    def remove_dc_offset(self, signal: np.ndarray) -> np.ndarray:
        """Removes the DC offset from a signal by subtracting its mean.

        For a signal ``x[n]``, computes::

            DC = mean(x[n])
            x_processed[n] = x[n] - DC

        Args:
            signal: 1-D float64 signal.

        Returns:
            A new array with the DC component removed. Length is
            unchanged.
        """
        dc_component = np.mean(signal)
        return signal - dc_component

    def normalize_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Normalizes a signal to unit maximum absolute amplitude.

        Computes::

            x_normalized[n] = x[n] / max(abs(x[n]))

        Division by zero is handled safely: a signal whose maximum
        absolute amplitude is (numerically) zero is returned unchanged
        rather than producing NaN/Inf values.

        Args:
            signal: 1-D float64 signal.

        Returns:
            A new array with ``max(abs(x_normalized)) == 1`` (unless the
            input had zero amplitude everywhere, in which case it is
            returned unchanged). Length is unchanged.
        """
        peak_amplitude = np.max(np.abs(signal))

        if peak_amplitude <= 0.0 or not np.isfinite(peak_amplitude):
            logger.warning(
                "Skipping normalization: signal has zero or non-finite peak amplitude."
            )
            return signal.copy()

        return signal / peak_amplitude

    def design_filter(self) -> np.ndarray:
        """Designs the Butterworth filter specified by the configuration.

        Uses second-order sections (SOS) for numerical stability. The
        result is cached on the instance so repeated calls to
        :meth:`apply_butterworth_filter` do not redesign the filter
        every time.

        Returns:
            The filter's second-order-section representation, as
            returned by ``scipy.signal.butter(..., output="sos")``.

        Raises:
            FilterDesignError: If the filter cannot be designed (e.g.
                invalid normalized frequency).
        """
        if self._sos is not None:
            return self._sos

        nyquist = self.config.sampling_frequency / 2.0

        try:
            if self.config.filter_type == "bandpass":
                low, high = self.config.cutoff_frequency
                normalized_cutoff = (low / nyquist, high / nyquist)
            else:
                normalized_cutoff = self.config.cutoff_frequency / nyquist

            sos = butter(
                N=self.config.filter_order,
                Wn=normalized_cutoff,
                btype=self.config.filter_type,
                output="sos",
            )
        except ValueError as exc:
            raise FilterDesignError(
                f"Failed to design {self.config.filter_type} Butterworth filter "
                f"(order={self.config.filter_order}, cutoff={self.config.cutoff_frequency}, "
                f"fs={self.config.sampling_frequency}): {exc}"
            ) from exc

        self._sos = sos
        logger.debug(
            "Designed %s Butterworth filter: order=%d, cutoff=%s Hz, fs=%.1f Hz.",
            self.config.filter_type,
            self.config.filter_order,
            self.config.cutoff_frequency,
            self.config.sampling_frequency,
        )
        return sos

    def apply_butterworth_filter(self, signal: np.ndarray) -> np.ndarray:
        """Applies zero-phase Butterworth filtering to a signal.

        Uses ``scipy.signal.sosfiltfilt`` (zero-phase, second-order
        sections) so that filtering introduces no phase distortion.

        Args:
            signal: 1-D float64 signal.

        Returns:
            The filtered signal. Length is unchanged.

        Raises:
            FilterDesignError: If the filter cannot be designed.
            ValueError: If the signal is too short for the configured
                filter order (``sosfiltfilt`` requires a minimum number
                of samples relative to the filter's padding length).
        """
        sos = self.design_filter()

        min_required_length = 3 * (2 * self.config.filter_order + 1)
        if signal.shape[0] <= min_required_length:
            raise ValueError(
                f"Signal length ({signal.shape[0]} samples) is too short for "
                f"zero-phase filtering with filter_order={self.config.filter_order} "
                f"(requires > {min_required_length} samples)."
            )

        return sosfiltfilt(sos, signal)

    # -------------------------------------------------------------------- #
    # Main pipeline
    # -------------------------------------------------------------------- #

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Runs the full preprocessing pipeline on a single signal.

        Steps (each individually toggleable via the configuration):
            1. Convert to float64.
            2. Validate the input signal.
            3. Remove DC offset.
            4. Apply Butterworth filtering.
            5. Normalize amplitude.
            6. Validate the processed signal.

        The output always has the same number of samples as the input.

        Args:
            signal: 1-D array-like signal (any numeric dtype).

        Returns:
            The processed 1-D float64 NumPy array.

        Raises:
            SignalValidationError: If the input or output signal fails
                validation.
            FilterDesignError: If filtering is enabled but the filter
                cannot be designed.
            ValueError: If the signal is too short for the configured
                filter.
        """
        original_length = np.asarray(signal).shape[0] if hasattr(signal, "shape") else len(signal)

        processed = np.asarray(signal, dtype=np.float64)
        self.validator.validate(processed)

        if self.config.remove_dc:
            processed = self.remove_dc_offset(processed)

        if self.config.apply_filter:
            processed = self.apply_butterworth_filter(processed)

        if self.config.normalize:
            processed = self.normalize_amplitude(processed)

        self.validator.validate(processed)

        if processed.shape[0] != original_length:
            raise SignalValidationError(
                f"Preprocessing changed signal length: expected {original_length}, "
                f"got {processed.shape[0]}."
            )

        return processed

    # -------------------------------------------------------------------- #
    # File-level processing
    # -------------------------------------------------------------------- #

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Loads a single ``.npy`` signal file, preprocesses it, and
        optionally saves the result.

        Args:
            input_path: Path to the ``.npy`` file to load.
            output_path: If provided, the processed signal is saved to
                this path via ``numpy.save``. Parent directories are
                created automatically if needed. The original file at
                ``input_path`` is never modified.

        Returns:
            The processed 1-D float64 NumPy array.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
            ValueError: If ``input_path`` does not have a ``.npy``
                extension, or the loaded array fails validation.
            OSError: If the file cannot be read or the output cannot be
                written.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: '{input_path}'.")

        if input_path.suffix.lower() != ".npy":
            raise ValueError(
                f"Expected a '.npy' file, got '{input_path.suffix}' ('{input_path}')."
            )

        try:
            raw_signal = np.load(input_path)
        except (OSError, ValueError) as exc:
            raise OSError(f"Failed to load '{input_path}': {exc}") from exc

        logger.debug("Loaded signal '%s' (%d samples).", input_path, raw_signal.shape[0])

        processed = self.process(raw_signal)

        if output_path is not None:
            output_path = Path(output_path)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, processed)
            except OSError as exc:
                raise OSError(f"Failed to save processed signal to '{output_path}': {exc}") from exc
            logger.debug("Saved processed signal to '%s'.", output_path)

        return processed

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        pattern: str = "*.npy",
        recursive: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """Batch-preprocesses every ``.npy`` file found under ``input_dir``.

        This method NEVER overwrites the original raw dataset. If
        ``output_dir`` is provided, processed signals are written there,
        mirroring the relative sub-directory structure of ``input_dir``
        (e.g. ``AM/AM_0001.npy`` -> ``<output_dir>/AM/AM_0001.npy``). If
        ``output_dir`` is omitted, signals are processed in memory only
        and discarded after validation (useful for dry-run sanity
        checks).

        Args:
            input_dir: Root directory to search for ``.npy`` files (e.g.
                the ``dataset`` directory produced by
                ``signal_generator.py``).
            output_dir: Optional directory to write processed signals
                to. Should NOT be the same as ``input_dir``.
            pattern: Glob pattern used to find signal files.
            recursive: Whether to search sub-directories recursively.

        Returns:
            A tuple ``(success_count, failure_count, failed_files)``
            where ``failed_files`` lists the string paths of files that
            failed to process.

        Raises:
            FileNotFoundError: If ``input_dir`` does not exist.
            ValueError: If ``output_dir`` resolves to the same directory
                as ``input_dir``.
        """
        input_dir = Path(input_dir)

        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: '{input_dir}'.")

        if output_dir is not None:
            output_dir = Path(output_dir)
            if output_dir.resolve() == input_dir.resolve():
                raise ValueError(
                    "output_dir must differ from input_dir to avoid overwriting "
                    "the raw dataset."
                )

        search = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        npy_files = sorted(
            f for f in search
            if f.is_file() and (output_dir is None or output_dir not in f.parents)
        )

        if not npy_files:
            logger.warning("No files matching '%s' found under '%s'.", pattern, input_dir)
            return 0, 0, []

        success_count = 0
        failed_files: List[str] = []

        for file_path in tqdm(npy_files, desc="Preprocessing signals", unit="file"):
            try:
                target_path = None
                if output_dir is not None:
                    relative_path = file_path.relative_to(input_dir)
                    target_path = output_dir / relative_path

                self.process_file(file_path, output_path=target_path)
                success_count += 1
            except (FileNotFoundError, ValueError, OSError, SignalValidationError,
                    FilterDesignError) as exc:
                logger.error("Failed to preprocess '%s': %s", file_path, exc)
                failed_files.append(str(file_path))

        logger.info(
            "Batch preprocessing complete: %d succeeded, %d failed (out of %d files).",
            success_count,
            len(failed_files),
            len(npy_files),
        )

        return success_count, len(failed_files), failed_files


# ==============================================================================
# MODULE-LEVEL CONVENIENCE API
# ==============================================================================

def preprocess_signal(
    signal: np.ndarray,
    config: Optional[PreprocessingConfig] = None,
) -> np.ndarray:
    """Convenience function to preprocess a single signal without manually
    instantiating a :class:`SignalPreprocessor`.

    Args:
        signal: 1-D array-like signal.
        config: Optional preprocessing configuration. Defaults to
            :class:`PreprocessingConfig` defaults if omitted.

    Returns:
        The processed 1-D float64 NumPy array.
    """
    return SignalPreprocessor(config).process(signal)


# ==============================================================================
# STANDALONE DEMONSTRATION / COMPATIBILITY TEST
# ==============================================================================

def _run_synthetic_demo() -> None:
    """Runs a small, self-contained demonstration on a synthetic signal.

    Does not touch the filesystem or any real dataset files.
    """
    logger.info("Running synthetic-signal preprocessing demonstration.")

    fs = 20000.0
    duration = 0.1
    t = np.arange(int(fs * duration), dtype=np.float64) / fs

    # A synthetic AM-like signal with an artificial DC offset, for
    # demonstration purposes only.
    synthetic_signal = 3.0 + (1.0 + 0.5 * np.cos(2 * np.pi * 200 * t)) * np.cos(
        2 * np.pi * 2000 * t
    )

    config = PreprocessingConfig(sampling_frequency=fs, apply_filter=True)
    preprocessor = SignalPreprocessor(config)

    processed = preprocessor.process(synthetic_signal)

    logger.info(
        "Synthetic demo: input samples=%d, output samples=%d, "
        "input mean=%.4f, output mean=%.4f, output peak=%.4f.",
        synthetic_signal.shape[0],
        processed.shape[0],
        float(np.mean(synthetic_signal)),
        float(np.mean(processed)),
        float(np.max(np.abs(processed))),
    )


def _run_dataset_compatibility_test() -> None:
    """Optionally verifies compatibility with a real generated signal.

    If ``dataset/AM/`` exists and contains at least one ``.npy`` file,
    loads the first one, preprocesses it, and checks the invariants
    required by the rest of the pipeline. Does NOT regenerate the
    dataset, modify ``signal_generator.py``, or overwrite the sample
    signal. Silently skipped if the dataset is not present.
    """
    am_dir = Path("dataset") / "AM"
    if not am_dir.exists():
        logger.info("No 'dataset/AM/' directory found; skipping compatibility test.")
        return

    sample_files = sorted(am_dir.glob("*.npy"))
    if not sample_files:
        logger.info("No '.npy' files found in 'dataset/AM/'; skipping compatibility test.")
        return

    sample_path = sample_files[0]
    logger.info("Running compatibility test on '%s'.", sample_path)

    raw_signal = np.load(sample_path)
    preprocessor = SignalPreprocessor(PreprocessingConfig(sampling_frequency=20000.0))
    processed_signal = preprocessor.process(raw_signal)

    assert raw_signal.ndim == 1, "Raw signal is not 1-D."
    assert processed_signal.ndim == 1, "Processed signal is not 1-D."
    assert raw_signal.shape[0] == processed_signal.shape[0], (
        "Sample count mismatch between raw and processed signal."
    )
    assert np.all(np.isfinite(processed_signal)), "Processed signal contains non-finite values."
    assert np.sum(np.square(processed_signal)) > 0.0, "Processed signal has zero energy."

    logger.info(
        "Compatibility test passed: %d samples in, %d samples out, all finite, non-zero energy.",
        raw_signal.shape[0],
        processed_signal.shape[0],
    )


def main() -> None:
    """Standalone entry point.

    Running ``python preprocessing.py`` performs a safe, self-contained
    demonstration on a synthetic signal and, if a generated dataset is
    present, a small read-only compatibility check against one real
    ``dataset/AM/*.npy`` file. It NEVER bulk-processes the full dataset
    and NEVER writes into ``dataset/``.
    """
    _run_synthetic_demo()
    _run_dataset_compatibility_test()


if __name__ == "__main__":
    main()