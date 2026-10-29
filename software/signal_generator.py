"""
signal_generator.py
====================

Analog Modulation Classifier Using Machine Learning
-----------------------------------------------------

This module is responsible EXCLUSIVELY for the generation of a labeled
dataset of analog-modulated communication signals. The dataset produced
by this module is intended to be consumed by downstream feature
extraction and machine learning modules that are NOT part of this file.

Modulation schemes implemented:
    * AM      - Amplitude Modulation
    * FM      - Frequency Modulation
    * DSB-SC  - Double Sideband Suppressed Carrier
    * SSB     - Single Sideband (upper sideband, via Hilbert transform)

Design goals:
    * Mathematically correct modulation equations.
    * Reproducible via configurable random seeds.
    * Fully validated signals (no NaNs, no Infs, correct shape/dtype).
    * AWGN noise injection at controllable SNR levels.
    * Persisted as individual .npy files with an accompanying
      metadata.csv describing every generated signal.
    * Optional visualization utilities (time-domain, FFT, power
      spectrum) for a random sample of the generated dataset.

Author:
    Generated for a B.Tech AI & Data Science final-year project:
    "Analog Modulation Classifier Using Machine Learning"

This file performs ONLY signal generation, validation, storage, and
optional plotting. It does NOT perform feature extraction, model
training, prediction, or evaluation -- those responsibilities belong
to separate downstream modules.
"""

from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend safe for servers / notebooks.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def _build_logger(name: str = "signal_generator") -> logging.Logger:
    """Builds and returns a configured module-level logger.

    The logger writes to both the console and a rotating log file located
    inside the ``dataset`` directory (created lazily on first use).

    Args:
        name: Name to assign to the logger instance.

    Returns:
        A fully configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured (e.g., re-imported in same session).
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

class SignalGenerationError(Exception):
    """Raised when a signal cannot be generated due to invalid parameters."""


class SignalValidationError(Exception):
    """Raised when a generated signal fails one or more validation checks."""


class DatasetIOError(Exception):
    """Raised when a filesystem operation related to dataset storage fails."""


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Central configuration for dataset generation.

    All tunable parameters for the dataset generation pipeline live here.
    Keeping configuration centralized avoids magic numbers scattered
    throughout the codebase and makes the pipeline easy to reproduce.

    Attributes:
        sampling_frequency: Sampling frequency in Hz used for every
            generated signal.
        signals_per_class: Number of signals to generate per modulation
            type.
        modulation_types: List of modulation type labels.
        output_root: Root directory in which the dataset is stored.
        plots_dirname: Sub-directory (under ``output_root``) for plots.
        metadata_filename: Filename for the metadata CSV file.
        random_seed: Seed used to initialize NumPy's and Python's random
            number generators for reproducibility.
        carrier_freq_range: Inclusive (min, max) range for carrier
            frequency in Hz.
        message_freq_range: Inclusive (min, max) range for message
            (baseband) frequency in Hz.
        carrier_amplitude_range: Inclusive (min, max) range for carrier
            amplitude.
        message_amplitude_range: Inclusive (min, max) range for message
            amplitude.
        modulation_index_range: Inclusive (min, max) range for the
            modulation index / index-like parameter (AM modulation
            index ``mu`` and FM modulation index ``beta`` both draw
            from this range).
        duration_range: Inclusive (min, max) range for signal duration
            in seconds.
        carrier_phase_range: Inclusive (min, max) range for the carrier
            phase in radians.
        message_phase_range: Inclusive (min, max) range for the message
            phase in radians.
        snr_choices_db: Discrete SNR values (in dB) that are randomly
            assigned to each generated signal.
        float_dtype: NumPy floating point dtype used for all generated
            signals.
    """

    # ---- Sampling & dataset size -------------------------------------------------
    sampling_frequency: float = 20000.0
    signals_per_class: int = 1000
    modulation_types: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")

    # ---- Output locations ----------------------------------------------------
    output_root: Path = field(default_factory=lambda: Path("dataset"))
    plots_dirname: str = "plots"
    metadata_filename: str = "metadata.csv"

    # ---- Reproducibility -------------------------------------------------------
    random_seed: int = 42

    # ---- Signal parameter ranges ------------------------------------------------
    carrier_freq_range: Tuple[float, float] = (1000.0, 5000.0)
    message_freq_range: Tuple[float, float] = (50.0, 500.0)
    carrier_amplitude_range: Tuple[float, float] = (0.5, 2.0)
    message_amplitude_range: Tuple[float, float] = (0.5, 1.5)
    modulation_index_range: Tuple[float, float] = (0.2, 1.0)
    duration_range: Tuple[float, float] = (0.05, 0.20)
    carrier_phase_range: Tuple[float, float] = (0.0, 2 * np.pi)
    message_phase_range: Tuple[float, float] = (0.0, 2 * np.pi)

    # ---- Noise ---------------------------------------------------------------
    snr_choices_db: Tuple[float, ...] = (30.0, 20.0, 10.0, 5.0)

    # ---- Misc ------------------------------------------------------------------
    float_dtype: type = np.float64

    def class_directory(self, modulation: str) -> Path:
        """Returns the output directory for a specific modulation class.

        Args:
            modulation: Modulation type label (e.g. ``"AM"``).

        Returns:
            Path to the directory where signals of this modulation type
            are stored.
        """
        return self.output_root / modulation

    def plots_directory(self) -> Path:
        """Returns the directory in which plots are stored."""
        return self.output_root / self.plots_dirname

    def metadata_path(self) -> Path:
        """Returns the full path to the metadata CSV file."""
        return self.output_root / self.metadata_filename


# ==============================================================================
# SIGNAL PARAMETERS
# ==============================================================================

@dataclass
class SignalParameters:
    """Container describing the randomized parameters of a single signal.

    Attributes:
        modulation: Modulation type label (e.g. ``"AM"``, ``"FM"``).
        carrier_frequency: Carrier frequency in Hz.
        message_frequency: Message (baseband tone) frequency in Hz.
        carrier_amplitude: Carrier amplitude (unitless).
        message_amplitude: Message amplitude (unitless).
        modulation_index: Modulation index. Interpreted as ``mu`` for AM
            and ``beta`` for FM. Unused for DSB-SC and SSB but retained
            for a uniform metadata schema.
        carrier_phase: Carrier phase offset in radians.
        message_phase: Message phase offset in radians.
        duration: Signal duration in seconds.
        sampling_frequency: Sampling frequency in Hz.
        snr_db: Signal-to-noise ratio (in dB) used for AWGN injection.
    """

    modulation: str
    carrier_frequency: float
    message_frequency: float
    carrier_amplitude: float
    message_amplitude: float
    modulation_index: float
    carrier_phase: float
    message_phase: float
    duration: float
    sampling_frequency: float
    snr_db: float

    def time_vector(self) -> np.ndarray:
        """Builds the discrete time vector for this parameter set.

        Returns:
            1-D NumPy array of time samples, in seconds, spaced by
            ``1 / sampling_frequency``.
        """
        num_samples = int(round(self.duration * self.sampling_frequency))
        return np.arange(num_samples, dtype=np.float64) / self.sampling_frequency

    def to_metadata_dict(self, filename: str) -> Dict[str, object]:
        """Converts this parameter set into a flat dict for metadata storage.

        Args:
            filename: The ``.npy`` filename this signal was saved under.

        Returns:
            Dictionary suitable for appending as a row to the metadata
            DataFrame.
        """
        return {
            "filename": filename,
            "modulation": self.modulation,
            "carrier_frequency": self.carrier_frequency,
            "message_frequency": self.message_frequency,
            "carrier_amplitude": self.carrier_amplitude,
            "message_amplitude": self.message_amplitude,
            "modulation_index": self.modulation_index,
            "carrier_phase": self.carrier_phase,
            "message_phase": self.message_phase,
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "snr": self.snr_db,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


# ==============================================================================
# PARAMETER SAMPLER
# ==============================================================================

class ParameterSampler:
    """Randomly samples :class:`SignalParameters` instances given a Config.

    Centralizing parameter sampling in its own class keeps randomness
    logic separate from modulation mathematics and makes it trivial to
    swap in alternative sampling strategies (e.g. Latin hypercube) in
    the future without touching generator classes.
    """

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        """Initializes the sampler.

        Args:
            config: Global dataset configuration.
            rng: NumPy random Generator used for all draws, ensuring
                reproducibility when seeded consistently.
        """
        self._config = config
        self._rng = rng

    def sample(self, modulation: str) -> SignalParameters:
        """Draws a new random :class:`SignalParameters` for a modulation type.

        Args:
            modulation: The modulation type label these parameters are
                being generated for.

        Returns:
            A freshly sampled :class:`SignalParameters` instance.
        """
        cfg = self._config
        return SignalParameters(
            modulation=modulation,
            carrier_frequency=self._uniform(cfg.carrier_freq_range),
            message_frequency=self._uniform(cfg.message_freq_range),
            carrier_amplitude=self._uniform(cfg.carrier_amplitude_range),
            message_amplitude=self._uniform(cfg.message_amplitude_range),
            modulation_index=self._uniform(cfg.modulation_index_range),
            carrier_phase=self._uniform(cfg.carrier_phase_range),
            message_phase=self._uniform(cfg.message_phase_range),
            duration=self._uniform(cfg.duration_range),
            sampling_frequency=cfg.sampling_frequency,
            snr_db=float(self._rng.choice(cfg.snr_choices_db)),
        )

    def _uniform(self, value_range: Tuple[float, float]) -> float:
        """Draws a single uniform random float from an inclusive range.

        Args:
            value_range: A ``(low, high)`` tuple.

        Returns:
            A random float sampled uniformly from ``[low, high]``.
        """
        low, high = value_range
        return float(self._rng.uniform(low, high))


# ==============================================================================
# NOISE GENERATOR
# ==============================================================================

class NoiseGenerator:
    """Generates Additive White Gaussian Noise (AWGN) at a target SNR.

    The noise power is derived from the average power of the clean
    signal so that the resulting noisy signal achieves (in expectation)
    the requested signal-to-noise ratio.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        """Initializes the noise generator.

        Args:
            rng: NumPy random Generator used to draw Gaussian samples.
        """
        self._rng = rng

    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Adds AWGN to a signal at the specified SNR.

        Args:
            signal: Clean 1-D input signal.
            snr_db: Desired signal-to-noise ratio, in decibels.

        Returns:
            The noisy signal, with the same shape and dtype as the
            input.

        Raises:
            SignalGenerationError: If the input signal has zero power,
                making a well-defined SNR impossible to target.
        """
        signal_power = np.mean(np.square(signal))

        if signal_power <= 0.0 or not np.isfinite(signal_power):
            raise SignalGenerationError(
                "Cannot add AWGN to a signal with zero or non-finite power."
            )

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)

        noise = self._rng.normal(loc=0.0, scale=noise_std, size=signal.shape)
        noisy_signal = signal + noise

        return noisy_signal.astype(signal.dtype, copy=False)


# ==============================================================================
# SIGNAL VALIDATOR
# ==============================================================================

class SignalValidator:
    """Performs sanity/validation checks on generated signals.

    All checks raise :class:`SignalValidationError` with an informative
    message on failure, allowing calling code to catch and log/skip
    invalid signals without crashing the entire generation pipeline.
    """

    def __init__(self, expected_dtype: type = np.float64) -> None:
        """Initializes the validator.

        Args:
            expected_dtype: The NumPy dtype every valid signal must
                match.
        """
        self._expected_dtype = expected_dtype

    def validate(self, signal: np.ndarray, expected_length: int) -> None:
        """Runs the full validation suite on a signal.

        Args:
            signal: The signal array to validate.
            expected_length: The number of samples the signal is
                expected to contain.

        Raises:
            SignalValidationError: If any validation check fails.
        """
        self._check_is_array(signal)
        self._check_no_nan(signal)
        self._check_no_inf(signal)
        self._check_length(signal, expected_length)
        self._check_dtype(signal)
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
    def _check_no_nan(signal: np.ndarray) -> None:
        if np.isnan(signal).any():
            raise SignalValidationError("Signal contains NaN value(s).")

    @staticmethod
    def _check_no_inf(signal: np.ndarray) -> None:
        if np.isinf(signal).any():
            raise SignalValidationError("Signal contains infinite value(s).")

    @staticmethod
    def _check_length(signal: np.ndarray, expected_length: int) -> None:
        if signal.shape[0] != expected_length:
            raise SignalValidationError(
                f"Signal length mismatch: expected {expected_length}, "
                f"got {signal.shape[0]}."
            )

    def _check_dtype(self, signal: np.ndarray) -> None:
        if signal.dtype != self._expected_dtype:
            raise SignalValidationError(
                f"Signal dtype mismatch: expected {self._expected_dtype}, "
                f"got {signal.dtype}."
            )

    @staticmethod
    def _check_non_zero_energy(signal: np.ndarray) -> None:
        energy = np.sum(np.square(signal))
        if energy <= 0.0:
            raise SignalValidationError("Signal has zero energy.")


# ==============================================================================
# BASE SIGNAL GENERATOR
# ==============================================================================

class BaseSignalGenerator(ABC):
    """Abstract base class for all modulation-specific signal generators.

    Subclasses implement :meth:`_modulate` with the actual modulation
    mathematics. This base class handles the shared workflow: building
    the message/time vectors, invoking the subclass modulation
    equation, optionally injecting AWGN, and validating the result.
    """

    #: Human readable modulation label, overridden by subclasses.
    modulation_label: str = "BASE"

    def __init__(
        self,
        noise_generator: NoiseGenerator,
        validator: SignalValidator,
    ) -> None:
        """Initializes the base generator.

        Args:
            noise_generator: Shared :class:`NoiseGenerator` instance
                used to inject AWGN into generated signals.
            validator: Shared :class:`SignalValidator` instance used to
                sanity-check generated signals.
        """
        self._noise_generator = noise_generator
        self._validator = validator

    def generate(self, params: SignalParameters) -> np.ndarray:
        """Generates a single validated, noisy signal.

        Args:
            params: Randomized parameters describing the signal to
                generate.

        Returns:
            A 1-D NumPy array containing the noisy modulated signal.

        Raises:
            SignalGenerationError: If the underlying modulation
                equation fails.
            SignalValidationError: If the resulting signal fails
                validation.
        """
        try:
            t = params.time_vector()
            message = self._message_waveform(params, t)
            clean_signal = self._modulate(params, t, message)
        except SignalGenerationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive re-raise
            raise SignalGenerationError(
                f"Failed to generate {self.modulation_label} signal: {exc}"
            ) from exc

        clean_signal = clean_signal.astype(np.float64, copy=False)

        noisy_signal = self._noise_generator.add_awgn(clean_signal, params.snr_db)

        self._validator.validate(noisy_signal, expected_length=t.shape[0])

        return noisy_signal

    @staticmethod
    def _message_waveform(params: SignalParameters, t: np.ndarray) -> np.ndarray:
        """Builds the baseband message tone shared across modulation types.

        Args:
            params: Signal parameters providing message amplitude,
                frequency, and phase.
            t: Time vector, in seconds.

        Returns:
            The baseband message signal ``m(t) = Am * cos(2*pi*fm*t + phi_m)``.
        """
        return params.message_amplitude * np.cos(
            2 * np.pi * params.message_frequency * t + params.message_phase
        )

    @abstractmethod
    def _modulate(
        self,
        params: SignalParameters,
        t: np.ndarray,
        message: np.ndarray,
    ) -> np.ndarray:
        """Applies the modulation equation for this scheme.

        Args:
            params: Randomized signal parameters.
            t: Time vector, in seconds.
            message: Baseband message waveform ``m(t)``.

        Returns:
            The clean (noise-free) modulated signal.
        """
        raise NotImplementedError


# ==============================================================================
# AM GENERATOR
# ==============================================================================

class AMGenerator(BaseSignalGenerator):
    """Generates standard Amplitude Modulation (AM) signals.

    Implements::

        s(t) = Ac * (1 + mu * m(t)) * cos(2*pi*fc*t + phi_c)
    """

    modulation_label = "AM"

    def _modulate(
        self,
        params: SignalParameters,
        t: np.ndarray,
        message: np.ndarray,
    ) -> np.ndarray:
        # Normalize the message to unit amplitude before applying the
        # modulation index so that `mu` has its conventional meaning
        # (mu=1 -> 100% modulation depth) regardless of message_amplitude.
        normalized_message = message / max(params.message_amplitude, 1e-12)

        carrier = np.cos(2 * np.pi * params.carrier_frequency * t + params.carrier_phase)

        envelope = 1.0 + params.modulation_index * normalized_message
        signal = params.carrier_amplitude * envelope * carrier
        return signal


# ==============================================================================
# FM GENERATOR
# ==============================================================================

class FMGenerator(BaseSignalGenerator):
    """Generates Frequency Modulation (FM) signals.

    Implements::

        s(t) = Ac * cos(2*pi*fc*t + beta * sin(2*pi*fm*t + phi_m))
    """

    modulation_label = "FM"

    def _modulate(
        self,
        params: SignalParameters,
        t: np.ndarray,
        message: np.ndarray,
    ) -> np.ndarray:
        # FM directly uses a sinusoidal phase deviation term, driven by
        # message frequency/phase and scaled by the modulation index beta.
        phase_deviation = params.modulation_index * np.sin(
            2 * np.pi * params.message_frequency * t + params.message_phase
        )
        signal = params.carrier_amplitude * np.cos(
            2 * np.pi * params.carrier_frequency * t + params.carrier_phase + phase_deviation
        )
        return signal


# ==============================================================================
# DSB-SC GENERATOR
# ==============================================================================

class DSBSCGenerator(BaseSignalGenerator):
    """Generates Double Sideband Suppressed Carrier (DSB-SC) signals.

    Implements::

        s(t) = Ac * m(t) * cos(2*pi*fc*t + phi_c)
    """

    modulation_label = "DSB_SC"

    def _modulate(
        self,
        params: SignalParameters,
        t: np.ndarray,
        message: np.ndarray,
    ) -> np.ndarray:
        carrier = np.cos(2 * np.pi * params.carrier_frequency * t + params.carrier_phase)
        signal = params.carrier_amplitude * message * carrier
        return signal


# ==============================================================================
# SSB GENERATOR
# ==============================================================================

class SSBGenerator(BaseSignalGenerator):
    """Generates Single Sideband (SSB) signals using the Hilbert transform.

    The upper-sideband SSB signal is constructed from the analytic
    signal of the baseband message, following the standard
    phasing-method identity::

        m_hat(t) = Hilbert transform of m(t)
        s(t) = Ac * [ m(t) * cos(wc*t + phi_c) - m_hat(t) * sin(wc*t + phi_c) ]

    This produces the upper sideband (USB). No approximation is used;
    ``scipy.signal.hilbert`` computes the analytic signal directly.
    """

    modulation_label = "SSB"

    def _modulate(
        self,
        params: SignalParameters,
        t: np.ndarray,
        message: np.ndarray,
    ) -> np.ndarray:
        analytic_message = hilbert(message)
        message_hilbert = np.imag(analytic_message)

        carrier_cos = np.cos(2 * np.pi * params.carrier_frequency * t + params.carrier_phase)
        carrier_sin = np.sin(2 * np.pi * params.carrier_frequency * t + params.carrier_phase)

        # Upper sideband via the phasing method.
        signal = params.carrier_amplitude * (
            message * carrier_cos - message_hilbert * carrier_sin
        )
        return signal


# ==============================================================================
# GENERATOR FACTORY
# ==============================================================================

class GeneratorFactory:
    """Builds and caches the modulation-specific generator instances."""

    _GENERATOR_CLASSES: Dict[str, type] = {
        "AM": AMGenerator,
        "FM": FMGenerator,
        "DSB_SC": DSBSCGenerator,
        "SSB": SSBGenerator,
    }

    def __init__(
        self,
        noise_generator: NoiseGenerator,
        validator: SignalValidator,
    ) -> None:
        """Initializes the factory.

        Args:
            noise_generator: Shared noise generator to inject into each
                created generator instance.
            validator: Shared validator to inject into each created
                generator instance.
        """
        self._noise_generator = noise_generator
        self._validator = validator
        self._cache: Dict[str, BaseSignalGenerator] = {}

    def get(self, modulation: str) -> BaseSignalGenerator:
        """Retrieves (and lazily builds) the generator for a modulation type.

        Args:
            modulation: Modulation type label.

        Returns:
            The corresponding :class:`BaseSignalGenerator` subclass
            instance.

        Raises:
            ValueError: If ``modulation`` is not a recognized type.
        """
        if modulation not in self._GENERATOR_CLASSES:
            raise ValueError(
                f"Unknown modulation type '{modulation}'. "
                f"Valid options: {list(self._GENERATOR_CLASSES.keys())}"
            )

        if modulation not in self._cache:
            generator_cls = self._GENERATOR_CLASSES[modulation]
            self._cache[modulation] = generator_cls(
                noise_generator=self._noise_generator,
                validator=self._validator,
            )

        return self._cache[modulation]


# ==============================================================================
# SIGNAL PLOTTER
# ==============================================================================

class SignalPlotter:
    """Produces optional diagnostic plots for generated signals.

    Plots are saved as PNG files inside ``dataset/plots``. Plotting is
    entirely optional and gracefully skipped (with a warning) if
    matplotlib is unavailable.
    """

    def __init__(self, config: Config) -> None:
        """Initializes the plotter.

        Args:
            config: Global dataset configuration, used to resolve the
                output plots directory.
        """
        self._config = config

    def plot_signal(
        self,
        signal: np.ndarray,
        params: SignalParameters,
        filename_stem: str,
    ) -> Optional[Path]:
        """Plots time-domain, FFT magnitude, and power spectrum for a signal.

        Args:
            signal: The signal samples to plot.
            params: The parameters used to generate the signal (used
                for axis scaling and plot titles).
            filename_stem: Base filename (without extension) to save
                the plot under.

        Returns:
            The path to the saved plot file, or ``None`` if plotting
            was skipped because matplotlib is unavailable.
        """
        if plt is None:
            logger.warning("matplotlib is unavailable; skipping plot generation.")
            return None

        fs = params.sampling_frequency
        n = signal.shape[0]
        t = np.arange(n) / fs

        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.fft.rfft(signal)
        magnitude = np.abs(spectrum) / n
        power = np.square(magnitude)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9))

        axes[0].plot(t * 1000.0, signal, color="tab:blue", linewidth=0.8)
        axes[0].set_title(f"{params.modulation} - Time Domain")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(freqs, magnitude, color="tab:orange", linewidth=0.8)
        axes[1].set_title(f"{params.modulation} - FFT Magnitude Spectrum")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Magnitude")
        axes[1].set_xlim(0, min(fs / 2, params.carrier_frequency * 3))
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(freqs, power, color="tab:green", linewidth=0.8)
        axes[2].set_title(f"{params.modulation} - Power Spectrum")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Power")
        axes[2].set_xlim(0, min(fs / 2, params.carrier_frequency * 3))
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()

        output_dir = self._config.plots_directory()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename_stem}.png"

        try:
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)

        logger.info("Saved plot: %s", output_path)
        return output_path

    def plot_random_examples(
        self,
        dataset_manager: "DatasetManager",
        metadata_df: pd.DataFrame,
        samples_per_class: int = 1,
    ) -> List[Path]:
        """Plots a small random sample of signals from each modulation class.

        Args:
            dataset_manager: Manager used to load saved ``.npy`` signal
                files back from disk.
            metadata_df: The full metadata DataFrame describing the
                generated dataset.
            samples_per_class: How many random examples to plot per
                modulation type.

        Returns:
            List of paths to the generated plot files.
        """
        saved_paths: List[Path] = []

        for modulation in metadata_df["modulation"].unique():
            class_rows = metadata_df[metadata_df["modulation"] == modulation]
            sample_rows = class_rows.sample(
                n=min(samples_per_class, len(class_rows)),
                random_state=None,
            )

            for _, row in sample_rows.iterrows():
                signal = dataset_manager.load_signal(row["modulation"], row["filename"])
                params = SignalParameters(
                    modulation=row["modulation"],
                    carrier_frequency=row["carrier_frequency"],
                    message_frequency=row["message_frequency"],
                    carrier_amplitude=row["carrier_amplitude"],
                    message_amplitude=row["message_amplitude"],
                    modulation_index=row["modulation_index"],
                    carrier_phase=row["carrier_phase"],
                    message_phase=row["message_phase"],
                    duration=row["duration"],
                    sampling_frequency=row["sampling_frequency"],
                    snr_db=row["snr"],
                )
                stem = Path(row["filename"]).stem
                path = self.plot_signal(signal, params, filename_stem=f"sample_{stem}")
                if path is not None:
                    saved_paths.append(path)

        return saved_paths


# ==============================================================================
# DATASET MANAGER
# ==============================================================================

class DatasetManager:
    """Handles all filesystem I/O for storing and loading signal data."""

    def __init__(self, config: Config) -> None:
        """Initializes the dataset manager.

        Args:
            config: Global dataset configuration describing output
                directory structure.
        """
        self._config = config

    def create_directories(self) -> None:
        """Creates the full dataset directory tree if it does not exist.

        Raises:
            DatasetIOError: If directory creation fails for any reason
                (e.g. permissions).
        """
        try:
            self._config.output_root.mkdir(parents=True, exist_ok=True)
            for modulation in self._config.modulation_types:
                self._config.class_directory(modulation).mkdir(parents=True, exist_ok=True)
            self._config.plots_directory().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatasetIOError(f"Failed to create dataset directories: {exc}") from exc

        logger.info("Dataset directory tree ready at '%s'.", self._config.output_root)

    def build_filename(self, modulation: str, index: int) -> str:
        """Builds the canonical filename for a signal.

        Args:
            modulation: Modulation type label.
            index: 1-based index of the signal within its class.

        Returns:
            Filename in the form ``AM_0001.npy``.
        """
        return f"{modulation}_{index:04d}.npy"

    def save_signal(self, signal: np.ndarray, modulation: str, filename: str) -> Path:
        """Persists a single signal array as a ``.npy`` file.

        Args:
            signal: The signal samples to save.
            modulation: Modulation type label (used to resolve the
                target sub-directory).
            filename: Filename (including ``.npy`` extension) to save
                the signal under.

        Returns:
            The full path the signal was saved to.

        Raises:
            DatasetIOError: If the save operation fails.
        """
        target_dir = self._config.class_directory(modulation)
        target_path = target_dir / filename

        try:
            np.save(target_path, signal)
        except OSError as exc:
            raise DatasetIOError(f"Failed to save signal to '{target_path}': {exc}") from exc

        return target_path

    def load_signal(self, modulation: str, filename: str) -> np.ndarray:
        """Loads a previously saved signal from disk.

        Args:
            modulation: Modulation type label.
            filename: Filename of the ``.npy`` file to load.

        Returns:
            The loaded signal array.

        Raises:
            DatasetIOError: If the file cannot be found or loaded.
        """
        target_path = self._config.class_directory(modulation) / filename
        try:
            return np.load(target_path)
        except (OSError, ValueError) as exc:
            raise DatasetIOError(f"Failed to load signal from '{target_path}': {exc}") from exc


# ==============================================================================
# METADATA MANAGER
# ==============================================================================

class MetadataManager:
    """Accumulates per-signal metadata records and writes them to CSV."""

    _COLUMNS = [
        "filename",
        "modulation",
        "carrier_frequency",
        "message_frequency",
        "carrier_amplitude",
        "message_amplitude",
        "modulation_index",
        "carrier_phase",
        "message_phase",
        "duration",
        "sampling_frequency",
        "snr",
        "timestamp",
    ]

    def __init__(self, config: Config) -> None:
        """Initializes the metadata manager.

        Args:
            config: Global dataset configuration describing where the
                metadata CSV file should be written.
        """
        self._config = config
        self._records: List[Dict[str, object]] = []

    def add_record(self, params: SignalParameters, filename: str) -> None:
        """Registers a new metadata record for a saved signal.

        Args:
            params: Parameters describing the signal.
            filename: The ``.npy`` filename the signal was saved as.
        """
        self._records.append(params.to_metadata_dict(filename))

    def as_dataframe(self) -> pd.DataFrame:
        """Returns all accumulated records as a pandas DataFrame.

        Returns:
            A DataFrame with columns matching ``_COLUMNS``, in the
            order the records were added.
        """
        if not self._records:
            return pd.DataFrame(columns=self._COLUMNS)
        return pd.DataFrame(self._records, columns=self._COLUMNS)

    def save_metadata(self) -> Path:
        """Writes the accumulated metadata records to the configured CSV path.

        Returns:
            Path to the written metadata CSV file.

        Raises:
            DatasetIOError: If the CSV cannot be written.
        """
        df = self.as_dataframe()
        target_path = self._config.metadata_path()

        try:
            df.to_csv(target_path, index=False)
        except OSError as exc:
            raise DatasetIOError(f"Failed to write metadata CSV to '{target_path}': {exc}") from exc

        logger.info("Saved metadata for %d signals to '%s'.", len(df), target_path)
        return target_path

    def record_count(self) -> int:
        """Returns the number of metadata records accumulated so far."""
        return len(self._records)


# ==============================================================================
# TOP-LEVEL SIGNAL GENERATOR (FACADE / ORCHESTRATOR)
# ==============================================================================

class SignalGenerator:
    """High-level facade orchestrating the full dataset generation pipeline.

    This is the primary public entry point of the module. It wires
    together the configuration, parameter sampler, modulation-specific
    generators, validator, noise generator, dataset manager, metadata
    manager, and optional plotter into a single cohesive workflow.

    Typical usage::

        config = Config()
        generator = SignalGenerator(config)
        generator.generate_dataset(with_plots=True)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initializes the signal generator pipeline.

        Args:
            config: Optional custom :class:`Config`. If omitted, a
                default configuration is used.
        """
        self.config = config or Config()

        # Seed both NumPy's Generator and the stdlib `random` module
        # (the latter is not strictly required here but is seeded for
        # completeness/reproducibility of any incidental randomness).
        self._rng = np.random.default_rng(self.config.random_seed)
        random.seed(self.config.random_seed)

        self._validator = SignalValidator(expected_dtype=self.config.float_dtype)
        self._noise_generator = NoiseGenerator(self._rng)
        self._sampler = ParameterSampler(self.config, self._rng)
        self._factory = GeneratorFactory(self._noise_generator, self._validator)

        self.dataset_manager = DatasetManager(self.config)
        self.metadata_manager = MetadataManager(self.config)
        self.plotter = SignalPlotter(self.config)

        logger.info(
            "SignalGenerator initialized (seed=%d, fs=%.1f Hz, classes=%s).",
            self.config.random_seed,
            self.config.sampling_frequency,
            self.config.modulation_types,
        )

    # -------------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------------- #

    def create_directories(self) -> None:
        """Creates the dataset output directory structure.

        Delegates to :meth:`DatasetManager.create_directories`.
        """
        self.dataset_manager.create_directories()

    def generate_signal(self, modulation: str) -> Tuple[np.ndarray, SignalParameters]:
        """Generates a single random, validated, noisy signal.

        Args:
            modulation: Modulation type label to generate (e.g.
                ``"AM"``, ``"FM"``, ``"DSB_SC"``, ``"SSB"``).

        Returns:
            A tuple of ``(signal, params)`` where ``signal`` is the
            noisy modulated waveform and ``params`` is the
            :class:`SignalParameters` used to generate it.

        Raises:
            SignalGenerationError: If modulation fails.
            SignalValidationError: If the resulting signal is invalid.
        """
        params = self._sampler.sample(modulation)
        generator = self._factory.get(modulation)
        signal = generator.generate(params)
        return signal, params

    def save_signal(
        self,
        signal: np.ndarray,
        modulation: str,
        index: int,
    ) -> Tuple[Path, str]:
        """Saves a signal to disk using the canonical naming convention.

        Args:
            signal: The signal samples to save.
            modulation: Modulation type label.
            index: 1-based index of this signal within its class.

        Returns:
            A tuple ``(path, filename)`` describing where the signal
            was saved.
        """
        filename = self.dataset_manager.build_filename(modulation, index)
        path = self.dataset_manager.save_signal(signal, modulation, filename)
        return path, filename

    def save_metadata(self) -> Path:
        """Writes all accumulated metadata to the configured CSV path.

        Returns:
            Path to the saved metadata CSV file.
        """
        return self.metadata_manager.save_metadata()

    def plot_signal(
        self,
        signal: np.ndarray,
        params: SignalParameters,
        filename_stem: str,
    ) -> Optional[Path]:
        """Plots a single signal's time/frequency-domain characteristics.

        Args:
            signal: Signal samples to plot.
            params: Parameters describing the signal.
            filename_stem: Base filename (no extension) for the saved
                plot image.

        Returns:
            Path to the saved plot, or ``None`` if plotting was
            skipped.
        """
        return self.plotter.plot_signal(signal, params, filename_stem)

    def generate_dataset(self, with_plots: bool = True, samples_per_class_plot: int = 2) -> pd.DataFrame:
        """Generates the complete labeled dataset for all modulation types.

        This is the main orchestration method: for every modulation
        type, it generates ``config.signals_per_class`` validated
        signals, saves each to disk, and records its metadata. After
        all signals are generated, the metadata CSV is written and
        (optionally) a handful of random example plots are produced.

        Args:
            with_plots: Whether to produce diagnostic plots for a
                random sample of generated signals after dataset
                creation.
            samples_per_class_plot: Number of random example plots to
                generate per modulation class (only used if
                ``with_plots`` is True).

        Returns:
            The full metadata DataFrame describing the generated
            dataset.

        Raises:
            DatasetIOError: If directory creation or file I/O fails.
        """
        logger.info("Starting full dataset generation.")
        self.create_directories()

        total_signals = self.config.signals_per_class * len(self.config.modulation_types)
        num_failed = 0

        with tqdm(total=total_signals, desc="Generating dataset", unit="signal") as progress:
            for modulation in self.config.modulation_types:
                logger.info(
                    "Generating %d signals for modulation class '%s'.",
                    self.config.signals_per_class,
                    modulation,
                )

                index = 1
                generated_for_class = 0

                # Guard against pathological infinite loops: cap total
                # attempts per class at a generous multiple of the target.
                max_attempts = self.config.signals_per_class * 5
                attempts = 0

                while generated_for_class < self.config.signals_per_class and attempts < max_attempts:
                    attempts += 1
                    try:
                        signal, params = self.generate_signal(modulation)
                    except (SignalGenerationError, SignalValidationError) as exc:
                        num_failed += 1
                        logger.warning(
                            "Skipping invalid %s signal (attempt %d): %s",
                            modulation,
                            attempts,
                            exc,
                        )
                        continue

                    _, filename = self.save_signal(signal, modulation, index)
                    self.metadata_manager.add_record(params, filename)

                    index += 1
                    generated_for_class += 1
                    progress.update(1)

                if generated_for_class < self.config.signals_per_class:
                    logger.error(
                        "Only generated %d/%d signals for class '%s' after %d attempts.",
                        generated_for_class,
                        self.config.signals_per_class,
                        modulation,
                        attempts,
                    )

        metadata_path = self.save_metadata()
        metadata_df = self.metadata_manager.as_dataframe()

        logger.info(
            "Dataset generation complete: %d signals saved, %d generation attempts failed validation.",
            self.metadata_manager.record_count(),
            num_failed,
        )
        logger.info("Metadata written to '%s'.", metadata_path)

        if with_plots:
            logger.info("Generating diagnostic plots for random dataset examples.")
            self.plotter.plot_random_examples(
                self.dataset_manager,
                metadata_df,
                samples_per_class=samples_per_class_plot,
            )

        self._write_run_summary(metadata_df, num_failed)

        return metadata_df

    # -------------------------------------------------------------------- #
    # Internal helpers
    # -------------------------------------------------------------------- #

    def _write_run_summary(self, metadata_df: pd.DataFrame, num_failed: int) -> None:
        """Writes a small JSON summary of the generation run for auditing.

        Args:
            metadata_df: The final metadata DataFrame.
            num_failed: Number of generation attempts that failed
                validation and were skipped.
        """
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "random_seed": self.config.random_seed,
            "sampling_frequency": self.config.sampling_frequency,
            "signals_per_class": self.config.signals_per_class,
            "modulation_types": list(self.config.modulation_types),
            "total_signals_generated": int(len(metadata_df)),
            "failed_generation_attempts": num_failed,
        }

        summary_path = self.config.output_root / "run_summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=4)
        except OSError as exc:
            logger.warning("Could not write run summary to '%s': %s", summary_path, exc)
            return

        logger.info("Run summary written to '%s'.", summary_path)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Entry point: builds the default configuration and generates the dataset."""
    config = Config()
    generator = SignalGenerator(config)
    generator.generate_dataset(with_plots=True, samples_per_class_plot=2)


if __name__ == "__main__":
    main()