"""
predict.py
==========

Analog Modulation Classifier Using Machine Learning
-----------------------------------------------------

This module is responsible EXCLUSIVELY for running inference on a
single, previously-unseen ``.npy`` signal file:

    1. Loading and validating the raw signal array.
    2. Preprocessing it via the EXISTING, unmodified
       ``preprocessing.SignalPreprocessor`` (same configuration used
       during feature extraction: fs=20000 Hz, remove_dc=True,
       normalize=True, apply_filter=False).
    3. Extracting the 12 DSP features via the EXISTING, unmodified
       ``feature_extraction.FeatureExtractor`` (no feature math is
       reimplemented here).
    4. Loading ``trained_models/best_model.joblib`` (never retraining,
       never refitting a scaler or encoder).
    5. Predicting the modulation class, decoding XGBoost's integer
       output via ``trained_models/label_encoder.joblib`` when
       necessary, and computing a class-aligned confidence score.
    6. Reporting the result on the terminal and via a reusable
       ``predict_signal()`` API.

It does NOT retrain, refit, or save any model. It does NOT duplicate
preprocessing or feature-extraction logic -- both are imported
directly from ``preprocessing.py`` and ``feature_extraction.py``. It
does NOT use the input filename or its parent directory to determine
the predicted class -- the prediction comes entirely from the
extracted feature vector and the trained model.

Pipeline stage handled here::

    <unknown_signal>.npy
            |
            v
    SignalLoader (validate + normalize array shape)
            |
            v
    preprocessing.SignalPreprocessor.process()   (reused as-is)
            |
            v
    feature_extraction.FeatureExtractor.extract_features()   (reused as-is)
            |
            v
    12-feature vector, exact training order
            |
            v
    trained_models/best_model.joblib.predict() / .predict_proba()
            |
            v
    (decode via label_encoder.joblib if integer-encoded)
            |
            v
    PredictionResult (modulation, confidence, class probabilities, features)

Running::

    python predict.py dataset/AM/AM_0001.npy

prints a full prediction report and requires no interactive input.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from feature_extraction import FeatureExtractionError, FeatureExtractor
from preprocessing import PreprocessingConfig, SignalPreprocessor, SignalValidationError


# ==============================================================================
# LOGGING CONFIGURATION (mirrors the rest of the project)
# ==============================================================================

def _build_logger(name: str = "predict") -> logging.Logger:
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

class SignalLoadError(Exception):
    """Raised when the input ``.npy`` signal cannot be loaded or is invalid."""


class ArtifactLoadError(Exception):
    """Raised when a required trained-model artifact cannot be found or loaded."""


class PredictionError(Exception):
    """Raised when preprocessing, feature extraction, or model prediction fails."""


# ==============================================================================
# CONSTANTS / CONFIGURATION
# ==============================================================================
#
# These MUST stay in lockstep with train_models.py / feature_extraction.py.
# They are duplicated as data (not logic) so this module stays self-contained.

#: Exact, ordered list of feature columns the models were trained on.
FEATURE_COLUMNS: Tuple[str, ...] = (
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

#: Canonical class ordering used for reporting and probability alignment.
CLASS_ORDER: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")

#: Sampling frequency used throughout the project -- MUST match
#: signal_generator.py / feature_extraction.py.
SAMPLING_FREQUENCY: float = 20000.0


@dataclass
class PredictConfig:
    """Central configuration for the prediction pipeline.

    Attributes:
        models_dir: Directory containing the trained model artifacts.
        sampling_frequency: Sampling frequency (Hz) used for
            preprocessing and feature extraction. MUST match the value
            used when ``dataset/features.csv`` was generated.
    """

    models_dir: Path = field(default_factory=lambda: Path("trained_models"))
    sampling_frequency: float = SAMPLING_FREQUENCY

    def artifact_paths(self) -> Dict[str, Path]:
        """Returns the expected paths of every required artifact.

        Returns:
            Dictionary mapping a short artifact key to its full path.
        """
        return {
            "best_model": self.models_dir / "best_model.joblib",
            "label_encoder": self.models_dir / "label_encoder.joblib",
            "training_summary": self.models_dir / "training_summary.json",
        }


# ==============================================================================
# SIGNAL LOADING & SHAPE VALIDATION
# ==============================================================================

class SignalLoader:
    """Loads a raw ``.npy`` signal file and validates/normalizes its shape.

    Detailed numeric validation (NaN/Inf/zero-energy checks) is left to
    ``preprocessing.SignalPreprocessor`` -- this class only handles
    file-level loading and the 1-D shape contract the preprocessor
    requires.
    """

    def load(self, path: Path) -> np.ndarray:
        """Loads a ``.npy`` file and returns a validated 1-D numeric array.

        Args:
            path: Path to the ``.npy`` signal file.

        Returns:
            A 1-D NumPy array (dtype coercible to float64).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            SignalLoadError: If the file is not a valid ``.npy`` file,
                is empty, is non-numeric, or has an incompatible shape
                that cannot be safely squeezed to 1-D.
        """
        if not path.exists():
            raise FileNotFoundError(f"Input signal file not found: '{path}'.")

        if path.suffix.lower() != ".npy":
            raise SignalLoadError(
                f"Expected a '.npy' file, got '{path.suffix}' ('{path}')."
            )

        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise SignalLoadError(f"Failed to load '.npy' file '{path}': {exc}") from exc

        if not isinstance(array, np.ndarray):
            raise SignalLoadError(f"'{path}' does not contain a NumPy array.")

        if array.size == 0:
            raise SignalLoadError(f"Signal in '{path}' is empty (zero samples).")

        if not np.issubdtype(array.dtype, np.number):
            raise SignalLoadError(
                f"Signal in '{path}' is non-numeric (dtype={array.dtype})."
            )

        array = self._to_1d(array, path)

        return array.astype(np.float64, copy=False)

    @staticmethod
    def _to_1d(array: np.ndarray, path: Path) -> np.ndarray:
        """Safely squeezes a compatible array to 1-D, rejecting the rest.

        Accepts arrays that are already 1-D, or 2-D arrays with a
        singleton dimension (shape ``(N, 1)`` or ``(1, N)``). Any other
        shape (e.g. ``(N, M)`` with both dimensions > 1, or more than 2
        dimensions) is rejected as genuinely incompatible.

        Args:
            array: The loaded array.
            path: Original file path, used only for error messages.

        Returns:
            A 1-D view/copy of ``array``.

        Raises:
            SignalLoadError: If the array's shape cannot be safely
                reduced to 1-D.
        """
        if array.ndim == 1:
            return array

        if array.ndim == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
            return array.reshape(-1)

        raise SignalLoadError(
            f"Signal in '{path}' has an incompatible shape {array.shape}. "
            "Expected a 1-D array, or a 2-D array with a singleton dimension "
            "(e.g. (N, 1) or (1, N))."
        )


# ==============================================================================
# ARTIFACT LOADING (read-only; NEVER fits/retrains/saves anything)
# ==============================================================================

class ArtifactBundle:
    """Holds every artifact needed for prediction, loaded once and reused.

    Attributes:
        model: The prediction-ready best-model estimator
            (``best_model.joblib``). Directly usable on raw
            (unscaled) feature vectors in ``FEATURE_COLUMNS`` order.
        label_encoder: The ``LabelEncoder`` fit during training, used
            only to decode integer-encoded predictions/probability
            columns (e.g. for an XGBoost best model).
        best_model_name: The model name reported by
            ``training_summary.json`` (for display purposes only --
            never used to branch prediction logic).
    """

    def __init__(self, model: Any, label_encoder: Any, best_model_name: str) -> None:
        self.model = model
        self.label_encoder = label_encoder
        self.best_model_name = best_model_name


class ArtifactLoader:
    """Loads the already-trained best-model artifact, label encoder, and
    training summary. Never fits, retrains, or overwrites anything.
    """

    def __init__(self, config: PredictConfig) -> None:
        """Initializes the loader.

        Args:
            config: Prediction configuration describing where trained
                artifacts live.
        """
        self._config = config

    def validate_files_exist(self) -> None:
        """Validates that every required artifact file exists on disk.

        Raises:
            ArtifactLoadError: If any required artifact is missing.
        """
        missing = [
            str(path) for path in self._config.artifact_paths().values() if not path.exists()
        ]
        if missing:
            raise ArtifactLoadError(
                f"Missing required trained artifact(s): {missing}. "
                "Run train_models.py first to generate them."
            )
        logger.info("All required trained artifacts were found.")

    def load(self) -> ArtifactBundle:
        """Loads the best model, label encoder, and training summary.

        Returns:
            A populated :class:`ArtifactBundle`.

        Raises:
            ArtifactLoadError: If any artifact fails to load or parse.
        """
        paths = self._config.artifact_paths()

        try:
            model = joblib.load(paths["best_model"])
            label_encoder = joblib.load(paths["label_encoder"])
        except (OSError, ValueError, EOFError) as exc:
            raise ArtifactLoadError(f"Failed to load trained artifact(s): {exc}") from exc

        try:
            with open(paths["training_summary"], "r", encoding="utf-8") as fh:
                training_summary = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactLoadError(
                f"Failed to load '{paths['training_summary']}': {exc}"
            ) from exc

        best_model_name = training_summary.get("best_model", "Unknown")
        logger.info("Loaded best model artifact (reported as '%s').", best_model_name)

        return ArtifactBundle(model=model, label_encoder=label_encoder, best_model_name=best_model_name)


# ==============================================================================
# PREDICTION RESULT
# ==============================================================================

@dataclass
class PredictionResult:
    """Structured result of a single-signal prediction.

    Attributes:
        input_path: String path to the input ``.npy`` file.
        predicted_modulation: The predicted class label.
        confidence: Probability assigned to the predicted class, in
            ``[0, 1]``. ``None`` if the model does not support
            ``predict_proba``.
        class_probabilities: Mapping of every class in ``CLASS_ORDER``
            to its predicted probability, in ``[0, 1]``. ``None`` if
            unavailable.
        features: Mapping of each of the 12 feature names (in
            ``FEATURE_COLUMNS`` order) to its extracted value.
        model_name: Display name of the model used for prediction, as
            reported by ``training_summary.json``.
    """

    input_path: str
    predicted_modulation: str
    confidence: Optional[float]
    class_probabilities: Optional[Dict[str, float]]
    features: Dict[str, float]
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts the result into a plain, JSON-serializable dictionary.

        Returns:
            A dictionary with the same information as this dataclass.
        """
        return {
            "input_path": self.input_path,
            "predicted_modulation": self.predicted_modulation,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "features": self.features,
            "model_name": self.model_name,
        }


# ==============================================================================
# SIGNAL PREDICTOR
# ==============================================================================

class SignalPredictor:
    """End-to-end predictor for a single unknown analog-modulated signal.

    Wires together the existing, unmodified preprocessing and feature
    extraction pipelines with the already-trained best-model artifact.
    Instantiate once and reuse across multiple predictions -- artifacts
    and the preprocessor/extractor are loaded/initialized only once in
    ``__init__``.
    """

    def __init__(self, config: Optional[PredictConfig] = None) -> None:
        """Initializes the predictor: loads artifacts and builds the
        preprocessing/feature-extraction pipeline.

        Args:
            config: Optional prediction configuration. Defaults to
                :class:`PredictConfig` defaults if omitted.

        Raises:
            ArtifactLoadError: If required trained artifacts are
                missing or fail to load.
        """
        self._config = config or PredictConfig()

        artifact_loader = ArtifactLoader(self._config)
        artifact_loader.validate_files_exist()
        self._artifacts = artifact_loader.load()

        # Reuse the EXISTING preprocessing pipeline, with the SAME
        # configuration used during feature_extraction.py's dataset build:
        # fs=20000 Hz, remove_dc=True, normalize=True, apply_filter=False.
        preprocessing_config = PreprocessingConfig(
            sampling_frequency=self._config.sampling_frequency,
            remove_dc=True,
            normalize=True,
            apply_filter=False,
        )
        self._preprocessor = SignalPreprocessor(preprocessing_config)

        # Reuse the EXISTING feature extractor; no feature math is
        # reimplemented in this module.
        self._extractor = FeatureExtractor(
            sampling_frequency=self._config.sampling_frequency,
            preprocessing_config=preprocessing_config,
        )

        self._signal_loader = SignalLoader()

        logger.info(
            "SignalPredictor ready (best model: '%s').", self._artifacts.best_model_name
        )

    def predict(self, signal_path: Path) -> PredictionResult:
        """Runs the full prediction pipeline on a single ``.npy`` signal.

        Args:
            signal_path: Path to the unknown ``.npy`` signal file.

        Returns:
            A populated :class:`PredictionResult`.

        Raises:
            FileNotFoundError: If ``signal_path`` does not exist.
            SignalLoadError: If the signal cannot be loaded or has an
                incompatible shape/dtype.
            PredictionError: If preprocessing, feature extraction, or
                model prediction fails.
        """
        signal_path = Path(signal_path)

        # ---- Load & shape-validate the raw signal --------------------------
        raw_signal = self._signal_loader.load(signal_path)

        # ---- Preprocess (existing SignalPreprocessor, unmodified) ---------
        try:
            processed_signal = self._preprocessor.process(raw_signal)
        except SignalValidationError as exc:
            raise PredictionError(f"Preprocessing failed for '{signal_path}': {exc}") from exc

        # ---- Extract features (existing FeatureExtractor, unmodified) -----
        try:
            feature_dict = self._extractor.extract_features(processed_signal)
        except FeatureExtractionError as exc:
            raise PredictionError(
                f"Feature extraction failed for '{signal_path}': {exc}"
            ) from exc

        # ---- Assemble the feature vector in the EXACT training order -------
        # A single-row DataFrame (not a bare array) is used so that models
        # expecting named columns (as they were trained on) receive
        # consistent input.
        import pandas as pd  # local import keeps the module lightweight when unused

        feature_row = pd.DataFrame(
            [[feature_dict[name] for name in FEATURE_COLUMNS]], columns=list(FEATURE_COLUMNS)
        )

        # ---- Predict using the already-trained best model ------------------
        predicted_label, confidence, class_probabilities = self._predict_with_model(feature_row)

        return PredictionResult(
            input_path=str(signal_path),
            predicted_modulation=predicted_label,
            confidence=confidence,
            class_probabilities=class_probabilities,
            features=feature_dict,
            model_name=self._artifacts.best_model_name,
        )

    def _predict_with_model(
        self, feature_row: "pd.DataFrame"
    ) -> Tuple[str, Optional[float], Optional[Dict[str, float]]]:
        """Predicts the class (+ optional confidence/probabilities).

        Handles both directly human-readable models (Random Forest,
        SVM -- whose ``classes_`` are the string labels themselves) and
        integer-encoded models (e.g. XGBoost, whose ``classes_`` are
        integers matching ``label_encoder.classes_``), WITHOUT hard-
        coding any specific model name: the branch is chosen purely by
        inspecting the dtype of the model's ``classes_`` attribute.

        Args:
            feature_row: Single-row feature DataFrame, columns in
                ``FEATURE_COLUMNS`` order.

        Returns:
            A tuple ``(predicted_label, confidence, class_probabilities)``.
            ``confidence`` and ``class_probabilities`` are ``None`` if
            the model does not support ``predict_proba``.

        Raises:
            PredictionError: If the model has no usable ``classes_``
                attribute, or prediction fails.
        """
        model = self._artifacts.model
        label_encoder = self._artifacts.label_encoder

        if not hasattr(model, "classes_"):
            raise PredictionError(
                "The loaded best model does not expose a 'classes_' attribute; "
                "cannot determine class ordering."
            )

        try:
            raw_prediction = model.predict(feature_row)
        except (ValueError, AttributeError) as exc:
            raise PredictionError(f"Model prediction failed: {exc}") from exc

        is_integer_encoded = np.issubdtype(np.asarray(model.classes_).dtype, np.integer)

        if is_integer_encoded:
            # Integer-encoded model (e.g. XGBoost) -> decode via the saved
            # LabelEncoder. predict_proba's columns follow the same
            # integer-class order as model.classes_, which corresponds to
            # label_encoder.classes_ (both are 0..K-1 sorted alphabetically).
            predicted_label = str(label_encoder.inverse_transform(raw_prediction)[0])
            proba_class_order = list(label_encoder.classes_)
        else:
            predicted_label = str(raw_prediction[0])
            proba_class_order = list(model.classes_)

        if not hasattr(model, "predict_proba"):
            logger.warning("Best model does not support predict_proba(); confidence unavailable.")
            return predicted_label, None, None

        try:
            proba_raw = model.predict_proba(feature_row)
        except (ValueError, AttributeError) as exc:
            raise PredictionError(f"Failed to compute prediction probabilities: {exc}") from exc

        proba_aligned = self._reorder_proba(proba_raw, proba_class_order)[0]
        class_probabilities = {
            class_name: float(prob) for class_name, prob in zip(CLASS_ORDER, proba_aligned)
        }
        confidence = float(class_probabilities[predicted_label])

        return predicted_label, confidence, class_probabilities

    @staticmethod
    def _reorder_proba(proba: np.ndarray, source_class_order: List[Any]) -> np.ndarray:
        """Reorders a probability matrix's columns to match ``CLASS_ORDER``.

        Args:
            proba: Raw probability matrix as returned by
                ``predict_proba``, with columns in ``source_class_order``.
            source_class_order: The class order corresponding to
                ``proba``'s columns (``model.classes_`` or
                ``label_encoder.classes_``, converted to plain
                strings/ints as appropriate).

        Returns:
            The probability matrix with columns reordered so column
            ``i`` corresponds to ``CLASS_ORDER[i]``.
        """
        # Normalize to strings for lookup against CLASS_ORDER when the
        # source order came from label_encoder.classes_ (already strings)
        # or model.classes_ (already strings for RF/SVM).
        normalized_order = [str(cls) for cls in source_class_order]
        column_indices = [normalized_order.index(cls) for cls in CLASS_ORDER]
        return proba[:, column_indices]


# ==============================================================================
# REUSABLE PUBLIC API
# ==============================================================================

def predict_signal(signal_path: Path | str, config: Optional[PredictConfig] = None) -> PredictionResult:
    """Convenience function to predict a single signal's modulation class.

    Creates a new :class:`SignalPredictor` (loading artifacts and
    building the preprocessing/feature-extraction pipeline) and runs a
    single prediction. For repeated predictions, instantiate
    :class:`SignalPredictor` once directly instead, to avoid reloading
    artifacts on every call.

    Args:
        signal_path: Path to the unknown ``.npy`` signal file.
        config: Optional prediction configuration.

    Returns:
        A populated :class:`PredictionResult`.

    Raises:
        FileNotFoundError: If ``signal_path`` does not exist.
        SignalLoadError: If the signal is invalid.
        ArtifactLoadError: If required trained artifacts are missing.
        PredictionError: If preprocessing, feature extraction, or
            prediction fails.
    """
    predictor = SignalPredictor(config)
    return predictor.predict(Path(signal_path))


# ==============================================================================
# TERMINAL REPORTING
# ==============================================================================

def print_prediction_report(result: PredictionResult) -> None:
    """Prints a clean, human-readable prediction report to the terminal.

    Args:
        result: The prediction result to display.
    """
    line = "=" * 60
    print(f"\n{line}")
    print("ANALOG MODULATION PREDICTION")
    print(line)
    print(f"\nInput Signal : {result.input_path}")

    print("\nExtracted Features:")
    max_label_len = max(len(name) for name in FEATURE_COLUMNS)
    for name in FEATURE_COLUMNS:
        value = result.features[name]
        print(f"{name.ljust(max_label_len)}: {value:.6f}")

    print("\n" + "-" * 60)
    print("PREDICTION")
    print("-" * 60 + "\n")
    print(f"Predicted Modulation : {result.predicted_modulation}")

    if result.confidence is not None:
        print(f"Confidence            : {result.confidence * 100:.2f}%")
    else:
        print("Confidence            : N/A (model does not support predict_proba)")

    if result.class_probabilities is not None:
        print("\nClass Probabilities:")
        max_class_len = max(len(name) for name in CLASS_ORDER)
        for class_name in CLASS_ORDER:
            prob = result.class_probabilities[class_name]
            print(f"{class_name.ljust(max_class_len)}  : {prob * 100:.2f}%")

    print(f"\n{line}\n")


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser.

    Returns:
        A configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "Predict the analog modulation class (AM, FM, DSB_SC, SSB) of an "
            "unknown .npy signal file using the trained best model."
        ),
    )
    parser.add_argument(
        "signal_path",
        type=str,
        help="Path to the unknown .npy signal file (e.g. dataset/AM/AM_0001.npy).",
    )
    return parser


def main() -> None:
    """Standalone entry point: predicts the modulation of a single signal file.

    Running::

        python predict.py <path_to_npy_file>

    loads the trained best model, preprocesses and extracts features
    from the given signal using the existing pipeline modules, prints a
    full prediction report, and exits. Never retrains, never modifies
    any dataset or model file, and never requires interactive input.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = PredictConfig()

    try:
        predictor = SignalPredictor(config)
        result = predictor.predict(Path(args.signal_path))
    except FileNotFoundError as exc:
        logger.error("Input file error: %s", exc)
        sys.exit(1)
    except SignalLoadError as exc:
        logger.error("Signal loading failed: %s", exc)
        sys.exit(1)
    except ArtifactLoadError as exc:
        logger.error("Artifact loading failed: %s", exc)
        sys.exit(1)
    except PredictionError as exc:
        logger.error("Prediction failed: %s", exc)
        sys.exit(1)

    print_prediction_report(result)


if __name__ == "__main__":
    main()