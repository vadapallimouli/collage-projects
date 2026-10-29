"""
train_models.py
================

Analog Modulation Classifier Using Machine Learning
-----------------------------------------------------

This module is responsible EXCLUSIVELY for:

    1. Loading the extracted feature dataset (``dataset/features.csv``,
       produced by ``feature_extraction.py``).
    2. Validating that dataset.
    3. Splitting it into stratified 80/20 train/test partitions.
    4. Training exactly three traditional ML classifiers:
         - Random Forest   (sklearn.ensemble.RandomForestClassifier)
         - SVM             (sklearn.svm.SVC)
         - XGBoost         (xgboost.XGBClassifier)
    5. Computing test-set performance metrics (accuracy, precision,
       recall, weighted F1) for each model.
    6. Selecting the best model (by weighted F1, tie-broken by
       accuracy) using TEST performance only.
    7. Persisting all trained models, the SVM scaler, the label
       encoder, a prediction-ready ``best_model.joblib`` artifact, and
       a machine-readable ``training_summary.json`` under
       ``trained_models/``.

It does NOT generate evaluation plots (confusion matrices, ROC curves,
feature-importance charts) -- those belong to ``evaluate_models.py``.
It does NOT implement prediction logic -- that belongs to
``predict.py``. It NEVER modifies ``dataset/features.csv``.

Pipeline stage handled here::

    dataset/features.csv
            |
            v
    load + validate
            |
            v
    stratified 80/20 train/test split (BEFORE any scaler is fit)
            |
            v
    train Random Forest / SVM (+StandardScaler) / XGBoost (+LabelEncoder)
            |
            v
    evaluate on held-out test set (accuracy/precision/recall/F1)
            |
            v
    select best model by weighted F1 (tie-break: accuracy)
            |
            v
    trained_models/*.joblib + trained_models/training_summary.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

def _build_logger(name: str = "train_models") -> logging.Logger:
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

class DatasetValidationError(Exception):
    """Raised when the loaded feature dataset fails a validation check."""


class ModelTrainingError(Exception):
    """Raised when a specific model fails to train."""


class ArtifactSaveError(Exception):
    """Raised when a trained model or summary artifact cannot be saved."""


# ==============================================================================
# CONSTANTS / CONFIGURATION
# ==============================================================================

#: Exact, ordered list of feature columns expected in features.csv.
#: This order MUST be preserved -- it is the order every downstream
#: model (and predict.py, later) expects feature vectors to arrive in.
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

LABEL_COLUMN: str = "Label"

EXPECTED_COLUMNS: Tuple[str, ...] = FEATURE_COLUMNS + (LABEL_COLUMN,)

EXPECTED_CLASSES: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")

#: Fixed random seed used everywhere reproducibility matters: the
#: train/test split, RandomForest, SVC, and XGBClassifier.
RANDOM_STATE: int = 42

#: Fraction of the dataset held out for testing.
TEST_SIZE: float = 0.20


@dataclass
class TrainingConfig:
    """Central configuration for the training pipeline.

    Attributes:
        dataset_path: Path to the input ``features.csv`` file.
        output_dir: Directory in which trained models and the summary
            JSON are saved.
        random_state: Random seed used for the split and every model
            that supports one, ensuring reproducible results.
        test_size: Fraction of the dataset reserved for the held-out
            test set.
    """

    dataset_path: Path = field(default_factory=lambda: Path("dataset") / "features.csv")
    output_dir: Path = field(default_factory=lambda: Path("trained_models"))
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE


# ==============================================================================
# DATASET LOADING & VALIDATION
# ==============================================================================

class DatasetLoader:
    """Loads and validates the feature dataset produced by
    ``feature_extraction.py``.

    All validation failures raise :class:`DatasetValidationError` with
    an informative message. Nothing is ever silently repaired (e.g. no
    NaN filling) -- invalid data must be fixed upstream.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initializes the loader.

        Args:
            config: Training configuration describing where the
                dataset lives.
        """
        self._config = config

    def load(self) -> pd.DataFrame:
        """Loads and fully validates ``features.csv``.

        Returns:
            The validated feature dataframe, with columns in exactly
            the order ``FEATURE_COLUMNS + (LABEL_COLUMN,)``.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            DatasetValidationError: If any validation check fails.
        """
        dataset_path = self._config.dataset_path

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found at '{dataset_path}'. "
                "Run feature_extraction.py first to generate it."
            )

        logger.info("Loading dataset from '%s'.", dataset_path)
        dataframe = pd.read_csv(dataset_path)

        self._validate_columns(dataframe)
        self._validate_no_missing_values(dataframe)
        self._validate_no_infinite_values(dataframe)
        self._validate_feature_dtypes(dataframe)
        self._validate_classes(dataframe)

        # Reorder columns defensively to the canonical order, in case
        # the CSV on disk ever has columns in a different order.
        dataframe = dataframe[list(EXPECTED_COLUMNS)]

        logger.info("Dataset shape: %s.", dataframe.shape)
        self._log_class_distribution(dataframe)

        return dataframe

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame) -> None:
        """Ensures every expected column is present.

        Args:
            dataframe: The loaded dataframe.

        Raises:
            DatasetValidationError: If any expected column is missing.
        """
        missing = [col for col in EXPECTED_COLUMNS if col not in dataframe.columns]
        if missing:
            raise DatasetValidationError(
                f"Feature dataset is missing required column(s): {missing}. "
                f"Expected columns: {list(EXPECTED_COLUMNS)}."
            )

    @staticmethod
    def _validate_no_missing_values(dataframe: pd.DataFrame) -> None:
        """Ensures no NaN values exist anywhere in the expected columns.

        Args:
            dataframe: The loaded dataframe.

        Raises:
            DatasetValidationError: If any NaN value is found.
        """
        subset = dataframe[list(EXPECTED_COLUMNS)]
        if subset.isnull().values.any():
            null_counts = subset.isnull().sum()
            offending = null_counts[null_counts > 0].to_dict()
            raise DatasetValidationError(
                f"Feature dataset contains NaN value(s) in column(s): {offending}. "
                "Refusing to silently fill missing values."
            )

    @staticmethod
    def _validate_no_infinite_values(dataframe: pd.DataFrame) -> None:
        """Ensures no infinite values exist in the numeric feature columns.

        Args:
            dataframe: The loaded dataframe.

        Raises:
            DatasetValidationError: If any infinite value is found.
        """
        numeric = dataframe[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise DatasetValidationError(
                "Feature dataset contains infinite (+/-inf) value(s) in one or "
                "more feature columns."
            )

    @staticmethod
    def _validate_feature_dtypes(dataframe: pd.DataFrame) -> None:
        """Ensures every feature column is numeric.

        Args:
            dataframe: The loaded dataframe.

        Raises:
            DatasetValidationError: If a feature column is not numeric.
        """
        non_numeric = [
            col for col in FEATURE_COLUMNS
            if not pd.api.types.is_numeric_dtype(dataframe[col])
        ]
        if non_numeric:
            raise DatasetValidationError(
                f"Feature column(s) are not numeric: {non_numeric}."
            )

    @staticmethod
    def _validate_classes(dataframe: pd.DataFrame) -> None:
        """Ensures exactly four expected classes are present, each with samples.

        Args:
            dataframe: The loaded dataframe.

        Raises:
            DatasetValidationError: If the class set is unexpected or a
                class has zero samples.
        """
        class_counts = dataframe[LABEL_COLUMN].value_counts()
        found_classes = set(class_counts.index)
        expected_classes = set(EXPECTED_CLASSES)

        if found_classes != expected_classes:
            raise DatasetValidationError(
                f"Unexpected class labels found: {sorted(found_classes)}. "
                f"Expected exactly: {sorted(expected_classes)}."
            )

        empty_classes = [cls for cls, count in class_counts.items() if count <= 0]
        if empty_classes:
            raise DatasetValidationError(
                f"Class(es) with zero samples detected: {empty_classes}."
            )

    @staticmethod
    def _log_class_distribution(dataframe: pd.DataFrame) -> None:
        """Logs the per-class sample counts.

        Args:
            dataframe: The loaded (validated) dataframe.
        """
        counts = dataframe[LABEL_COLUMN].value_counts().sort_index()
        logger.info("Class distribution (before splitting):")
        for label, count in counts.items():
            logger.info("    %-10s %d", label, count)


# ==============================================================================
# MODEL RESULT CONTAINER
# ==============================================================================

@dataclass
class ModelResult:
    """Holds the trained estimator plus its evaluation results for one model.

    Attributes:
        name: Human-readable model name (e.g. ``"Random Forest"``).
        estimator: The fitted estimator (or Pipeline) ready for
            prediction on raw (unscaled) feature vectors in
            ``FEATURE_COLUMNS`` order.
        training_time_seconds: Wall-clock training time.
        accuracy: Test-set accuracy.
        precision: Test-set weighted precision.
        recall: Test-set weighted recall.
        f1: Test-set weighted F1 score.
    """

    name: str
    estimator: Any
    training_time_seconds: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def metrics_dict(self) -> Dict[str, float]:
        """Returns the numeric metrics as a JSON-serializable dict.

        Returns:
            Dictionary with accuracy, precision, recall, f1, and
            training time.
        """
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "training_time_seconds": self.training_time_seconds,
        }


# ==============================================================================
# MODEL TRAINER
# ==============================================================================

class ModelTrainer:
    """Builds, trains, and evaluates the three required ML models.

    Each model is wrapped so that its saved artifact is directly
    usable on RAW (unscaled) feature vectors in ``FEATURE_COLUMNS``
    order:

        * Random Forest -- trained directly on raw features (tree-based
          models are invariant to monotonic feature scaling, so scaling
          would add complexity without benefit).
        * SVM -- wrapped in a ``sklearn.pipeline.Pipeline`` of
          ``StandardScaler -> SVC``. SVMs are distance/margin based, so
          features on vastly different scales (e.g. Energy vs. Zero
          Crossing Rate) would otherwise dominate the decision
          boundary. The pipeline guarantees the scaler is always
          applied consistently at both train and inference time.
        * XGBoost -- trained directly on raw features. Gradient-boosted
          trees split on feature thresholds and are likewise invariant
          to monotonic scaling.

    XGBoost requires numerically encoded integer class labels, so a
    ``LabelEncoder`` (fit ONLY on the training labels) is used to map
    between the human-readable string labels and the integer labels
    XGBoost trains on; predictions are decoded back to strings before
    metrics are computed so that all three models are compared on the
    same (string-label) footing.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initializes the trainer.

        Args:
            config: Training configuration (random state, test size,
                etc.).
        """
        self._config = config

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: np.ndarray,
    ) -> Tuple[RandomForestClassifier, float]:
        """Trains the Random Forest classifier on raw (unscaled) features.

        Args:
            X_train: Training feature matrix, raw (unscaled).
            y_train: Training labels (human-readable strings).

        Returns:
            A tuple ``(fitted_model, training_time_seconds)``.

        Raises:
            ModelTrainingError: If training fails.
        """
        logger.info("Training Random Forest ...")
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=self._config.random_state,
            n_jobs=-1,
        )

        start = time.perf_counter()
        try:
            model.fit(X_train, y_train)
        except (ValueError, RuntimeError) as exc:
            raise ModelTrainingError(f"Random Forest training failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        logger.info("Random Forest trained in %.3f seconds.", elapsed)
        return model, elapsed

    def train_svm(
        self, X_train: pd.DataFrame, y_train: np.ndarray,
    ) -> Tuple[Pipeline, float]:
        """Trains the SVM classifier inside a StandardScaler pipeline.

        The scaler is fit exclusively as part of ``pipeline.fit`` on
        ``X_train`` -- it never sees the test set, avoiding data
        leakage. ``probability=True`` is enabled so that later modules
        (evaluate_models.py, predict.py) can obtain class probabilities
        for confidence scores and ROC curves.

        Args:
            X_train: Training feature matrix, raw (unscaled). Scaling
                is performed internally by the pipeline.
            y_train: Training labels (human-readable strings).

        Returns:
            A tuple ``(fitted_pipeline, training_time_seconds)``.

        Raises:
            ModelTrainingError: If training fails.
        """
        logger.info("Training SVM (StandardScaler + SVC) ...")
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=self._config.random_state,
                    ),
                ),
            ]
        )

        start = time.perf_counter()
        try:
            pipeline.fit(X_train, y_train)
        except (ValueError, RuntimeError) as exc:
            raise ModelTrainingError(f"SVM training failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        logger.info("SVM trained in %.3f seconds.", elapsed)
        return pipeline, elapsed

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train_encoded: np.ndarray,
    ) -> Tuple[XGBClassifier, float]:
        """Trains the XGBoost classifier on raw (unscaled) features.

        Args:
            X_train: Training feature matrix, raw (unscaled).
            y_train_encoded: Training labels, integer-encoded via a
                ``LabelEncoder`` fit only on the training labels.

        Returns:
            A tuple ``(fitted_model, training_time_seconds)``.

        Raises:
            ModelTrainingError: If training fails.
        """
        logger.info("Training XGBoost ...")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self._config.random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

        start = time.perf_counter()
        try:
            model.fit(X_train, y_train_encoded)
        except (ValueError, RuntimeError) as exc:
            raise ModelTrainingError(f"XGBoost training failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        logger.info("XGBoost trained in %.3f seconds.", elapsed)
        return model, elapsed


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes multiclass-compatible weighted metrics on the test set.

    Weighted averaging is used (rather than macro or micro) so that
    each class's contribution to precision/recall/F1 is proportional
    to its support, which is appropriate for this project's balanced
    but potentially-imbalanced-in-practice four-class setup.

    Args:
        y_true: Ground-truth test labels.
        y_pred: Predicted test labels.

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        and ``f1``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ==============================================================================
# BEST MODEL SELECTION
# ==============================================================================

def select_best_model(results: List[ModelResult]) -> ModelResult:
    """Selects the best model using TEST weighted F1, tie-broken by accuracy.

    Training accuracy is deliberately never consulted -- only held-out
    test performance determines the best model, since training
    performance is not indicative of generalization.

    Args:
        results: List of trained model results.

    Returns:
        The :class:`ModelResult` judged best.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("Cannot select a best model from an empty results list.")

    best = max(results, key=lambda r: (r.f1, r.accuracy))
    logger.info(
        "Best model selected: '%s' (weighted F1=%.4f, accuracy=%.4f).",
        best.name,
        best.f1,
        best.accuracy,
    )
    return best


# ==============================================================================
# ARTIFACT SAVING
# ==============================================================================

class ArtifactSaver:
    """Persists trained models, preprocessing artifacts, and the summary
    JSON to ``trained_models/``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initializes the saver.

        Args:
            config: Training configuration describing the output
                directory.
        """
        self._config = config

    def prepare_output_dir(self) -> None:
        """Creates the output directory tree if it does not already exist.

        Raises:
            ArtifactSaveError: If the directory cannot be created.
        """
        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactSaveError(
                f"Failed to create output directory '{self._config.output_dir}': {exc}"
            ) from exc

    def save_joblib(self, obj: Any, filename: str) -> Path:
        """Saves a single object via ``joblib.dump``.

        Args:
            obj: The object to persist (estimator, pipeline, scaler,
                or encoder).
            filename: Target filename inside the output directory.

        Returns:
            The full path the object was saved to.

        Raises:
            ArtifactSaveError: If saving fails.
        """
        target_path = self._config.output_dir / filename
        try:
            joblib.dump(obj, target_path)
        except (OSError, ValueError) as exc:
            raise ArtifactSaveError(f"Failed to save '{target_path}': {exc}") from exc

        logger.info("Saved artifact: '%s'.", target_path)
        return target_path

    def save_summary(self, summary: Dict[str, Any], filename: str = "training_summary.json") -> Path:
        """Saves the JSON-serializable training summary.

        Args:
            summary: The summary dictionary to persist.
            filename: Target filename inside the output directory.

        Returns:
            The full path the summary was saved to.

        Raises:
            ArtifactSaveError: If saving fails.
        """
        target_path = self._config.output_dir / filename
        try:
            with open(target_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=4)
        except OSError as exc:
            raise ArtifactSaveError(f"Failed to save '{target_path}': {exc}") from exc

        logger.info("Saved training summary: '%s'.", target_path)
        return target_path


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def run_training_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    """Executes the full training pipeline end-to-end.

    Steps:
        1. Load and validate ``features.csv``.
        2. Separate features (X) from labels (y), preserving
           ``FEATURE_COLUMNS`` order.
        3. Perform a stratified 80/20 train/test split (BEFORE fitting
           any scaler or encoder).
        4. Train Random Forest, SVM (with internal StandardScaler), and
           XGBoost (with a LabelEncoder fit only on training labels).
        5. Evaluate all three models on the held-out test set.
        6. Select the best model by weighted F1 (tie-break: accuracy).
        7. Save every model, the label encoder, the best-model
           artifact, and a JSON training summary.

    Args:
        config: Training configuration.

    Returns:
        The training summary dictionary that was also written to
        ``training_summary.json``.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        DatasetValidationError: If the dataset fails validation.
        ModelTrainingError: If any model fails to train.
        ArtifactSaveError: If any artifact fails to save.
    """
    # ---- 1. Load & validate -------------------------------------------------
    loader = DatasetLoader(config)
    dataframe = loader.load()

    # ---- 2. Separate X / y ---------------------------------------------------
    X = dataframe[list(FEATURE_COLUMNS)].copy()
    y = dataframe[LABEL_COLUMN].to_numpy()

    # ---- 3. Stratified 80/20 split (performed BEFORE any scaler/encoder is
    #         fit, to prevent data leakage from the test set). --------------
    logger.info(
        "Performing stratified %.0f/%.0f train/test split (random_state=%d).",
        (1 - config.test_size) * 100,
        config.test_size * 100,
        config.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    logger.info(
        "Train set: %d samples | Test set: %d samples | Features: %d.",
        len(X_train),
        len(X_test),
        len(FEATURE_COLUMNS),
    )

    # ---- Label encoder for XGBoost, fit ONLY on training labels -----------
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    # Test labels are transformed (not fit) using the encoder learned from
    # training data only; since the split is stratified, every class is
    # guaranteed to appear in y_train, so this transform is always valid.
    y_test_encoded = label_encoder.transform(y_test)
    logger.info("Label encoder classes: %s.", list(label_encoder.classes_))

    # ---- 4. Train all three models ------------------------------------------
    trainer = ModelTrainer(config)
    results: List[ModelResult] = []
    trained_estimators: Dict[str, Any] = {}

    # --- Random Forest (raw features) ---
    rf_model, rf_time = trainer.train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_pred)
    results.append(
        ModelResult(name="Random Forest", estimator=rf_model, training_time_seconds=rf_time, **rf_metrics)
    )
    trained_estimators["Random Forest"] = rf_model

    # --- SVM (StandardScaler + SVC pipeline, raw features in / scaled internally) ---
    svm_pipeline, svm_time = trainer.train_svm(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    svm_metrics = evaluate_predictions(y_test, svm_pred)
    results.append(
        ModelResult(name="SVM", estimator=svm_pipeline, training_time_seconds=svm_time, **svm_metrics)
    )
    trained_estimators["SVM"] = svm_pipeline

    # --- XGBoost (raw features, encoded labels) ---
    xgb_model, xgb_time = trainer.train_xgboost(X_train, y_train_encoded)
    xgb_pred_encoded = xgb_model.predict(X_test)
    xgb_pred = label_encoder.inverse_transform(xgb_pred_encoded)
    xgb_metrics = evaluate_predictions(y_test, xgb_pred)
    results.append(
        ModelResult(name="XGBoost", estimator=xgb_model, training_time_seconds=xgb_time, **xgb_metrics)
    )
    trained_estimators["XGBoost"] = xgb_model

    for result in results:
        logger.info(
            "Model performance -- %-14s | Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f (%.3fs)",
            result.name,
            result.accuracy,
            result.precision,
            result.recall,
            result.f1,
            result.training_time_seconds,
        )

    # ---- 5/6. Select best model (TEST weighted F1, tie-break accuracy) -----
    best_result = select_best_model(results)

    # Build a prediction-ready artifact for the best model. For SVM this is
    # already a Pipeline (StandardScaler + SVC) that accepts raw feature
    # vectors directly. For Random Forest and XGBoost the raw estimator IS
    # the complete prediction-ready artifact, since neither requires
    # scaling. XGBoost predictions must additionally be decoded back to
    # human-readable labels downstream (predict.py), using the saved
    # label_encoder.joblib.
    best_model_artifact = best_result.estimator

    # ---- 7. Save artifacts ----------------------------------------------
    saver = ArtifactSaver(config)
    saver.prepare_output_dir()

    logger.info("Saving trained models and preprocessing artifacts ...")
    saver.save_joblib(trained_estimators["Random Forest"], "random_forest.joblib")
    saver.save_joblib(trained_estimators["SVM"], "svm.joblib")
    saver.save_joblib(trained_estimators["XGBoost"], "xgboost.joblib")
    saver.save_joblib(label_encoder, "label_encoder.joblib")
    saver.save_joblib(best_model_artifact, "best_model.joblib")

    # ---- Build & save the JSON-compatible training summary -----------------
    summary: Dict[str, Any] = {
        "dataset": {
            "path": str(config.dataset_path),
            "samples": int(len(dataframe)),
            "features": len(FEATURE_COLUMNS),
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "class_names": list(label_encoder.classes_),
        "split": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "models": {result.name: result.metrics_dict() for result in results},
        "best_model": best_result.name,
        "best_model_selection_criteria": {
            "primary": "weighted_f1",
            "tie_breaker": "accuracy",
            "evaluated_on": "test_set",
        },
        "artifacts": {
            "random_forest": "random_forest.joblib",
            "svm": "svm.joblib",
            "xgboost": "xgboost.joblib",
            "label_encoder": "label_encoder.joblib",
            "best_model": "best_model.joblib",
        },
    }

    saver.save_summary(summary)

    logger.info("Training pipeline complete. Best model: '%s'.", best_result.name)
    return summary


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """Standalone entry point: runs the full training pipeline.

    Running ``python train_models.py`` loads ``dataset/features.csv``,
    validates it, trains all three required models, evaluates them,
    selects the best one, and saves every artifact under
    ``trained_models/``. Never regenerates signals or features, never
    modifies ``dataset/features.csv``, and never prompts for
    interactive input.
    """
    config = TrainingConfig()

    try:
        summary = run_training_pipeline(config)
    except FileNotFoundError as exc:
        logger.error("Dataset file error: %s", exc)
        raise
    except DatasetValidationError as exc:
        logger.error("Dataset validation failed: %s", exc)
        raise
    except ModelTrainingError as exc:
        logger.error("Model training failed: %s", exc)
        raise
    except ArtifactSaveError as exc:
        logger.error("Artifact saving failed: %s", exc)
        raise

    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("Dataset samples : %d", summary["dataset"]["samples"])
    logger.info("Train / Test    : %d / %d", summary["split"]["train_samples"], summary["split"]["test_samples"])
    logger.info("Classes         : %s", summary["class_names"])
    for model_name, metrics in summary["models"].items():
        logger.info(
            "%-14s Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
    logger.info("Best model      : %s", summary["best_model"])
    logger.info("Artifacts saved to: %s", config.output_dir.resolve())


if __name__ == "__main__":
    main()