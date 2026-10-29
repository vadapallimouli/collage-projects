"""
evaluate_models.py
===================

Analog Modulation Classifier Using Machine Learning
-----------------------------------------------------

This module is responsible EXCLUSIVELY for:

    1. Reproducing the EXACT held-out test split used by
       ``train_models.py`` (stratified 80/20 split,
       ``random_state=42``) from ``dataset/features.csv``.
    2. Loading the already-trained model artifacts saved by
       ``train_models.py`` under ``trained_models/`` (Random Forest,
       SVM, XGBoost, LabelEncoder, best_model, training_summary.json).
    3. Evaluating each of the three models on that SAME held-out test
       set (accuracy, weighted precision/recall/F1, full
       classification report, confusion matrix).
    4. Generating and saving evaluation visualizations (confusion
       matrices, accuracy comparison, Random Forest feature
       importance, multiclass One-vs-Rest ROC curves).
    5. Persisting a machine-readable ``results/evaluation_summary.json``
       and printing a clean terminal summary.

It does NOT retrain any model, does NOT fit any scaler/encoder, does
NOT create a new model, does NOT modify ``dataset/features.csv``, and
does NOT alter any convention (feature order, class names, split
parameters) established by ``train_models.py``. It is a pure,
read-only consumer of the artifacts ``train_models.py`` produces.

Pipeline stage handled here::

    dataset/features.csv
            |
            v
    reproduce IDENTICAL stratified 80/20 split (random_state=42)
            |
            v
    load trained_models/*.joblib (NEVER refit)
            |
            v
    predict on X_test ONLY
            |
            v
    metrics + classification reports + confusion matrices
            |
            v
    results/*.png + results/evaluation_summary.json

Running ``python evaluate_models.py`` executes the complete evaluation
automatically, with no manual/interactive input required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # Headless backend, safe for servers / notebooks.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


# ==============================================================================
# LOGGING CONFIGURATION (mirrors train_models.py)
# ==============================================================================

def _build_logger(name: str = "evaluate_models") -> logging.Logger:
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


class ArtifactLoadError(Exception):
    """Raised when a required trained-model artifact cannot be found or loaded."""


class EvaluationError(Exception):
    """Raised when evaluation of a specific model fails."""


# ==============================================================================
# CONSTANTS / CONFIGURATION
# ==============================================================================
#
# These constants MUST stay in lockstep with train_models.py. They are
# duplicated here (rather than imported) so that evaluate_models.py
# remains a fully self-contained module, per project requirements.

#: Exact, ordered list of feature columns expected in features.csv.
#: This order MUST match train_models.py exactly.
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

#: Canonical class display/ordering used for confusion matrices, ROC
#: curves, and reporting throughout this module.
CLASS_ORDER: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")

EXPECTED_CLASSES: Tuple[str, ...] = CLASS_ORDER

#: Fixed random seed -- MUST match train_models.py exactly, or the
#: reproduced split will not match the one models were trained/tested on.
RANDOM_STATE: int = 42

#: Fraction of the dataset held out for testing -- MUST match train_models.py.
TEST_SIZE: float = 0.20

MODEL_DISPLAY_NAMES: Tuple[str, ...] = ("Random Forest", "SVM", "XGBoost")


@dataclass
class EvaluationConfig:
    """Central configuration for the evaluation pipeline.

    Attributes:
        dataset_path: Path to ``features.csv``.
        models_dir: Directory containing the trained model artifacts.
        results_dir: Directory evaluation outputs (plots, JSON summary)
            are written to.
        random_state: Random seed used to reproduce the train/test
            split. MUST match ``train_models.py``.
        test_size: Fraction of the dataset reserved for the held-out
            test set. MUST match ``train_models.py``.
    """

    dataset_path: Path = field(default_factory=lambda: Path("dataset") / "features.csv")
    models_dir: Path = field(default_factory=lambda: Path("trained_models"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE

    def artifact_paths(self) -> Dict[str, Path]:
        """Returns the expected paths of every required artifact.

        Returns:
            Dictionary mapping a short artifact key to its full path.
        """
        return {
            "random_forest": self.models_dir / "random_forest.joblib",
            "svm": self.models_dir / "svm.joblib",
            "xgboost": self.models_dir / "xgboost.joblib",
            "label_encoder": self.models_dir / "label_encoder.joblib",
            "best_model": self.models_dir / "best_model.joblib",
            "training_summary": self.models_dir / "training_summary.json",
        }


# ==============================================================================
# DATASET LOADING & VALIDATION (read-only; NEVER modifies features.csv)
# ==============================================================================

class DatasetLoader:
    """Loads and validates ``features.csv`` for evaluation purposes only.

    This class never writes to, fills, or otherwise mutates the
    dataset on disk. Validation failures raise
    :class:`DatasetValidationError` with an informative message.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initializes the loader.

        Args:
            config: Evaluation configuration describing where the
                dataset lives.
        """
        self._config = config

    def load(self) -> pd.DataFrame:
        """Loads and validates ``features.csv``.

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
        self._validate_classes(dataframe)

        # Defensive reorder to the canonical column order (does not
        # touch the file on disk -- this is an in-memory dataframe).
        dataframe = dataframe[list(EXPECTED_COLUMNS)]

        logger.info("Dataset shape: %s.", dataframe.shape)
        return dataframe

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame) -> None:
        """Ensures every expected feature/label column is present.

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
        """Ensures no NaN values exist in the expected columns.

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
                f"Feature dataset contains NaN value(s) in column(s): {offending}."
            )

    @staticmethod
    def _validate_classes(dataframe: pd.DataFrame) -> None:
        """Ensures exactly the four expected classes are present.

        Args:
            dataframe: The loaded dataframe.

        Raises:
            DatasetValidationError: If the class set does not exactly
                match ``EXPECTED_CLASSES``.
        """
        found_classes = set(dataframe[LABEL_COLUMN].unique())
        expected_classes = set(EXPECTED_CLASSES)

        if found_classes != expected_classes:
            raise DatasetValidationError(
                f"Unexpected class labels found: {sorted(found_classes)}. "
                f"Expected exactly: {sorted(expected_classes)}."
            )


def reproduce_test_split(
    dataframe: pd.DataFrame, config: EvaluationConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Reproduces the EXACT stratified train/test split used by train_models.py.

    Uses the identical call signature
    (``test_size=0.20, stratify=y, random_state=42``) against the
    identical ``X``/``y`` construction, so ``X_test``/``y_test`` here
    are guaranteed to be the same held-out samples the models were
    never trained on.

    Args:
        dataframe: The validated feature dataframe.
        config: Evaluation configuration (random_state, test_size).

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)``. Only
        ``X_test``/``y_test`` are used for evaluation; ``X_train``/
        ``y_train`` are returned for completeness but never fit on.
    """
    X = dataframe[list(FEATURE_COLUMNS)].copy()
    y = dataframe[LABEL_COLUMN].to_numpy()

    logger.info(
        "Reproducing stratified %.0f/%.0f train/test split (random_state=%d).",
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

    logger.info("Reproduced test set: %d samples (from %d total).", len(X_test), len(dataframe))
    return X_train, X_test, y_train, y_test


# ==============================================================================
# ARTIFACT LOADING (read-only; NEVER fits/retrains anything)
# ==============================================================================

class ArtifactLoader:
    """Loads the already-trained model artifacts produced by train_models.py.

    Every model returned by this class is fully fitted -- nothing here
    ever calls ``.fit()``.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initializes the loader.

        Args:
            config: Evaluation configuration describing where trained
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
                "Missing required trained artifact(s): "
                f"{missing}. Run train_models.py first to generate them."
            )
        logger.info("All required trained artifacts were found.")

    def load_models(self) -> Dict[str, Any]:
        """Loads the three trained classifiers plus the label encoder.

        Returns:
            Dictionary with keys ``"Random Forest"``, ``"SVM"``,
            ``"XGBoost"``, and ``"label_encoder"``.

        Raises:
            ArtifactLoadError: If any artifact fails to load.
        """
        paths = self._config.artifact_paths()
        try:
            random_forest = joblib.load(paths["random_forest"])
            svm = joblib.load(paths["svm"])
            xgboost_model = joblib.load(paths["xgboost"])
            label_encoder = joblib.load(paths["label_encoder"])
        except (OSError, ValueError, EOFError) as exc:
            raise ArtifactLoadError(f"Failed to load trained artifact(s): {exc}") from exc

        logger.info("Loaded Random Forest, SVM, XGBoost, and label encoder artifacts.")
        return {
            "Random Forest": random_forest,
            "SVM": svm,
            "XGBoost": xgboost_model,
            "label_encoder": label_encoder,
        }

    def load_training_summary(self) -> Dict[str, Any]:
        """Loads ``training_summary.json`` for best-model information.

        Returns:
            The parsed training summary dictionary.

        Raises:
            ArtifactLoadError: If the summary cannot be loaded/parsed.
        """
        path = self._config.artifact_paths()["training_summary"]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactLoadError(f"Failed to load '{path}': {exc}") from exc

        logger.info("Loaded training summary from '%s'.", path)
        return summary


# ==============================================================================
# PER-MODEL PREDICTION RESULT CONTAINER
# ==============================================================================

@dataclass
class ModelEvaluationResult:
    """Holds every evaluation artifact computed for a single model.

    Attributes:
        name: Model display name (e.g. ``"Random Forest"``).
        y_pred: Predicted labels (human-readable strings) on X_test.
        proba: Class-probability matrix on X_test, with columns
            reordered to match ``CLASS_ORDER``. ``None`` if
            unavailable.
        accuracy: Weighted-average-independent accuracy score.
        precision: Weighted precision.
        recall: Weighted recall.
        f1: Weighted F1 score.
        classification_report: Dict form of
            ``sklearn.metrics.classification_report``.
        confusion: Confusion matrix (rows=true, cols=pred), ordered by
            ``CLASS_ORDER``.
    """

    name: str
    y_pred: np.ndarray
    proba: np.ndarray | None
    accuracy: float
    precision: float
    recall: float
    f1: float
    classification_report: Dict[str, Any]
    confusion: np.ndarray

    def metrics_dict(self) -> Dict[str, float]:
        """Returns the four headline metrics as a JSON-serializable dict."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


# ==============================================================================
# EVALUATION
# ==============================================================================

class ModelEvaluator:
    """Evaluates each trained model on the SAME held-out test set.

    No model is ever fit here -- every method only calls ``.predict``
    / ``.predict_proba`` on already-trained estimators.
    """

    def __init__(self, class_order: Tuple[str, ...] = CLASS_ORDER) -> None:
        """Initializes the evaluator.

        Args:
            class_order: The canonical class ordering used for
                confusion matrices, ROC curves, and probability-column
                alignment.
        """
        self._class_order = class_order

    def evaluate_random_forest(
        self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> ModelEvaluationResult:
        """Evaluates the Random Forest model on raw (unscaled) test features.

        Args:
            model: The fitted ``RandomForestClassifier``.
            X_test: Held-out test feature matrix, raw (unscaled).
            y_test: Held-out true test labels (human-readable strings).

        Returns:
            A populated :class:`ModelEvaluationResult`.

        Raises:
            EvaluationError: If prediction fails.
        """
        try:
            y_pred = model.predict(X_test)
            proba = self._reorder_proba(model.predict_proba(X_test), model.classes_)
        except (ValueError, AttributeError) as exc:
            raise EvaluationError(f"Random Forest evaluation failed: {exc}") from exc

        return self._build_result("Random Forest", y_test, y_pred, proba)

    def evaluate_svm(
        self, pipeline: Any, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> ModelEvaluationResult:
        """Evaluates the SVM pipeline (StandardScaler + SVC) on raw test features.

        The saved artifact is a ``Pipeline``, so raw (unscaled)
        features are passed directly -- the scaler fit during training
        is applied internally and is NEVER refit here.

        Args:
            pipeline: The fitted ``Pipeline(StandardScaler, SVC)``.
            X_test: Held-out test feature matrix, raw (unscaled).
            y_test: Held-out true test labels (human-readable strings).

        Returns:
            A populated :class:`ModelEvaluationResult`.

        Raises:
            EvaluationError: If prediction fails.
        """
        try:
            y_pred = pipeline.predict(X_test)
            proba = self._reorder_proba(pipeline.predict_proba(X_test), pipeline.classes_)
        except (ValueError, AttributeError) as exc:
            raise EvaluationError(f"SVM evaluation failed: {exc}") from exc

        return self._build_result("SVM", y_test, y_pred, proba)

    def evaluate_xgboost(
        self,
        model: Any,
        label_encoder: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> ModelEvaluationResult:
        """Evaluates the XGBoost model, decoding integer predictions back to labels.

        XGBoost predicts integer-encoded classes; both the hard
        predictions and the probability-matrix columns are decoded /
        reordered using the SAME ``label_encoder.joblib`` fit during
        training (never refit here), so that XGBoost is compared on
        identical footing to Random Forest and SVM.

        Args:
            model: The fitted ``XGBClassifier``.
            label_encoder: The ``LabelEncoder`` fit on training labels.
            X_test: Held-out test feature matrix, raw (unscaled).
            y_test: Held-out true test labels (human-readable strings).

        Returns:
            A populated :class:`ModelEvaluationResult`.

        Raises:
            EvaluationError: If prediction or decoding fails.
        """
        try:
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            proba_raw = model.predict_proba(X_test)
            # XGBoost's predict_proba columns correspond to the integer
            # classes 0..K-1, i.e. label_encoder.classes_ order.
            proba = self._reorder_proba(proba_raw, label_encoder.classes_)
        except (ValueError, AttributeError) as exc:
            raise EvaluationError(f"XGBoost evaluation failed: {exc}") from exc

        return self._build_result("XGBoost", y_test, y_pred, proba)

    def _reorder_proba(self, proba: np.ndarray, model_class_order: np.ndarray) -> np.ndarray:
        """Reorders a probability matrix's columns to match ``CLASS_ORDER``.

        Args:
            proba: Raw probability matrix as returned by
                ``predict_proba``, with columns in ``model_class_order``.
            model_class_order: The class order corresponding to
                ``proba``'s columns (e.g. ``model.classes_`` or
                ``label_encoder.classes_``).

        Returns:
            The probability matrix with columns reordered so column
            ``i`` corresponds to ``self._class_order[i]``.
        """
        model_class_order = list(model_class_order)
        column_indices = [model_class_order.index(cls) for cls in self._class_order]
        return proba[:, column_indices]

    def _build_result(
        self,
        name: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray | None,
    ) -> ModelEvaluationResult:
        """Computes all shared metrics/report/confusion-matrix for one model.

        Args:
            name: Model display name.
            y_test: Held-out true test labels.
            y_pred: Predicted test labels.
            proba: Reordered probability matrix (or ``None``).

        Returns:
            A populated :class:`ModelEvaluationResult`.
        """
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        report = classification_report(
            y_test,
            y_pred,
            labels=list(self._class_order),
            target_names=list(self._class_order),
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred, labels=list(self._class_order))

        logger.info(
            "%-14s | Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
            name,
            accuracy,
            precision,
            recall,
            f1,
        )

        return ModelEvaluationResult(
            name=name,
            y_pred=y_pred,
            proba=proba,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            classification_report=report,
            confusion=cm,
        )


# ==============================================================================
# ROC / AUC (multiclass One-vs-Rest)
# ==============================================================================

def compute_roc_data(
    y_test: np.ndarray, results: Dict[str, ModelEvaluationResult], class_order: Tuple[str, ...] = CLASS_ORDER
) -> Dict[str, Dict[str, Any]]:
    """Computes One-vs-Rest ROC curves and AUC for every model, per class.

    This is technically appropriate here because the problem has four
    mutually exclusive classes and every model exposes calibrated
    class probabilities (Random Forest and XGBoost natively; SVM via
    ``probability=True``).

    Args:
        y_test: Held-out true test labels (human-readable strings).
        results: Mapping of model name -> :class:`ModelEvaluationResult`.
        class_order: Canonical class ordering matching each result's
            ``proba`` columns.

    Returns:
        Nested dict: ``{model_name: {class_name: {"fpr": [...], "tpr":
        [...], "auc": float}, "macro_auc": float}}``. A model with no
        probability estimates is omitted.
    """
    y_test_binarized = label_binarize(y_test, classes=list(class_order))

    roc_data: Dict[str, Dict[str, Any]] = {}

    for name, result in results.items():
        if result.proba is None:
            logger.warning("Skipping ROC computation for '%s': no probability estimates.", name)
            continue

        per_class: Dict[str, Any] = {}
        auc_values: List[float] = []

        for idx, class_name in enumerate(class_order):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, idx], result.proba[:, idx])
            class_auc = float(auc(fpr, tpr))
            auc_values.append(class_auc)
            per_class[class_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": class_auc,
            }

        per_class["macro_auc"] = float(np.mean(auc_values))
        roc_data[name] = per_class

    return roc_data


# ==============================================================================
# VISUALIZATION
# ==============================================================================

class EvaluationVisualizer:
    """Generates and saves every required evaluation plot under ``results/``."""

    def __init__(self, results_dir: Path, class_order: Tuple[str, ...] = CLASS_ORDER) -> None:
        """Initializes the visualizer.

        Args:
            results_dir: Directory to save generated plots into
                (created automatically if it does not exist).
            class_order: Canonical class ordering for axes/labels.
        """
        self._results_dir = results_dir
        self._class_order = class_order
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, result: ModelEvaluationResult, filename: str) -> Path:
        """Plots and saves a single model's confusion matrix.

        Rows represent the true class; columns represent the predicted
        class. Numeric counts are annotated inside every cell.

        Args:
            result: The model's evaluation result.
            filename: Output filename (relative to ``results_dir``).

        Returns:
            The path the plot was saved to.
        """
        cm = result.confusion
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(range(len(self._class_order)))
        ax.set_yticks(range(len(self._class_order)))
        ax.set_xticklabels(self._class_order)
        ax.set_yticklabels(self._class_order)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title(f"Confusion Matrix — {result.name}")

        threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        fig.colorbar(im, ax=ax, label="Count")
        fig.tight_layout()

        output_path = self._results_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("Saved confusion matrix plot: '%s'.", output_path)
        return output_path

    def plot_accuracy_comparison(
        self, results: Dict[str, ModelEvaluationResult], filename: str = "model_accuracy_comparison.png"
    ) -> Path:
        """Plots a combined bar chart comparing accuracy across all models.

        Args:
            results: Mapping of model name -> evaluation result.
            filename: Output filename (relative to ``results_dir``).

        Returns:
            The path the plot was saved to.
        """
        names = list(results.keys())
        accuracies = [results[name].accuracy for name in names]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(names, accuracies, color=["#4C72B0", "#DD8452", "#55A868"][: len(names)])

        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.4f}",
                ha="center",
                va="bottom",
            )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison (Held-Out Test Set)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        output_path = self._results_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("Saved accuracy comparison plot: '%s'.", output_path)
        return output_path

    def plot_random_forest_feature_importance(
        self,
        model: Any,
        feature_names: Tuple[str, ...] = FEATURE_COLUMNS,
        filename: str = "random_forest_feature_importance.png",
    ) -> Path:
        """Plots Random Forest feature importances as a horizontal bar chart.

        Args:
            model: The fitted ``RandomForestClassifier``.
            feature_names: Ordered feature names matching
                ``model.feature_importances_``.
            filename: Output filename (relative to ``results_dir``).

        Returns:
            The path the plot was saved to.
        """
        importances = model.feature_importances_
        order = np.argsort(importances)  # ascending, for a readable barh plot

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in order],
            importances[order],
            color="#4C72B0",
        )
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        output_path = self._results_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("Saved feature importance plot: '%s'.", output_path)
        return output_path

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, Any]],
        filename: str = "roc_curves.png",
    ) -> Path:
        """Plots multiclass One-vs-Rest ROC curves for every model.

        Produces one subplot per model, each showing the four
        per-class ROC curves with their AUC values in the legend.

        Args:
            roc_data: Output of :func:`compute_roc_data`.
            filename: Output filename (relative to ``results_dir``).

        Returns:
            The path the plot was saved to.
        """
        model_names = list(roc_data.keys())
        fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5.5), squeeze=False)
        axes = axes[0]

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for ax, name in zip(axes, model_names):
            per_class = roc_data[name]
            for color, class_name in zip(colors, self._class_order):
                class_data = per_class[class_name]
                ax.plot(
                    class_data["fpr"],
                    class_data["tpr"],
                    color=color,
                    linewidth=1.6,
                    label=f"{class_name} (AUC={class_data['auc']:.3f})",
                )
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{name} — One-vs-Rest ROC\n(macro AUC={per_class['macro_auc']:.3f})")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)

        fig.tight_layout()

        output_path = self._results_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info("Saved ROC curves plot: '%s'.", output_path)
        return output_path


# ==============================================================================
# SUMMARY BUILDING & REPORTING
# ==============================================================================

class SummaryBuilder:
    """Builds the JSON-serializable evaluation summary and terminal report."""

    def __init__(self, config: EvaluationConfig, class_order: Tuple[str, ...] = CLASS_ORDER) -> None:
        """Initializes the summary builder.

        Args:
            config: Evaluation configuration.
            class_order: Canonical class ordering.
        """
        self._config = config
        self._class_order = class_order

    def build(
        self,
        dataframe: pd.DataFrame,
        X_test: pd.DataFrame,
        results: Dict[str, ModelEvaluationResult],
        roc_data: Dict[str, Dict[str, Any]],
        training_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assembles the full evaluation summary dictionary.

        Args:
            dataframe: The full loaded feature dataset.
            X_test: The reproduced held-out test feature matrix.
            results: Mapping of model name -> evaluation result.
            roc_data: Output of :func:`compute_roc_data`.
            training_summary: Parsed ``training_summary.json`` content.

        Returns:
            The assembled, JSON-serializable summary dictionary.
        """
        summary: Dict[str, Any] = {
            "dataset": {
                "path": str(self._config.dataset_path),
                "total_samples": int(len(dataframe)),
            },
            "test_samples": int(len(X_test)),
            "feature_names": list(FEATURE_COLUMNS),
            "class_names": list(self._class_order),
            "split": {
                "test_size": self._config.test_size,
                "random_state": self._config.random_state,
                "stratified": True,
            },
            "models": {},
            "best_model": {
                "name": training_summary.get("best_model"),
                "selection_criteria": training_summary.get("best_model_selection_criteria"),
                "training_metrics": training_summary.get("models", {}).get(
                    training_summary.get("best_model", ""), {}
                ),
            },
        }

        for name, result in results.items():
            model_entry: Dict[str, Any] = {
                "metrics": result.metrics_dict(),
                "classification_report": result.classification_report,
                "confusion_matrix": {
                    "labels": list(self._class_order),
                    "matrix": result.confusion.tolist(),
                },
            }
            if name in roc_data:
                model_entry["roc_auc"] = {
                    class_name: roc_data[name][class_name]["auc"] for class_name in self._class_order
                }
                model_entry["roc_auc"]["macro_auc"] = roc_data[name]["macro_auc"]
            summary["models"][name] = model_entry

        return summary

    def save(self, summary: Dict[str, Any], filename: str = "evaluation_summary.json") -> Path:
        """Saves the summary dictionary as pretty-printed JSON.

        Args:
            summary: The summary dictionary to persist.
            filename: Output filename (relative to ``results_dir``).

        Returns:
            The path the summary was saved to.

        Raises:
            EvaluationError: If the summary cannot be written.
        """
        self._config.results_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._config.results_dir / filename
        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=4)
        except OSError as exc:
            raise EvaluationError(f"Failed to save evaluation summary to '{output_path}': {exc}") from exc

        logger.info("Saved evaluation summary: '%s'.", output_path)
        return output_path

    def print_terminal_summary(
        self,
        X_test: pd.DataFrame,
        results: Dict[str, ModelEvaluationResult],
        training_summary: Dict[str, Any],
    ) -> None:
        """Prints a clean, human-readable summary to the terminal.

        Args:
            X_test: The reproduced held-out test feature matrix.
            results: Mapping of model name -> evaluation result.
            training_summary: Parsed ``training_summary.json`` content.
        """
        line = "=" * 60
        print(f"\n{line}")
        print("MODEL EVALUATION")
        print(line)
        print(f"\nTest samples: {len(X_test)}")
        print(f"Classes: {', '.join(self._class_order)}\n")

        for name in MODEL_DISPLAY_NAMES:
            if name not in results:
                continue
            result = results[name]
            print(name)
            print(f"Accuracy : {result.accuracy:.4f}")
            print(f"Precision: {result.precision:.4f}")
            print(f"Recall   : {result.recall:.4f}")
            print(f"F1 Score : {result.f1:.4f}\n")

        print(line)
        print("BEST MODEL")
        print(line)
        best_name = training_summary.get("best_model", "Unknown")
        print(f"{best_name}")
        best_metrics = training_summary.get("models", {}).get(best_name, {})
        if best_metrics:
            print(f"Training-time Accuracy : {best_metrics.get('accuracy', float('nan')):.4f}")
            print(f"Training-time Precision: {best_metrics.get('precision', float('nan')):.4f}")
            print(f"Training-time Recall   : {best_metrics.get('recall', float('nan')):.4f}")
            print(f"Training-time F1 Score : {best_metrics.get('f1', float('nan')):.4f}")
        if best_name in results:
            eval_result = results[best_name]
            print(f"Re-evaluated Accuracy  : {eval_result.accuracy:.4f}")
            print(f"Re-evaluated F1 Score  : {eval_result.f1:.4f}")
        print(line + "\n")


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def run_evaluation_pipeline(config: EvaluationConfig) -> Dict[str, Any]:
    """Executes the full evaluation pipeline end-to-end.

    Steps:
        1. Validate that every required artifact/file exists.
        2. Load and validate ``features.csv``.
        3. Reproduce the exact stratified 80/20 test split.
        4. Load the trained models and label encoder (never refit).
        5. Evaluate every model on ``X_test``/``y_test``.
        6. Compute multiclass One-vs-Rest ROC/AUC data.
        7. Generate and save all required plots.
        8. Build, save, and print the evaluation summary.

    Args:
        config: Evaluation configuration.

    Returns:
        The evaluation summary dictionary that was also written to
        ``results/evaluation_summary.json``.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        DatasetValidationError: If the dataset fails validation.
        ArtifactLoadError: If any trained artifact is missing/invalid.
        EvaluationError: If model evaluation or summary saving fails.
    """
    # ---- 1. Validate artifacts exist ---------------------------------------
    artifact_loader = ArtifactLoader(config)
    artifact_loader.validate_files_exist()

    # ---- 2. Load & validate dataset -----------------------------------------
    dataset_loader = DatasetLoader(config)
    dataframe = dataset_loader.load()

    # ---- 3. Reproduce the exact test split -----------------------------------
    _, X_test, _, y_test = reproduce_test_split(dataframe, config)

    # ---- 4. Load trained models (never refit) --------------------------------
    artifacts = artifact_loader.load_models()
    training_summary = artifact_loader.load_training_summary()
    label_encoder = artifacts["label_encoder"]

    # ---- 5. Evaluate every model on X_test/y_test -----------------------------
    evaluator = ModelEvaluator()
    results: Dict[str, ModelEvaluationResult] = {
        "Random Forest": evaluator.evaluate_random_forest(artifacts["Random Forest"], X_test, y_test),
        "SVM": evaluator.evaluate_svm(artifacts["SVM"], X_test, y_test),
        "XGBoost": evaluator.evaluate_xgboost(artifacts["XGBoost"], label_encoder, X_test, y_test),
    }

    # ---- 6. Multiclass One-vs-Rest ROC/AUC ------------------------------------
    roc_data = compute_roc_data(y_test, results)

    # ---- 7. Generate & save all plots -----------------------------------------
    visualizer = EvaluationVisualizer(config.results_dir)
    visualizer.plot_confusion_matrix(results["Random Forest"], "confusion_matrix_random_forest.png")
    visualizer.plot_confusion_matrix(results["SVM"], "confusion_matrix_svm.png")
    visualizer.plot_confusion_matrix(results["XGBoost"], "confusion_matrix_xgboost.png")
    visualizer.plot_accuracy_comparison(results)
    visualizer.plot_random_forest_feature_importance(artifacts["Random Forest"])
    visualizer.plot_roc_curves(roc_data)

    # ---- 8. Build, save, and print the summary ---------------------------------
    summary_builder = SummaryBuilder(config)
    summary = summary_builder.build(dataframe, X_test, results, roc_data, training_summary)
    summary_builder.save(summary)
    summary_builder.print_terminal_summary(X_test, results, training_summary)

    logger.info("Evaluation pipeline complete. Outputs saved to '%s'.", config.results_dir.resolve())
    return summary


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    """Standalone entry point: runs the full evaluation pipeline.

    Running ``python evaluate_models.py`` reproduces the exact 20%
    held-out test split used by ``train_models.py``, loads the already
    -trained Random Forest, SVM, and XGBoost artifacts, evaluates each
    on that test set, generates all required visualizations under
    ``results/``, and writes ``results/evaluation_summary.json``.
    Never retrains, never refits a scaler/encoder, and never modifies
    ``dataset/features.csv``.
    """
    config = EvaluationConfig()

    try:
        run_evaluation_pipeline(config)
    except FileNotFoundError as exc:
        logger.error("Dataset file error: %s", exc)
        raise
    except DatasetValidationError as exc:
        logger.error("Dataset validation failed: %s", exc)
        raise
    except ArtifactLoadError as exc:
        logger.error("Artifact loading failed: %s", exc)
        raise
    except EvaluationError as exc:
        logger.error("Evaluation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()