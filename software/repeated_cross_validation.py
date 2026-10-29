#!/usr/bin/env python3
"""
repeated_cross_validation.py
============================

Repeated stratified k-fold cross-validation for the Analog Modulation
Classifier project.

Classes : AM, FM, DSB_SC, SSB (1,000 samples each)
Dataset : dataset/features.csv (12 numerical features + ``Label``)
Models  : Random Forest, SVM (StandardScaler + RBF SVC), XGBoost
          (hyperparameters identical to ``train_models.py``)

Design
------
* ``RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)``
  produces 15 train/validation partitions. The partitions are generated
  ONCE and re-used for all three models, so every model is evaluated on
  exactly the same 15 validation sets (paired, directly comparable).
* Each fold trains a brand-new model on 4 folds and validates on the 5th.
* The SVM scaler lives inside the sklearn ``Pipeline`` and is therefore fit
  only on each fold's training partition (no validation leakage).
* XGBoost needs integer labels. A single ``LabelEncoder`` fitted on the
  fixed, sorted class set is used for every fold; predictions are decoded
  back to the original string labels before any metric is computed, so all
  three models are scored on identical (string-label) footing.
* Per-fold metrics: accuracy, weighted precision, weighted recall,
  weighted F1, training time (seconds, ``fit`` only).
* Summary statistics use the sample standard deviation (ddof=1) over the
  15 validation results. No model is ranked on a single fold, and the
  script deliberately does not declare a "winner".

Units
-----
* CSV and JSON files store accuracy / precision / recall / F1 as FRACTIONS
  in [0, 1] (e.g. 0.9825). Training time is in seconds.
* The terminal summary and the PNG charts show these metrics as PERCENTAGES.

Important interpretation note
-----------------------------
Cross-validation scores are an ESTIMATE of generalisation performance
obtained by averaging over many train/validation partitions of the full
dataset. They are NOT directly interchangeable with the existing
held-out test results from ``train_models.py`` / ``evaluate_models.py``
(a single stratified 80:20 split with ``random_state=42``). The two
protocols use different training-set sizes (here 3,200 per fold, the same
size as the 80:20 split, but 15 different partitions instead of one) and
different evaluation samples. The held-out test score is never used as a
cross-validation fold score here.

Safeguards
----------
* Never writes to ``trained_models/`` and never saves model files.
* Never modifies the dataset or any existing project script.
* Does not import any other project module; does not regenerate signals
  or features. No deep learning.
* Writes only under ``results/repeated_cross_validation/``.

Usage (from the project root)::

    python repeated_cross_validation.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; figures are only saved
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError as _xgb_err:  # pragma: no cover - environment dependent
    xgboost = None  # type: ignore[assignment]
    XGBClassifier = None  # type: ignore[assignment,misc]
    _XGB_IMPORT_ERROR: Exception | None = _xgb_err
else:
    _XGB_IMPORT_ERROR = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_PATH: Path = PROJECT_ROOT / "dataset" / "features.csv"
OUTPUT_DIR: Path = PROJECT_ROOT / "results" / "repeated_cross_validation"
PROTECTED_DIR: Path = PROJECT_ROOT / "trained_models"

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
TARGET_COLUMN: str = "Label"
EXPECTED_CLASSES: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")
EXPECTED_N_ROWS: int = 4000
EXPECTED_SAMPLES_PER_CLASS: int = 1000

N_SPLITS: int = 5
N_REPEATS: int = 3
RANDOM_STATE: int = 42

MODEL_NAMES: Tuple[str, ...] = ("Random Forest", "SVM", "XGBoost")
METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
METRIC_LABELS: Dict[str, str] = {
    "accuracy": "Accuracy",
    "precision_weighted": "Weighted precision",
    "recall_weighted": "Weighted recall",
    "f1_weighted": "Weighted F1-score",
}
MODEL_COLORS: Dict[str, str] = {
    "Random Forest": "#4C72B0",
    "SVM": "#DD8452",
    "XGBoost": "#55A868",
}

FOLD_CSV_PATH: Path = OUTPUT_DIR / "cross_validation_fold_results.csv"
SUMMARY_CSV_PATH: Path = OUTPUT_DIR / "cross_validation_summary.csv"
SUMMARY_JSON_PATH: Path = OUTPUT_DIR / "cross_validation_summary.json"
ACC_PLOT_PATH: Path = OUTPUT_DIR / "cross_validation_accuracy.png"
F1_PLOT_PATH: Path = OUTPUT_DIR / "cross_validation_f1.png"
LOG_PATH: Path = OUTPUT_DIR / "cross_validation.log"

logger = logging.getLogger("repeated_cross_validation")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class FoldRecord:
    """Result of one (model, repeat, fold) evaluation. Metrics are fractions."""

    model: str
    repeat: int
    fold: int
    global_fold_index: int
    train_size: int
    validation_size: int
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    training_time_sec: float


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def ensure_output_dir_is_safe() -> None:
    """Refuse to write inside the protected model directory, then create the output dir."""
    out = OUTPUT_DIR.resolve()
    protected = PROTECTED_DIR.resolve()
    if out == protected or protected in out.parents:
        raise RuntimeError(f"Refusing to write inside protected directory: {protected}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Configure console + file logging (ASCII-safe for Windows consoles)."""
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def rel(path: Path) -> str:
    """Path relative to the project root, with forward slashes (for logs/JSON)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Data loading and validation
# --------------------------------------------------------------------------- #
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the feature CSV (read-only)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run this script from the project root so 'dataset/features.csv' exists."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read dataset '{path}': {exc}") from exc
    logger.info("Loaded dataset: %s shape=%s", rel(path), df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate columns, numeric/finite features, labels and class balance.

    Raises
    ------
    ValueError
        If any structural expectation is violated.
    """
    missing = [c for c in (*FEATURE_COLUMNS, TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. Found columns: {list(df.columns)}"
        )

    non_numeric = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature column(s): {non_numeric}")

    features = df[list(FEATURE_COLUMNS)]
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing values in feature column(s): {bad}")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite values (inf/-inf) found in feature columns.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Missing values in '{TARGET_COLUMN}' column.")

    labels = df[TARGET_COLUMN].astype(str).str.strip()
    found, expected = set(labels.unique()), set(EXPECTED_CLASSES)
    if found != expected:
        raise ValueError(
            f"Label mismatch. Expected {sorted(expected)}, found {sorted(found)}."
        )

    if len(df) != EXPECTED_N_ROWS:
        logger.warning("Expected %d rows but found %d.", EXPECTED_N_ROWS, len(df))
    counts = labels.value_counts()
    for cls in EXPECTED_CLASSES:
        if int(counts[cls]) != EXPECTED_SAMPLES_PER_CLASS:
            logger.warning(
                "Class '%s' has %d samples (expected %d).",
                cls, int(counts[cls]), EXPECTED_SAMPLES_PER_CLASS,
            )

    min_count = int(counts.min())
    if min_count < N_SPLITS:
        raise ValueError(
            f"Smallest class has {min_count} samples; need at least {N_SPLITS} "
            f"for {N_SPLITS}-fold stratification."
        )

    logger.info("Feature columns (%d): %s", len(FEATURE_COLUMNS), list(FEATURE_COLUMNS))
    logger.info("Class distribution: %s", {k: int(v) for k, v in counts.items()})
    logger.info("Dataset validation passed (all feature values finite and numeric).")


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return X (float DataFrame, exact feature order) and y (string labels)."""
    X = df[list(FEATURE_COLUMNS)].astype(np.float64).reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(str).str.strip().to_numpy()
    return X, y


# --------------------------------------------------------------------------- #
# Models (identical hyperparameters to train_models.py)
# --------------------------------------------------------------------------- #
def build_random_forest() -> Tuple[Any, Dict[str, Any]]:
    """Fresh Random Forest, exactly as in train_models.py."""
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), {}


def build_svm() -> Tuple[Any, Dict[str, Any]]:
    """Fresh StandardScaler + RBF SVC pipeline, exactly as in train_models.py.

    The scaler is inside the pipeline, so each fold fits it on its own
    training partition only.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )
    return pipeline, {}


def build_xgboost() -> Tuple[Any, Dict[str, Any]]:
    """Fresh XGBoost, exactly as in train_models.py.

    On very old XGBoost versions that reject ``eval_metric`` in the
    constructor, it is passed to ``fit`` instead (same behaviour).
    """
    if XGBClassifier is None:
        raise ImportError(f"xgboost is not installed: {_XGB_IMPORT_ERROR}")
    params: Dict[str, Any] = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    try:
        return XGBClassifier(**params), {}
    except TypeError:
        logger.warning(
            "XGBoost %s rejected 'eval_metric' in the constructor; passing it to fit().",
            getattr(xgboost, "__version__", "?"),
        )
        params.pop("eval_metric")
        return XGBClassifier(**params), {"eval_metric": "mlogloss"}


def get_model_factories() -> Dict[str, Callable[[], Tuple[Any, Dict[str, Any]]]]:
    """Map model name -> factory returning (fresh_model, fit_kwargs)."""
    return {
        "Random Forest": build_random_forest,
        "SVM": build_svm,
        "XGBoost": build_xgboost,
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy and weighted precision / recall / F1 (fractions in [0, 1])."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def generate_folds(X: pd.DataFrame, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate the 15 stratified train/validation index pairs ONCE.

    The returned list is shared by all models. It is also sanity-checked:
    train/validation are disjoint, together cover the data, and within each
    repeat every sample is validated exactly once.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    folds = [(tr.copy(), va.copy()) for tr, va in rskf.split(X, y)]

    expected_total = N_SPLITS * N_REPEATS
    if len(folds) != expected_total:
        raise RuntimeError(f"Expected {expected_total} folds, generated {len(folds)}.")

    n = len(y)
    for i, (tr, va) in enumerate(folds):
        if np.intersect1d(tr, va).size != 0:
            raise RuntimeError(f"Fold {i}: train and validation indices overlap.")
        if len(tr) + len(va) != n:
            raise RuntimeError(f"Fold {i}: train + validation does not cover the dataset.")

    for rep in range(N_REPEATS):
        seen = np.concatenate([folds[rep * N_SPLITS + k][1] for k in range(N_SPLITS)])
        if len(seen) != n or len(np.unique(seen)) != n:
            raise RuntimeError(
                f"Repeat {rep + 1}: validation folds do not partition the dataset exactly once."
            )
    return folds


def folds_fingerprint(folds: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """SHA-256 of all validation indices; proves all models share identical folds."""
    h = hashlib.sha256()
    for _, va in folds:
        h.update(np.ascontiguousarray(va, dtype=np.int64).tobytes())
    return h.hexdigest()


def evaluate_fold(
    model_name: str,
    factory: Callable[[], Tuple[Any, Dict[str, Any]]],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[Dict[str, float], float]:
    """Train a fresh model on one fold, time ``fit``, score on the validation fold."""
    model, fit_kwargs = factory()
    is_xgb = model_name == "XGBoost"
    y_fit = label_encoder.transform(y_train) if is_xgb else y_train

    start = time.perf_counter()
    model.fit(X_train, y_fit, **fit_kwargs)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_val)
    if is_xgb:
        y_pred = label_encoder.inverse_transform(np.asarray(y_pred).astype(int).ravel())

    return compute_metrics(y_val, np.asarray(y_pred)), float(train_time)


def run_cross_validation(X: pd.DataFrame, y: np.ndarray) -> Tuple[List[FoldRecord], str]:
    """Run all models over the shared folds. Returns records and fold fingerprint."""
    label_encoder = LabelEncoder().fit(sorted(EXPECTED_CLASSES))
    logger.info("Label encoding (used for XGBoost): %s", list(label_encoder.classes_))

    folds = generate_folds(X, y)
    fingerprint = folds_fingerprint(folds)
    logger.info(
        "Generated %d shared folds (RepeatedStratifiedKFold n_splits=%d n_repeats=%d "
        "random_state=%d). Fold fingerprint: %s",
        len(folds), N_SPLITS, N_REPEATS, RANDOM_STATE, fingerprint[:16],
    )

    factories = get_model_factories()
    records: List[FoldRecord] = []

    for model_name in MODEL_NAMES:
        logger.info("=" * 70)
        logger.info("Cross-validating %s", model_name)
        for idx, (tr_idx, va_idx) in enumerate(folds):
            repeat = idx // N_SPLITS + 1
            fold = idx % N_SPLITS + 1
            X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
            y_train, y_val = y[tr_idx], y[va_idx]

            try:
                metrics, train_time = evaluate_fold(
                    model_name, factories[model_name],
                    X_train, X_val, y_train, y_val, label_encoder,
                )
            except Exception:
                logger.exception(
                    "FAILED: %s repeat=%d fold=%d", model_name, repeat, fold
                )
                raise

            rec = FoldRecord(
                model=model_name,
                repeat=repeat,
                fold=fold,
                global_fold_index=idx + 1,
                train_size=len(tr_idx),
                validation_size=len(va_idx),
                training_time_sec=round(train_time, 4),
                **{k: round(v, 6) for k, v in metrics.items()},
            )
            records.append(rec)
            logger.info(
                "%s | repeat %d fold %d | train=%d val=%d | acc=%.4f prec=%.4f "
                "rec=%.4f f1=%.4f | %.2fs",
                model_name, repeat, fold, rec.train_size, rec.validation_size,
                rec.accuracy, rec.precision_weighted, rec.recall_weighted,
                rec.f1_weighted, rec.training_time_sec,
            )

    expected_records = len(MODEL_NAMES) * N_SPLITS * N_REPEATS
    if len(records) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, got {len(records)}.")
    return records, fingerprint


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarise(fold_df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean/std (ddof=1) of each metric, mean training time."""
    rows: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        sub = fold_df[fold_df["model"] == model_name]
        row: Dict[str, Any] = {"model": model_name, "n_folds": int(len(sub))}
        for key in METRIC_KEYS:
            row[f"{key}_mean"] = round(float(sub[key].mean()), 6)
            row[f"{key}_std"] = round(float(sub[key].std(ddof=1)), 6)
        row["training_time_sec_mean"] = round(float(sub["training_time_sec"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def print_terminal_summary(summary_df: pd.DataFrame) -> None:
    """Readable summary; metrics shown as percentages. No model is ranked."""
    header = (
        f"{'Model':<15}{'Accuracy %':>16}{'Precision %':>16}"
        f"{'Recall %':>16}{'F1 %':>16}{'Time(s)':>10}"
    )
    line = "-" * len(header)
    n_folds = N_SPLITS * N_REPEATS
    print()
    print(
        f"REPEATED STRATIFIED CROSS-VALIDATION "
        f"({N_SPLITS}-fold x {N_REPEATS} repeats = {n_folds} folds per model)"
    )
    print("Values: mean +/- std over all folds (weighted precision/recall/F1)")
    print(line)
    print(header)
    print(line)
    for r in summary_df.itertuples(index=False):
        def cell(key: str) -> str:
            mean = getattr(r, f"{key}_mean") * 100.0
            std = getattr(r, f"{key}_std") * 100.0
            return f"{mean:7.3f} +/- {std:5.3f}"

        print(
            f"{r.model:<15}{cell('accuracy'):>16}{cell('precision_weighted'):>16}"
            f"{cell('recall_weighted'):>16}{cell('f1_weighted'):>16}"
            f"{r.training_time_sec_mean:>10.2f}"
        )
    print(line)
    print("Note: CV scores estimate generalisation; they are not interchangeable with")
    print("the single held-out 80:20 test results. No model is ranked on a single fold.")
    print(f"Outputs saved to: {rel(OUTPUT_DIR)}")
    print()


# --------------------------------------------------------------------------- #
# File outputs
# --------------------------------------------------------------------------- #
def save_csvs(fold_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Write the fold-level and summary CSVs (metrics as fractions)."""
    fold_df.to_csv(FOLD_CSV_PATH, index=False)
    logger.info("Saved %s", rel(FOLD_CSV_PATH))
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    logger.info("Saved %s", rel(SUMMARY_CSV_PATH))


def save_json(
    summary_df: pd.DataFrame,
    y: np.ndarray,
    fingerprint: str,
    total_runtime_sec: float,
) -> None:
    """Write the structured JSON summary (metrics as fractions)."""
    counts = pd.Series(y).value_counts()
    model_summary: Dict[str, Any] = {}
    for r in summary_df.to_dict(orient="records"):
        name = r["model"]
        model_summary[name] = {
            "n_folds": r["n_folds"],
            **{
                key: {"mean": r[f"{key}_mean"], "std": r[f"{key}_std"]}
                for key in METRIC_KEYS
            },
            "training_time_sec_mean": r["training_time_sec_mean"],
        }

    payload: Dict[str, Any] = {
        "experiment": {
            "name": "Repeated stratified cross-validation",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_runtime_sec": round(total_runtime_sec, 2),
            "cv_splitter": "sklearn.model_selection.RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_STATE,
            "folds_per_model": N_SPLITS * N_REPEATS,
            "total_model_fold_records": len(MODEL_NAMES) * N_SPLITS * N_REPEATS,
            "shared_folds_across_models": True,
            "fold_assignment_sha256": fingerprint,
            "models": list(MODEL_NAMES),
            "metric_units": (
                "accuracy, precision_weighted, recall_weighted and f1_weighted are "
                "fractions in [0, 1] (not percentages); training_time_sec is seconds."
            ),
            "std_definition": "sample standard deviation (ddof=1) across all folds",
        },
        "dataset": {
            "path": rel(DATASET_PATH),
            "n_rows": int(len(y)),
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "class_counts": {k: int(v) for k, v in counts.items()},
        },
        "model_configurations": {
            "Random Forest": "RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)",
            "SVM": (
                "Pipeline([('scaler', StandardScaler()), ('classifier', "
                "SVC(kernel='rbf', probability=True, random_state=42))]); "
                "scaler fit on each fold's training partition only"
            ),
            "XGBoost": (
                "XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, "
                "subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, "
                "eval_metric='mlogloss'); trained on LabelEncoder-encoded labels, "
                "predictions decoded before scoring"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": getattr(xgboost, "__version__", None),
        },
        "model_summary": model_summary,
        "notes": [
            "Cross-validation results estimate generalisation performance and are not "
            "directly interchangeable with the existing held-out 80:20 test results.",
            "The held-out test score is not used as a cross-validation fold score.",
            "All three models were evaluated on identical train/validation partitions.",
            "Models are not ranked on any single fold; use mean and standard deviation "
            "across all folds. Repeats reuse the same samples, so folds are not fully "
            "independent and the standard deviation should be read descriptively.",
            "Training times depend on hardware and system load.",
        ],
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %s", rel(SUMMARY_JSON_PATH))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_metric(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Bar chart of mean metric (in %) per model with std error bars."""
    by_model = summary_df.set_index("model").loc[list(MODEL_NAMES)]
    means = by_model[f"{metric}_mean"].to_numpy() * 100.0
    stds = by_model[f"{metric}_std"].to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=(9, 6.5))
    x = np.arange(len(MODEL_NAMES))
    for i, name in enumerate(MODEL_NAMES):
        ax.bar(
            x[i], means[i], width=0.55, yerr=stds[i], capsize=8,
            color=MODEL_COLORS[name], edgecolor="black", linewidth=0.8,
            error_kw={"elinewidth": 1.6, "capthick": 1.6, "ecolor": "black"},
            label=name,
        )

    # Zoom the y-axis so small differences are visible (annotated on the figure).
    lo = float((means - stds).min())
    hi = float((means + stds).max())
    pad = max(hi - lo, 0.5)
    y_low = max(0.0, lo - 1.5 * pad)
    y_high = hi + 1.2 * pad
    ax.set_ylim(y_low, y_high)

    label_offset = 0.03 * (y_high - y_low)
    for i in range(len(MODEL_NAMES)):
        ax.text(
            x[i], means[i] + stds[i] + label_offset,
            f"{means[i]:.2f}%\n+/- {stds[i]:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{ylabel} (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Model", loc="lower right")
    ax.text(
        0.99, 0.01,
        f"Error bars: +/- 1 std over {N_SPLITS * N_REPEATS} folds "
        f"({N_SPLITS}-fold x {N_REPEATS} repeats). Y-axis truncated.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic",
    )
    ax.text(
        0.01, 0.01, "", transform=ax.transAxes
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", rel(out_path))


def generate_plots(summary_df: pd.DataFrame) -> None:
    """Create the accuracy and F1 comparison charts."""
    plot_metric(
        summary_df, "accuracy",
        "Repeated Cross-Validation: Mean Accuracy by Model",
        "Mean accuracy", ACC_PLOT_PATH,
    )
    plot_metric(
        summary_df, "f1_weighted",
        "Repeated Cross-Validation: Mean Weighted F1-Score by Model",
        "Mean weighted F1-score", F1_PLOT_PATH,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    """Entry point. Returns a process exit code."""
    overall_start = time.perf_counter()

    try:
        ensure_output_dir_is_safe()
        setup_logging()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not initialise output directory/logging: {exc}", file=sys.stderr)
        return 1

    logger.info("Repeated cross-validation started")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info(
        "Versions: python=%s sklearn=%s xgboost=%s pandas=%s numpy=%s",
        platform.python_version(), sklearn.__version__,
        getattr(xgboost, "__version__", "NOT INSTALLED"), pd.__version__, np.__version__,
    )

    if XGBClassifier is None:
        logger.error("xgboost is required but could not be imported: %s", _XGB_IMPORT_ERROR)
        return 1

    try:
        df = load_dataset()
        validate_dataset(df)
        X, y = prepare_xy(df)

        records, fingerprint = run_cross_validation(X, y)
        fold_df = pd.DataFrame([asdict(r) for r in records])
        summary_df = summarise(fold_df)

        save_csvs(fold_df, summary_df)
        generate_plots(summary_df)

        total_runtime = time.perf_counter() - overall_start
        save_json(summary_df, y, fingerprint, total_runtime)

        print_terminal_summary(summary_df)
        logger.info("Experiment completed in %.1f s", total_runtime)
        return 0

    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input/validation error: %s", exc)
        return 2
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during cross-validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())