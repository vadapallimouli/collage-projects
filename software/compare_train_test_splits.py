#!/usr/bin/env python3
"""
compare_train_test_splits.py
============================

Train/test split comparison for the Analog Modulation Classifier project.

Classes : AM, FM, DSB_SC, SSB
Dataset : dataset/features.csv  (12 numerical features + ``Label``)
Models  : Random Forest, SVM (RBF, with StandardScaler), XGBoost
          (configurations are identical to ``train_models.py``)

For each split ratio (70:30, 75:25, 80:20, 85:15, 90:10) the script:

1. Loads the same dataset.
2. Creates ONE stratified split (``random_state=42``) shared by all models.
3. Trains every model from scratch and measures training time.
4. Evaluates on that split's held-out test set
   (accuracy, weighted precision / recall / F1).
5. Records train/test sample counts and per-model results.

Outputs (all under ``results/split_comparison/``):

* split_comparison.csv
* split_comparison_summary.json
* accuracy_comparison.png
* f1_comparison.png
* training_time_comparison.png
* split_experiment.log

Safeguards
----------
* Never writes to ``trained_models/`` and never saves model files.
* Never modifies the dataset or any existing project script.
* Does not import ``train_models.py`` (so nothing there can be triggered).
* No deep learning, no signal generation, no feature regeneration.
* All results are reported; no "best split" is selected.

Usage (from the project root)::

    python compare_train_test_splits.py
"""

from __future__ import annotations

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

matplotlib.use("Agg")  # headless-safe backend; we only save figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
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
OUTPUT_DIR: Path = PROJECT_ROOT / "results" / "split_comparison"
PROTECTED_DIR: Path = PROJECT_ROOT / "trained_models"

TARGET_COLUMN: str = "Label"
EXPECTED_CLASSES: Tuple[str, ...] = ("AM", "FM", "DSB_SC", "SSB")
EXPECTED_N_FEATURES: int = 12
EXPECTED_N_ROWS: int = 4000
EXPECTED_SAMPLES_PER_CLASS: int = 1000

RANDOM_STATE: int = 42

# (label, test_size)  ->  train fraction = 1 - test_size
SPLIT_CONFIGS: Tuple[Tuple[str, float], ...] = (
    ("70:30", 0.30),
    ("75:25", 0.25),
    ("80:20", 0.20),
    ("85:15", 0.15),
    ("90:10", 0.10),
)

MODEL_NAMES: Tuple[str, ...] = ("Random Forest", "SVM", "XGBoost")

CSV_PATH: Path = OUTPUT_DIR / "split_comparison.csv"
JSON_PATH: Path = OUTPUT_DIR / "split_comparison_summary.json"
ACC_PLOT_PATH: Path = OUTPUT_DIR / "accuracy_comparison.png"
F1_PLOT_PATH: Path = OUTPUT_DIR / "f1_comparison.png"
TIME_PLOT_PATH: Path = OUTPUT_DIR / "training_time_comparison.png"
LOG_PATH: Path = OUTPUT_DIR / "split_experiment.log"

logger = logging.getLogger("split_experiment")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentResult:
    """Result of one (split, model) combination."""

    split: str
    train_ratio: float
    test_ratio: float
    train_samples: int
    test_samples: int
    model: str
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    training_time_sec: float


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    """Configure console + file logging. Creates the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def ensure_output_dir_is_safe() -> None:
    """Abort if the output directory would overlap the protected model directory."""
    out = OUTPUT_DIR.resolve()
    protected = PROTECTED_DIR.resolve()
    if out == protected or protected in out.parents:
        raise RuntimeError(
            f"Refusing to write inside protected directory: {protected}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    logger.info("Loaded dataset: %s  shape=%s", path.relative_to(PROJECT_ROOT), df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> List[str]:
    """
    Validate the dataframe and return the ordered list of feature columns.

    Feature columns are every column except ``Label`` (same convention as the
    training pipeline: all numeric columns are features, ``Label`` is the target).

    Raises
    ------
    ValueError
        If any structural expectation is violated.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    if len(feature_cols) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} feature columns, found "
            f"{len(feature_cols)}: {feature_cols}"
        )

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns detected: {non_numeric}")

    if df[feature_cols].isna().any().any():
        bad = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
        raise ValueError(f"Missing values found in feature columns: {bad}")

    if not np.isfinite(df[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values (inf/-inf) found in feature columns.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Missing values found in '{TARGET_COLUMN}' column.")

    labels = df[TARGET_COLUMN].astype(str).str.strip()
    found = set(labels.unique())
    expected = set(EXPECTED_CLASSES)
    if found != expected:
        raise ValueError(
            f"Label mismatch. Expected {sorted(expected)}, found {sorted(found)}."
        )

    if len(df) != EXPECTED_N_ROWS:
        logger.warning("Expected %d rows but found %d.", EXPECTED_N_ROWS, len(df))

    counts = labels.value_counts()
    for cls in EXPECTED_CLASSES:
        if counts[cls] != EXPECTED_SAMPLES_PER_CLASS:
            logger.warning(
                "Class '%s' has %d samples (expected %d).",
                cls, counts[cls], EXPECTED_SAMPLES_PER_CLASS,
            )

    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    logger.info("Class distribution: %s", counts.to_dict())
    return feature_cols


def prepare_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and string-label target y (whitespace-stripped)."""
    X = df[feature_cols].astype(float)
    y = df[TARGET_COLUMN].astype(str).str.strip()
    return X, y


# --------------------------------------------------------------------------- #
# Models (configurations identical to train_models.py)
# --------------------------------------------------------------------------- #
def build_random_forest() -> RandomForestClassifier:
    """Random Forest exactly as configured in train_models.py."""
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def build_svm() -> Pipeline:
    """StandardScaler + RBF SVM exactly as configured in train_models.py."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )


def build_xgboost() -> Tuple[Any, Dict[str, Any]]:
    """
    Build XGBoost exactly as configured in train_models.py.

    Returns
    -------
    (model, fit_kwargs)
        ``fit_kwargs`` is empty on modern XGBoost (>=1.6), where ``eval_metric``
        is a constructor argument. On older versions that reject it in the
        constructor, ``eval_metric`` is passed to ``fit`` instead.
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
            "XGBoost %s rejected 'eval_metric' in the constructor; "
            "passing it to fit() instead.", xgboost.__version__,
        )
        params.pop("eval_metric")
        return XGBClassifier(**params), {"eval_metric": "mlogloss"}


def get_model_factories() -> Dict[str, Callable[[], Tuple[Any, Dict[str, Any]]]]:
    """Map model name -> factory returning (fresh_model, fit_kwargs)."""
    return {
        "Random Forest": lambda: (build_random_forest(), {}),
        "SVM": lambda: (build_svm(), {}),
        "XGBoost": build_xgboost,
    }


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Accuracy and weighted precision / recall / F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def train_and_evaluate(
    model_name: str,
    factory: Callable[[], Tuple[Any, Dict[str, Any]]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
) -> Tuple[Dict[str, float], float]:
    """
    Train a fresh model, time the fit, and evaluate on the held-out test set.

    XGBoost requires integer class labels, so it is trained on encoded labels
    and its predictions are decoded back to the original string labels. All
    models are therefore scored against the same string ``y_test``.

    Returns
    -------
    (metrics, training_time_seconds)
    """
    model, fit_kwargs = factory()
    is_xgb = model_name == "XGBoost"

    y_fit = label_encoder.transform(y_train) if is_xgb else y_train.to_numpy()

    start = time.perf_counter()
    model.fit(X_train, y_fit, **fit_kwargs)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    if is_xgb:
        y_pred = label_encoder.inverse_transform(np.asarray(y_pred).astype(int).ravel())

    metrics = compute_metrics(y_test.to_numpy(), np.asarray(y_pred))
    return metrics, float(train_time)


def run_experiment(X: pd.DataFrame, y: pd.Series) -> List[ExperimentResult]:
    """Run every split x model combination and return all results."""
    label_encoder = LabelEncoder().fit(sorted(y.unique()))
    factories = get_model_factories()
    results: List[ExperimentResult] = []

    for split_label, test_size in SPLIT_CONFIGS:
        # One split per ratio, shared by all three models.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
        )
        n_train, n_test = len(X_train), len(X_test)
        logger.info("=" * 70)
        logger.info(
            "Split %s  |  train=%d  test=%d  |  test class counts=%s",
            split_label, n_train, n_test, y_test.value_counts().to_dict(),
        )

        for model_name in MODEL_NAMES:
            logger.info("Training %s on split %s ...", model_name, split_label)
            try:
                metrics, train_time = train_and_evaluate(
                    model_name, factories[model_name],
                    X_train, X_test, y_train, y_test, label_encoder,
                )
            except Exception:
                logger.exception("FAILED: %s on split %s", model_name, split_label)
                raise

            result = ExperimentResult(
                split=split_label,
                train_ratio=round(1.0 - test_size, 4),
                test_ratio=round(test_size, 4),
                train_samples=n_train,
                test_samples=n_test,
                model=model_name,
                training_time_sec=round(train_time, 4),
                **{k: round(v, 6) for k, v in metrics.items()},
            )
            results.append(result)
            logger.info(
                "%s | %s | acc=%.4f prec=%.4f rec=%.4f f1=%.4f time=%.2fs",
                split_label, model_name, result.accuracy, result.precision_weighted,
                result.recall_weighted, result.f1_weighted, result.training_time_sec,
            )

    return results


# --------------------------------------------------------------------------- #
# Output: terminal, CSV, JSON
# --------------------------------------------------------------------------- #
def print_terminal_summary(results_df: pd.DataFrame) -> None:
    """Print a clean fixed-width summary table to the terminal."""
    header = (
        f"{'Split':<7}{'Train':>7}{'Test':>7}  {'Model':<14}"
        f"{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Time(s)':>10}"
    )
    line = "-" * len(header)
    print()
    print("TRAIN-TEST SPLIT COMPARISON (all results; no split is ranked)")
    print(line)
    print(header)
    print(line)
    prev_split = None
    for row in results_df.itertuples(index=False):
        if prev_split is not None and row.split != prev_split:
            print(line)
        prev_split = row.split
        print(
            f"{row.split:<7}{row.train_samples:>7}{row.test_samples:>7}  {row.model:<14}"
            f"{row.accuracy:>10.4f}{row.precision_weighted:>11.4f}"
            f"{row.recall_weighted:>9.4f}{row.f1_weighted:>9.4f}"
            f"{row.training_time_sec:>10.2f}"
        )
    print(line)
    print(f"Outputs saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print()


def save_csv(results_df: pd.DataFrame, path: Path = CSV_PATH) -> None:
    """Write one row per split-model combination."""
    results_df.to_csv(path, index=False)
    logger.info("Saved %s", path.relative_to(PROJECT_ROOT))


def build_summary(
    results: List[ExperimentResult],
    results_df: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
    total_runtime_sec: float,
) -> Dict[str, Any]:
    """Assemble the structured JSON summary (descriptive only, no 'best' pick)."""
    per_model: Dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        sub = results_df[results_df["model"] == model_name]
        per_model[model_name] = {
            metric: {
                "mean": round(float(sub[metric].mean()), 6),
                "std": round(float(sub[metric].std(ddof=0)), 6),
                "min": round(float(sub[metric].min()), 6),
                "max": round(float(sub[metric].max()), 6),
            }
            for metric in (
                "accuracy", "precision_weighted", "recall_weighted",
                "f1_weighted", "training_time_sec",
            )
        }

    return {
        "experiment": {
            "name": "Train-test split comparison",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_runtime_sec": round(total_runtime_sec, 2),
            "random_state": RANDOM_STATE,
            "stratified": True,
            "splits": [label for label, _ in SPLIT_CONFIGS],
            "models": list(MODEL_NAMES),
        },
        "dataset": {
            "path": str(DATASET_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "n_rows": int(len(y)),
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "target_column": TARGET_COLUMN,
            "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        },
        "model_configurations": {
            "Random Forest": "RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)",
            "SVM": "Pipeline([StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)])",
            "XGBoost": (
                "XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, "
                "subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, "
                "eval_metric='mlogloss'); trained on LabelEncoder-encoded labels"
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
        "results": [asdict(r) for r in results],
        "per_model_statistics_across_splits": per_model,
        "notes": [
            "Each split uses a different held-out test set, so test sets of "
            "different sizes are not identical and metrics are not directly "
            "paired comparisons; smaller test sets give noisier estimates.",
            "Within a split, all three models share the exact same train/test partition.",
            "All results are reported; no split was selected as 'best' from accuracy alone.",
            "Training times depend on hardware and system load.",
        ],
    }


def save_json(summary: Dict[str, Any], path: Path = JSON_PATH) -> None:
    """Write the JSON summary."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %s", path.relative_to(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_grouped_bars(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    zoom_to_data: bool = False,
    value_fmt: str = "{:.4f}",
) -> None:
    """
    Save a grouped bar chart: x = split, one bar per model.

    If ``zoom_to_data`` is True the y-axis is truncated to the data range
    (clearly annotated on the figure) so small differences are visible.
    """
    splits = [label for label, _ in SPLIT_CONFIGS]
    pivot = results_df.pivot(index="split", columns="model", values=metric).loc[
        splits, list(MODEL_NAMES)
    ]

    x = np.arange(len(splits))
    n_models = len(MODEL_NAMES)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, model_name in enumerate(MODEL_NAMES):
        offsets = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(offsets, pivot[model_name].to_numpy(), width, label=model_name)
        for rect, val in zip(bars, pivot[model_name].to_numpy()):
            ax.annotate(
                value_fmt.format(val),
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(train={int(results_df[results_df['split'] == s]['train_samples'].iloc[0])},"
                        f" test={int(results_df[results_df['split'] == s]['test_samples'].iloc[0])})"
                        for s in splits])
    ax.set_xlabel("Train : Test split")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    lo, hi = float(pivot.to_numpy().min()), float(pivot.to_numpy().max())
    if zoom_to_data:
        span = max(hi - lo, 0.005)
        ax.set_ylim(max(0.0, lo - 3 * span), min(1.0, hi + 3 * span) if hi <= 1.0 else hi + 3 * span)
        ax.text(
            0.99, 0.01, "Note: y-axis is truncated to show small differences",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic",
        )
    else:
        ax.set_ylim(0, hi * 1.15 if hi > 0 else 1)

    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=n_models)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path.relative_to(PROJECT_ROOT))


def generate_plots(results_df: pd.DataFrame) -> None:
    """Create the three comparison figures."""
    plot_grouped_bars(
        results_df, "accuracy",
        "Test Accuracy vs Train-Test Split", "Accuracy",
        ACC_PLOT_PATH, zoom_to_data=True,
    )
    plot_grouped_bars(
        results_df, "f1_weighted",
        "Weighted F1-Score vs Train-Test Split", "Weighted F1-score",
        F1_PLOT_PATH, zoom_to_data=True,
    )
    plot_grouped_bars(
        results_df, "training_time_sec",
        "Training Time vs Train-Test Split", "Training time (seconds)",
        TIME_PLOT_PATH, zoom_to_data=False, value_fmt="{:.2f}",
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

    logger.info("Train-test split comparison started")
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
        feature_cols = validate_dataset(df)
        X, y = prepare_xy(df, feature_cols)

        results = run_experiment(X, y)
        results_df = pd.DataFrame([asdict(r) for r in results])

        save_csv(results_df)
        generate_plots(results_df)

        total_runtime = time.perf_counter() - overall_start
        summary = build_summary(results, results_df, feature_cols, y, total_runtime)
        save_json(summary)

        print_terminal_summary(results_df)
        logger.info("Experiment completed in %.1f s", total_runtime)
        return 0

    except (FileNotFoundError, ValueError) as exc:
        logger.error("Input/validation error: %s", exc)
        return 2
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during experiment")
        return 1


if __name__ == "__main__":
    sys.exit(main())