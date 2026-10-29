"""
Train and evaluate the localized 17-feature failure prediction models.

Uses:
    X_train_localized.csv
    X_test_localized.csv
    y_train_localized.csv
    y_test_localized.csv

Saves all outputs with "_localized" so the original experiment is untouched.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    ) from exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "failure_prediction"
MODEL_DIR = DATA_DIR / "models"

X_TRAIN_PATH = DATA_DIR / "X_train_localized.csv"
X_TEST_PATH = DATA_DIR / "X_test_localized.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train_localized.csv"
Y_TEST_PATH = DATA_DIR / "y_test_localized.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print("=" * 70)
print("TRAINING LOCALIZED FAILURE PREDICTION MODELS")
print("=" * 70)

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0].astype(int)
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0].astype(int)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

if X_train.shape[1] != 17:
    raise RuntimeError(
        f"Expected 17 localized features, found {X_train.shape[1]}"
    )

if list(X_train.columns) != list(X_test.columns):
    raise RuntimeError("Train/test feature columns do not match.")

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise RuntimeError("Missing values found in feature data.")

if not np.isfinite(X_train.to_numpy(dtype=float)).all():
    raise RuntimeError("Non-finite values found in X_train.")

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    raise RuntimeError("Non-finite values found in X_test.")

FEATURE_NAMES = list(X_train.columns)

with open(MODEL_DIR / "feature_names_localized.json", "w", encoding="utf-8") as f:
    json.dump(FEATURE_NAMES, f, indent=2)

print(f"\nNumber of localized features: {len(FEATURE_NAMES)}")

print(
    f"Train labels: {(y_train == 0).sum()} success / "
    f"{(y_train == 1).sum()} failure"
)
print(
    f"Test labels : {(y_test == 0).sum()} success / "
    f"{(y_test == 1).sum()} failure"
)


# ---------------------------------------------------------------------
# Models
# Same hyperparameters as the original experiment
# ---------------------------------------------------------------------

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    ),

    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}


# ---------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------

results = []

all_predictions = pd.DataFrame({
    "y_true": y_test.to_numpy()
})

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {name.upper()}")
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Failure precision: {precision:.4f}")
    print(f"Failure recall   : {recall:.4f}")
    print(f"Failure F1       : {f1:.4f}")
    print(f"ROC-AUC          : {roc_auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Success", "Failure"],
            digits=4,
            zero_division=0,
        )
    )

    # Save model separately
    model_path = MODEL_DIR / f"{name}_localized.joblib"
    joblib.dump(model, model_path)

    # Save confusion matrix separately
    cm_path = DATA_DIR / f"confusion_matrix_{name}_localized.csv"

    pd.DataFrame(
        cm,
        index=["actual_success", "actual_failure"],
        columns=["predicted_success", "predicted_failure"],
    ).to_csv(cm_path)

    # Store predictions
    all_predictions[f"{name}_prediction"] = predictions
    all_predictions[f"{name}_failure_probability"] = probabilities

    results.append({
        "model": name,
        "accuracy": accuracy,
        "failure_precision": precision,
        "failure_recall": recall,
        "failure_f1": f1,
        "roc_auc": roc_auc,
    })

    print(f"\nSaved model: {model_path}")


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_path = DATA_DIR / "model_results_localized.csv"
results_df.to_csv(results_path, index=False)

predictions_path = DATA_DIR / "test_predictions_localized.csv"
all_predictions.to_csv(predictions_path, index=False)


# ---------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL LOCALIZED MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved:")
print(f"  {results_path}")
print(f"  {predictions_path}")
print(
    f"  {MODEL_DIR / 'feature_names_localized.json'}"
)

print("\nLocalized model training PASSED.")