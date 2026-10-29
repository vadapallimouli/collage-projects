"""
Train and evaluate ML models for trajectory failure prediction.

Models:
1. Random Forest
2. SVM
3. XGBoost (if installed)

The models are evaluated on planning problems that were NOT seen during
training. This prevents trajectories from the same planning problem
appearing in both train and test sets.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "failure_prediction"
MODEL_DIR = DATA_DIR / "models"

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"


# ============================================================
# FEATURES USED BY THE ML MODELS
# ============================================================

ML_FEATURES = [
    "interior_mean_clearance",
    "interior_clearance_std",
    "interior_maximum_clearance",
    "interior_low_clearance_count",
    "interior_low_clearance_fraction",
    "joint_space_length",
    "max_joint_step",
    "max_joint_velocity",
    "mean_joint_velocity",
    "max_joint_acceleration",
    "mean_joint_acceleration",
    "minimum_joint_limit_distance",
    "mean_joint_limit_distance",
    "minimum_normalized_joint_limit_distance",
    "mean_normalized_joint_limit_distance",
]


# ============================================================
# HELPERS
# ============================================================

def load_dataset():
    """Load prepared train/test datasets and verify them."""

    print("=" * 70)
    print("LOADING PREPARED DATASET")
    print("=" * 70)

    X_train_full = pd.read_csv(X_TRAIN_PATH)
    X_test_full = pd.read_csv(X_TEST_PATH)

    y_train_df = pd.read_csv(Y_TRAIN_PATH)
    y_test_df = pd.read_csv(Y_TEST_PATH)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_train = [
        feature for feature in ML_FEATURES
        if feature not in X_train_full.columns
    ]

    missing_test = [
        feature for feature in ML_FEATURES
        if feature not in X_test_full.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing features in X_train.csv: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing features in X_test.csv: {missing_test}"
        )

    # --------------------------------------------------------
    # Extract ONLY ML features
    # --------------------------------------------------------

    X_train = X_train_full[ML_FEATURES].copy()
    X_test = X_test_full[ML_FEATURES].copy()

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    y_train = y_train_df["failure"].astype(int).to_numpy()
    y_test = y_test_df["failure"].astype(int).to_numpy()

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Training size mismatch: X={len(X_train)}, "
            f"y={len(y_train)}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"Test size mismatch: X={len(X_test)}, "
            f"y={len(y_test)}"
        )

    # --------------------------------------------------------
    # Check for missing/non-finite values
    # --------------------------------------------------------

    if X_train.isnull().any().any():
        raise ValueError("X_train contains missing values.")

    if X_test.isnull().any().any():
        raise ValueError("X_test contains missing values.")

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("X_train contains non-finite values.")

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("X_test contains non-finite values.")

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Number of features: {len(ML_FEATURES)}")

    print("\nTraining labels:")
    print(
        f"  Success (0): {(y_train == 0).sum()}"
    )
    print(
        f"  Failure (1): {(y_train == 1).sum()}"
    )

    print("\nTest labels:")
    print(
        f"  Success (0): {(y_test == 0).sum()}"
    )
    print(
        f"  Failure (1): {(y_test == 1).sum()}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_full,
        X_test_full,
    )


def calculate_metrics(model, X_test, y_test):
    """Calculate classification metrics."""

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)

    else:
        probabilities = predictions

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_failure": precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall_failure": recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "f1_failure": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics, predictions, probabilities


def save_confusion_matrix(
    model_name,
    y_test,
    predictions,
):
    """Save confusion matrix as CSV."""

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Success", "Actual_Failure"],
        columns=["Predicted_Success", "Predicted_Failure"],
    )

    path = MODEL_DIR / f"confusion_matrix_{model_name}.csv"

    cm_df.to_csv(path)

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_full,
        X_test_full,
    ) = load_dataset()

    print("\n")
    print("=" * 70)
    print("FEATURES USED FOR TRAINING")
    print("=" * 70)

    for i, feature in enumerate(ML_FEATURES, start=1):
        print(f"{i:2d}. {feature}")

    # ========================================================
    # DEFINE MODELS
    # ========================================================

    models = {}

    # --------------------------------------------------------
    # 1. Random Forest
    # --------------------------------------------------------

    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    )

    # --------------------------------------------------------
    # 2. SVM
    # --------------------------------------------------------

    models["svm"] = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # 3. XGBoost
    # --------------------------------------------------------

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        print("\nXGBoost detected.")

    except ImportError:

        print("\nXGBoost is not installed.")
        print(
            "XGBoost model will be skipped."
        )
        print(
            "Install later with:"
        )
        print(
            "pip install xgboost"
        )

    # ========================================================
    # TRAIN + EVALUATE
    # ========================================================

    all_results = []
    all_predictions = X_test_full.copy()

    print("\n")
    print("=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    for model_name, model in models.items():

        print("\n")
        print("-" * 70)
        print(f"MODEL: {model_name.upper()}")
        print("-" * 70)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print("Training...")

        model.fit(
            X_train,
            y_train,
        )

        print("Training complete.")

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics, predictions, probabilities = calculate_metrics(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print("\nTest-set results:")

        print(
            f"Accuracy           : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision (failure): "
            f"{metrics['precision_failure']:.4f}"
        )

        print(
            f"Recall (failure)   : "
            f"{metrics['recall_failure']:.4f}"
        )

        print(
            f"F1-score (failure) : "
            f"{metrics['f1_failure']:.4f}"
        )

        print(
            f"ROC-AUC            : "
            f"{metrics['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        print("\nClassification report:")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "Success",
                    "Failure",
                ],
                zero_division=0,
            )
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        print("Confusion matrix:")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        print(cm)

        cm_path = save_confusion_matrix(
            model_name,
            y_test,
            predictions,
        )

        print(
            f"Confusion matrix saved: {cm_path}"
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = MODEL_DIR / f"{model_name}.joblib"

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"Model saved: {model_path}"
        )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        all_predictions[
            f"{model_name}_prediction"
        ] = predictions

        all_predictions[
            f"{model_name}_failure_probability"
        ] = probabilities

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        result = {
            "model": model_name,
            **metrics,
        }

        all_results.append(result)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        MODEL_DIR / "model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    predictions_path = (
        MODEL_DIR / "test_predictions.csv"
    )

    all_predictions.to_csv(
        predictions_path,
        index=False,
    )

    # ========================================================
    # SAVE FEATURE LIST
    # ========================================================

    feature_path = (
        MODEL_DIR / "feature_names.json"
    )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            ML_FEATURES,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL MODEL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved files:")

    print(
        f"  Models       : {MODEL_DIR}"
    )

    print(
        f"  Results      : {results_path}"
    )

    print(
        f"  Predictions  : {predictions_path}"
    )

    print(
        f"  Features     : {feature_path}"
    )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()