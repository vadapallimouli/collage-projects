import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

RECORDS_PATH = "data/failure_prediction/trajectory_records.pkl"
MODEL_PATH = "data/failure_prediction/models/random_forest.joblib"
FEATURE_NAMES_PATH = "data/failure_prediction/models/feature_names.json"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("FAILURE PREDICTOR - DATASET VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    for path in [RECORDS_PATH, MODEL_PATH, FEATURE_NAMES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # --------------------------------------------------------
    # 2. Load trajectory records
    # --------------------------------------------------------

    print("\nLoading trajectory records...")

    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)

    print(f"Records loaded: {len(records)}")

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------

    print("\nLoading Random Forest model...")

    model = joblib.load(MODEL_PATH)

    print(f"Model type: {type(model).__name__}")

    # --------------------------------------------------------
    # 4. Load feature names
    # --------------------------------------------------------

    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    print(f"Number of ML features: {len(feature_names)}")

    print("\nML features:")
    for i, feature in enumerate(feature_names, start=1):
        print(f"{i:2d}. {feature}")

    # --------------------------------------------------------
    # 5. Verify every record contains required features
    # --------------------------------------------------------

    print("\nChecking feature availability...")

    missing_features = []

    for i, record in enumerate(records):

        missing = [
            feature
            for feature in feature_names
            if feature not in record
        ]

        if missing:
            missing_features.append((i, missing))

    if missing_features:
        print("\nERROR: Missing features found!")

        for index, missing in missing_features[:10]:
            print(f"Record {index}: {missing}")

        raise ValueError("Some records are missing ML features.")

    print("Feature availability: PASS")

    # --------------------------------------------------------
    # 6. Build feature matrix
    # --------------------------------------------------------

    print("\nBuilding feature matrix...")

    X = pd.DataFrame(
        [
            {
                feature: record[feature]
                for feature in feature_names
            }
            for record in records
        ],
        columns=feature_names,
    )

    # Ground-truth failure label
    y = np.array(
        [record["failure"] for record in records],
        dtype=int,
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # --------------------------------------------------------
    # 7. Check values
    # --------------------------------------------------------

    print("\nChecking feature values...")

    if X.isnull().any().any():
        raise ValueError("NaN values found in feature matrix.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Infinite values found in feature matrix.")

    print("Feature values: PASS")

    # --------------------------------------------------------
    # 8. Verify labels
    # --------------------------------------------------------

    print("\nGround-truth distribution:")

    success_count = int(np.sum(y == 0))
    failure_count = int(np.sum(y == 1))

    print(f"Successful trajectories : {success_count}")
    print(f"Failed trajectories     : {failure_count}")
    print(f"Total                   : {len(y)}")

    # --------------------------------------------------------
    # 9. Predict
    # --------------------------------------------------------

    print("\nRunning Random Forest predictions...")

    # Suppress only unrelated sklearn feature-name warnings.
    # We are actually passing a DataFrame with feature names.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_probability = model.predict_proba(X)[:, 1]
        else:
            y_probability = None

    print("Prediction complete.")

    # --------------------------------------------------------
    # 10. Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(
        y,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(y, y_pred)

    if y_probability is not None:
        roc_auc = roc_auc_score(y, y_probability)
    else:
        roc_auc = None

    # --------------------------------------------------------
    # 11. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PREDICTION RESULTS")
    print("=" * 75)

    print(f"\nAccuracy          : {accuracy:.4f}")
    print(f"Failure Precision : {precision:.4f}")
    print(f"Failure Recall    : {recall:.4f}")
    print(f"Failure F1 Score  : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC           : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  Success  Failure")
    print(
        f"Actual Success    {cm[0,0]:7d}  {cm[0,1]:7d}"
    )
    print(
        f"Actual Failure    {cm[1,0]:7d}  {cm[1,1]:7d}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=["Success", "Failure"],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # 12. Identify incorrect predictions
    # --------------------------------------------------------

    incorrect_indices = np.where(y != y_pred)[0]

    print("=" * 75)
    print("INCORRECT PREDICTIONS")
    print("=" * 75)

    print(
        f"\nTotal incorrect predictions: "
        f"{len(incorrect_indices)}"
    )

    false_positives = []
    false_negatives = []

    for index in incorrect_indices:

        record = records[index]

        actual = int(y[index])
        predicted = int(y_pred[index])

        probability = (
            float(y_probability[index])
            if y_probability is not None
            else None
        )

        information = {
            "record_index": index,
            "environment": record.get("environment_type"),
            "subset": record.get("subset"),
            "problem_index": record.get("problem_index"),
            "source": record.get("source"),
            "trajectory_index": record.get("trajectory_index"),
            "actual_failure": actual,
            "predicted_failure": predicted,
            "failure_probability": probability,
            "minimum_clearance": record.get(
                "minimum_clearance"
            ),
            "interior_mean_clearance": record.get(
                "interior_mean_clearance"
            ),
            "joint_space_length": record.get(
                "joint_space_length"
            ),
        }

        # False positive:
        # Actual success, predicted failure
        if actual == 0 and predicted == 1:
            false_positives.append(information)

        # False negative:
        # Actual failure, predicted success
        elif actual == 1 and predicted == 0:
            false_negatives.append(information)

    print(
        f"\nFalse positives "
        f"(actual success → predicted failure): "
        f"{len(false_positives)}"
    )

    print(
        f"False negatives "
        f"(actual failure → predicted success): "
        f"{len(false_negatives)}"
    )

    # --------------------------------------------------------
    # 13. Print misclassification examples
    # --------------------------------------------------------

    print("\n" + "-" * 75)
    print("FALSE POSITIVE EXAMPLES")
    print("-" * 75)

    for item in false_positives[:10]:

        print(
            f"\nRecord {item['record_index']}"
        )

        print(
            f"Environment       : {item['environment']}"
        )

        print(
            f"Problem           : {item['problem_index']}"
        )

        print(
            f"Source            : {item['source']}"
        )

        print(
            f"Failure probability: "
            f"{item['failure_probability']:.4f}"
        )

        print(
            f"Interior mean clearance: "
            f"{item['interior_mean_clearance']:.6f}"
        )

        print(
            f"Joint-space length: "
            f"{item['joint_space_length']:.6f}"
        )

    print("\n" + "-" * 75)
    print("FALSE NEGATIVE EXAMPLES")
    print("-" * 75)

    for item in false_negatives[:10]:

        print(
            f"\nRecord {item['record_index']}"
        )

        print(
            f"Environment       : {item['environment']}"
        )

        print(
            f"Problem           : {item['problem_index']}"
        )

        print(
            f"Source            : {item['source']}"
        )

        print(
            f"Failure probability: "
            f"{item['failure_probability']:.4f}"
        )

        print(
            f"Minimum clearance: "
            f"{item['minimum_clearance']:.6f}"
        )

        print(
            f"Interior mean clearance: "
            f"{item['interior_mean_clearance']:.6f}"
        )

        print(
            f"Joint-space length: "
            f"{item['joint_space_length']:.6f}"
        )

    # --------------------------------------------------------
    # 14. Breakdown by trajectory source
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PER-SOURCE PERFORMANCE")
    print("=" * 75)

    source_values = sorted(
        set(record["source"] for record in records)
    )

    for source in source_values:

        indices = [
            i
            for i, record in enumerate(records)
            if record["source"] == source
        ]

        y_source = y[indices]
        pred_source = y_pred[indices]

        source_accuracy = accuracy_score(
            y_source,
            pred_source,
        )

        source_failures = int(np.sum(y_source == 1))

        print(
            f"\n{source}"
        )

        print(
            f"  Samples       : {len(indices)}"
        )

        print(
            f"  Actual failures: {source_failures}"
        )

        print(
            f"  Accuracy      : {source_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # 15. Breakdown by environment
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("PER-ENVIRONMENT PERFORMANCE")
    print("=" * 75)

    environments = sorted(
        set(record["environment_type"] for record in records)
    )

    for environment in environments:

        indices = [
            i
            for i, record in enumerate(records)
            if record["environment_type"] == environment
        ]

        y_env = y[indices]
        pred_env = y_pred[indices]

        env_accuracy = accuracy_score(
            y_env,
            pred_env,
        )

        env_failures = int(np.sum(y_env == 1))

        print(
            f"\n{environment}"
        )

        print(
            f"  Samples       : {len(indices)}"
        )

        print(
            f"  Actual failures: {env_failures}"
        )

        print(
            f"  Accuracy      : {env_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # 16. Final
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DATASET VALIDATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()