"""
Analyze Random Forest feature importance for trajectory failure prediction.

Uses the already-trained Random Forest model and the prepared test dataset.
No retraining is performed.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "failure_prediction"
MODEL_DIR = DATA_DIR / "models"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
X_TEST_PATH = DATA_DIR / "X_test.csv"

OUTPUT_CSV = MODEL_DIR / "feature_importance.csv"
OUTPUT_PLOT = MODEL_DIR / "feature_importance.png"


# ============================================================
# FEATURES
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained Random Forest...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test_full = pd.read_csv(X_TEST_PATH)

    missing = [
        feature
        for feature in ML_FEATURES
        if feature not in X_test_full.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features in X_test.csv: {missing}"
        )

    X_test = X_test_full[ML_FEATURES]

    # --------------------------------------------------------
    # Get feature importance
    # --------------------------------------------------------

    importances = model.feature_importances_

    if len(importances) != len(ML_FEATURES):
        raise ValueError(
            "Number of feature importances does not match "
            "number of features."
        )

    # --------------------------------------------------------
    # Create results table
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "feature": ML_FEATURES,
            "importance": importances,
        }
    )

    results = results.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    results["importance_percent"] = (
        results["importance"] * 100
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    for index, row in results.iterrows():

        print(
            f"{index + 1:2d}. "
            f"{row['feature']:<45} "
            f"{row['importance']:.6f} "
            f"({row['importance_percent']:.2f}%)"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_data = results.sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        plot_data["feature"],
        plot_data["importance"],
    )

    plt.xlabel(
        "Random Forest Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Feature Importance for Trajectory Failure Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Feature importance plot saved to:\n"
        f"{OUTPUT_PLOT}"
    )

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 5 FEATURES")
    print("=" * 70)

    for index, row in results.head(5).iterrows():

        print(
            f"{index + 1}. "
            f"{row['feature']} "
            f"({row['importance_percent']:.2f}%)"
        )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    total_importance = results["importance"].sum()

    print("\n")
    print(
        f"Total importance: "
        f"{total_importance:.6f}"
    )

    if abs(total_importance - 1.0) < 1e-6:
        print("Importance check: PASS")
    else:
        print("Importance check: WARNING")

    print("\n")
    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()