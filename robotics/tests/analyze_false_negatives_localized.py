from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "failure_prediction"
MODEL_DIR = DATA_DIR / "models"

X_TEST_PATH = DATA_DIR / "X_test_localized.csv"
Y_TEST_PATH = DATA_DIR / "y_test_localized.csv"
META_PATH = DATA_DIR / "split_metadata_localized.csv"

RF_MODEL_PATH = MODEL_DIR / "random_forest_localized.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_localized.joblib"

OUTPUT_PATH = DATA_DIR / "false_negatives_localized.csv"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("LOCALIZED FALSE-NEGATIVE ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------

    X_test = pd.read_csv(X_TEST_PATH)

    y_test = pd.read_csv(
        Y_TEST_PATH
    ).iloc[:, 0].astype(int)

    # The metadata file contains BOTH train and test rows.
    # We only need the test rows here.
    metadata = pd.read_csv(META_PATH)

    metadata = metadata[
        metadata["split"] == "test"
    ].reset_index(drop=True)

    rf = joblib.load(RF_MODEL_PATH)
    xgb = joblib.load(XGB_MODEL_PATH)

    print(f"\nTest trajectories: {len(X_test)}")
    print(f"Number of features: {X_test.shape[1]}")
    print(f"Test metadata rows: {len(metadata)}")

    # -------------------------------------------------------------
    # Validate alignment
    # -------------------------------------------------------------

    if len(X_test) != len(y_test):
        raise RuntimeError(
            "X_test and y_test have different numbers of rows."
        )

    if len(metadata) != len(X_test):
        raise RuntimeError(
            "Filtered test metadata and test set have "
            "different numbers of rows."
        )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    rf_pred = rf.predict(X_test)

    rf_prob = rf.predict_proba(
        X_test
    )[:, 1]

    xgb_pred = xgb.predict(X_test)

    xgb_prob = xgb.predict_proba(
        X_test
    )[:, 1]

    # -------------------------------------------------------------
    # False negatives
    #
    # Ground truth = failure (1)
    # Prediction    = success (0)
    # -------------------------------------------------------------

    rf_fn_mask = (
        (y_test.to_numpy() == 1)
        & (rf_pred == 0)
    )

    xgb_fn_mask = (
        (y_test.to_numpy() == 1)
        & (xgb_pred == 0)
    )

    rf_fn_indices = np.where(
        rf_fn_mask
    )[0]

    xgb_fn_indices = np.where(
        xgb_fn_mask
    )[0]

    print("\n" + "=" * 70)
    print("FALSE-NEGATIVE COUNTS")
    print("=" * 70)

    print(
        f"Random Forest false negatives : "
        f"{len(rf_fn_indices)}"
    )

    print(
        f"XGBoost false negatives       : "
        f"{len(xgb_fn_indices)}"
    )

    # -------------------------------------------------------------
    # Analyze one model
    # -------------------------------------------------------------

    def analyze_model(
        model_name,
        fn_indices,
        probabilities,
    ):

        print("\n" + "=" * 70)
        print(f"{model_name} FALSE NEGATIVES")
        print("=" * 70)

        if len(fn_indices) == 0:
            print("\nNo false negatives.")
            return pd.DataFrame()

        rows = []

        for idx in fn_indices:

            meta_row = metadata.iloc[idx]
            feature_row = X_test.iloc[idx]

            rows.append({

                "test_row":
                    int(idx),

                "environment_type":
                    meta_row["environment_type"],

                "problem_index":
                    int(meta_row["problem_index"]),

                "group":
                    meta_row["group"],

                "source":
                    meta_row["source"],

                "trajectory_index":
                    int(meta_row["trajectory_index"]),

                "failure_probability":
                    float(probabilities[idx]),

                "interior_mean_clearance":
                    float(
                        feature_row[
                            "interior_mean_clearance"
                        ]
                    ),

                "interior_clearance_std":
                    float(
                        feature_row[
                            "interior_clearance_std"
                        ]
                    ),

                "interior_maximum_clearance":
                    float(
                        feature_row[
                            "interior_maximum_clearance"
                        ]
                    ),

                "interior_low_clearance_count":
                    float(
                        feature_row[
                            "interior_low_clearance_count"
                        ]
                    ),

                "interior_low_clearance_fraction":
                    float(
                        feature_row[
                            "interior_low_clearance_fraction"
                        ]
                    ),

                "number_of_low_clearance_runs":
                    float(
                        feature_row[
                            "number_of_low_clearance_runs"
                        ]
                    ),

                "longest_low_clearance_run":
                    float(
                        feature_row[
                            "longest_low_clearance_run"
                        ]
                    ),

                "joint_space_length":
                    float(
                        feature_row[
                            "joint_space_length"
                        ]
                    ),

                "max_joint_step":
                    float(
                        feature_row[
                            "max_joint_step"
                        ]
                    ),

                "max_joint_velocity":
                    float(
                        feature_row[
                            "max_joint_velocity"
                        ]
                    ),

                "mean_joint_velocity":
                    float(
                        feature_row[
                            "mean_joint_velocity"
                        ]
                    ),

                "max_joint_acceleration":
                    float(
                        feature_row[
                            "max_joint_acceleration"
                        ]
                    ),

                "mean_joint_acceleration":
                    float(
                        feature_row[
                            "mean_joint_acceleration"
                        ]
                    ),
            })

        result = pd.DataFrame(rows)

        print()

        display_columns = [
            "test_row",
            "environment_type",
            "problem_index",
            "source",
            "failure_probability",
            "interior_mean_clearance",
            "interior_low_clearance_count",
            "interior_low_clearance_fraction",
            "number_of_low_clearance_runs",
            "longest_low_clearance_run",
            "joint_space_length",
        ]

        print(
            result[
                display_columns
            ].to_string(
                index=False,
                formatters={
                    "failure_probability":
                        lambda x: f"{x:.4f}",

                    "interior_mean_clearance":
                        lambda x: f"{x:.6f}",

                    "interior_low_clearance_fraction":
                        lambda x: f"{x:.6f}",
                }
            )
        )

        return result

    # -------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------

    rf_results = analyze_model(
        "RANDOM FOREST",
        rf_fn_indices,
        rf_prob,
    )

    # -------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------

    xgb_results = analyze_model(
        "XGBOOST",
        xgb_fn_indices,
        xgb_prob,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    actual_failures = int(
        (y_test == 1).sum()
    )

    print("\n" + "=" * 70)
    print("FALSE-NEGATIVE SUMMARY")
    print("=" * 70)

    print(
        f"\nActual failures in test set: "
        f"{actual_failures}"
    )

    print(
        f"Random Forest: "
        f"{len(rf_fn_indices)} / "
        f"{actual_failures} failures missed"
    )

    print(
        f"XGBoost:      "
        f"{len(xgb_fn_indices)} / "
        f"{actual_failures} failures missed"
    )

    # -------------------------------------------------------------
    # Localized feature statistics
    # -------------------------------------------------------------

    def print_feature_stats(
        name,
        result,
    ):

        if result.empty:
            return

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        features = [
            "number_of_low_clearance_runs",
            "longest_low_clearance_run",
            "interior_low_clearance_fraction",
            "interior_low_clearance_count",
            "interior_mean_clearance",
        ]

        for feature in features:

            values = result[feature]

            print(
                f"{feature:40s} "
                f"mean={values.mean():.4f} | "
                f"median={values.median():.4f} | "
                f"max={values.max():.4f}"
            )

    print_feature_stats(
        "RANDOM FOREST FALSE NEGATIVES",
        rf_results,
    )

    print_feature_stats(
        "XGBOOST FALSE NEGATIVES",
        xgb_results,
    )

    # -------------------------------------------------------------
    # Environment breakdown
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES BY ENVIRONMENT")
    print("=" * 70)

    for model_name, result in [
        ("Random Forest", rf_results),
        ("XGBoost", xgb_results),
    ]:

        print(f"\n{model_name}:")

        if result.empty:
            print("  None")
            continue

        counts = (
            result[
                "environment_type"
            ]
            .value_counts()
            .sort_index()
        )

        for environment, count in counts.items():

            print(
                f"  {environment:15s}: "
                f"{count}"
            )

    # -------------------------------------------------------------
    # Source breakdown
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES BY TRAJECTORY SOURCE")
    print("=" * 70)

    for model_name, result in [
        ("Random Forest", rf_results),
        ("XGBoost", xgb_results),
    ]:

        print(f"\n{model_name}:")

        if result.empty:
            print("  None")
            continue

        counts = (
            result[
                "source"
            ]
            .value_counts()
            .sort_index()
        )

        for source, count in counts.items():

            print(
                f"  {source:15s}: "
                f"{count}"
            )

    # -------------------------------------------------------------
    # Compare overlap
    # -------------------------------------------------------------

    rf_set = set(
        rf_fn_indices
    )

    xgb_set = set(
        xgb_fn_indices
    )

    both = sorted(
        rf_set & xgb_set
    )

    rf_only = sorted(
        rf_set - xgb_set
    )

    xgb_only = sorted(
        xgb_set - rf_set
    )

    print("\n" + "=" * 70)
    print("FALSE-NEGATIVE OVERLAP")
    print("=" * 70)

    print(
        f"Missed by BOTH models : "
        f"{len(both)}"
    )

    print(
        f"RF only               : "
        f"{len(rf_only)}"
    )

    print(
        f"XGBoost only          : "
        f"{len(xgb_only)}"
    )

    if both:
        print(
            "\nRows missed by both:"
        )
        print(both)

    # -------------------------------------------------------------
    # Save combined results
    # -------------------------------------------------------------

    combined_rows = []

    for idx in rf_fn_indices:

        feature_row = X_test.iloc[idx]
        meta_row = metadata.iloc[idx]

        combined_rows.append({

            "test_row":
                int(idx),

            "model":
                "random_forest",

            "failure_probability":
                float(rf_prob[idx]),

            "environment_type":
                meta_row["environment_type"],

            "problem_index":
                int(meta_row["problem_index"]),

            "source":
                meta_row["source"],

            "interior_mean_clearance":
                float(
                    feature_row[
                        "interior_mean_clearance"
                    ]
                ),

            "interior_low_clearance_count":
                float(
                    feature_row[
                        "interior_low_clearance_count"
                    ]
                ),

            "interior_low_clearance_fraction":
                float(
                    feature_row[
                        "interior_low_clearance_fraction"
                    ]
                ),

            "number_of_low_clearance_runs":
                float(
                    feature_row[
                        "number_of_low_clearance_runs"
                    ]
                ),

            "longest_low_clearance_run":
                float(
                    feature_row[
                        "longest_low_clearance_run"
                    ]
                ),
        })

    for idx in xgb_fn_indices:

        feature_row = X_test.iloc[idx]
        meta_row = metadata.iloc[idx]

        combined_rows.append({

            "test_row":
                int(idx),

            "model":
                "xgboost",

            "failure_probability":
                float(xgb_prob[idx]),

            "environment_type":
                meta_row["environment_type"],

            "problem_index":
                int(meta_row["problem_index"]),

            "source":
                meta_row["source"],

            "interior_mean_clearance":
                float(
                    feature_row[
                        "interior_mean_clearance"
                    ]
                ),

            "interior_low_clearance_count":
                float(
                    feature_row[
                        "interior_low_clearance_count"
                    ]
                ),

            "interior_low_clearance_fraction":
                float(
                    feature_row[
                        "interior_low_clearance_fraction"
                    ]
                ),

            "number_of_low_clearance_runs":
                float(
                    feature_row[
                        "number_of_low_clearance_runs"
                    ]
                ),

            "longest_low_clearance_run":
                float(
                    feature_row[
                        "longest_low_clearance_run"
                    ]
                ),
        })

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved detailed analysis:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("FALSE-NEGATIVE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()