from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from prediction.failure_predictor import FailurePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECORDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "trajectory_records.pkl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "models"
    / "random_forest.joblib"
)

FEATURE_NAMES_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "models"
    / "feature_names.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

LOW_CLEARANCE_THRESHOLD = 0.02


# ============================================================
# LOAD DATA
# ============================================================

def load_records():
    print("=" * 70)
    print("LOADING TRAJECTORY RECORDS")
    print("=" * 70)

    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)

    print(f"Total records: {len(records)}")

    return records


# ============================================================
# LOAD MODEL
# ============================================================

def load_predictor():
    predictor = FailurePredictor(
        model_path=str(MODEL_PATH),
        feature_names_path=str(FEATURE_NAMES_PATH),
    )

    print("Random Forest predictor loaded.")

    return predictor


# ============================================================
# CREATE PREDICTIONS
# ============================================================

def generate_predictions(records, predictor):

    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    # Load the exact feature names used during model training
    import json

    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = json.load(f)

    print(f"Number of ML features: {len(feature_names)}")

    rows = []

    for index, record in enumerate(records):

        # ----------------------------------------------------
        # Extract exactly the features used by the model
        # ----------------------------------------------------

        feature_dict = {
            feature: record[feature]
            for feature in feature_names
        }

        prediction = predictor.predict_from_features(feature_dict)

        predicted_failure = int(prediction["prediction"])
        failure_probability = float(
            prediction["failure_probability"]
        )

        actual_failure = int(record["failure"])

        rows.append({
            "record_index": index,
            "actual_failure": actual_failure,
            "predicted_failure": predicted_failure,
            "failure_probability": failure_probability,

            "environment_type": record["environment_type"],
            "source": record["source"],
            "subset": record["subset"],
            "problem_index": record["problem_index"],

            "collision_count": record["collision_count"],
            "obstacle_collision_count": record[
                "obstacle_collision_count"
            ],
            "self_collision_count": record[
                "self_collision_count"
            ],

            "minimum_clearance": record["minimum_clearance"],
            "interior_minimum_clearance": record[
                "interior_minimum_clearance"
            ],
            "interior_mean_clearance": record[
                "interior_mean_clearance"
            ],
            "interior_clearance_std": record[
                "interior_clearance_std"
            ],
            "interior_maximum_clearance": record[
                "interior_maximum_clearance"
            ],

            "interior_low_clearance_count": record[
                "interior_low_clearance_count"
            ],
            "interior_low_clearance_fraction": record[
                "interior_low_clearance_fraction"
            ],

            "joint_space_length": record[
                "joint_space_length"
            ],
            "max_joint_step": record[
                "max_joint_step"
            ],
            "max_joint_velocity": record[
                "max_joint_velocity"
            ],
            "mean_joint_velocity": record[
                "mean_joint_velocity"
            ],
            "max_joint_acceleration": record[
                "max_joint_acceleration"
            ],
            "mean_joint_acceleration": record[
                "mean_joint_acceleration"
            ],

            "minimum_joint_limit_distance": record[
                "minimum_joint_limit_distance"
            ],
            "mean_joint_limit_distance": record[
                "mean_joint_limit_distance"
            ],
            "minimum_normalized_joint_limit_distance": record[
                "minimum_normalized_joint_limit_distance"
            ],
            "mean_normalized_joint_limit_distance": record[
                "mean_normalized_joint_limit_distance"
            ],

            "num_points": record["num_points"],
        })

    df = pd.DataFrame(rows)

    return df

            


# ============================================================
# FIND FALSE NEGATIVES
# ============================================================

def find_false_negatives(df):

    false_negatives = df[
        (df["actual_failure"] == 1)
        & (df["predicted_failure"] == 0)
    ].copy()

    false_negatives = false_negatives.sort_values(
        "failure_probability"
    )

    return false_negatives


# ============================================================
# PRINT FALSE NEGATIVES
# ============================================================

def print_false_negatives(df):

    print("\n" + "=" * 70)
    print("FALSE NEGATIVE SUMMARY")
    print("=" * 70)

    print(f"False negatives found: {len(df)}")

    if len(df) == 0:
        print("\nNo false negatives found.")
        return

    display_columns = [
        "record_index",
        "environment_type",
        "source",
        "problem_index",
        "failure_probability",
        "collision_count",
        "obstacle_collision_count",
        "interior_minimum_clearance",
        "interior_mean_clearance",
        "interior_low_clearance_count",
        "interior_low_clearance_fraction",
        "joint_space_length",
    ]

    print("\n")
    print(df[display_columns].to_string(index=False))


# ============================================================
# COLLISION LOCALIZATION ANALYSIS
# ============================================================

def analyze_collision_localization(records, false_negative_df):

    print("\n" + "=" * 70)
    print("COLLISION LOCALIZATION ANALYSIS")
    print("=" * 70)

    results = []

    for _, row in false_negative_df.iterrows():

        record_index = int(row["record_index"])
        record = records[record_index]

        trajectory = np.asarray(record["_trajectory"])

        collision_mask = []

        # ----------------------------------------------------
        # Reconstruct collision locations using the stored
        # collision count information.
        #
        # The record does not store the complete per-point
        # collision mask, so this section reports the overall
        # collision density rather than inventing exact points.
        # ----------------------------------------------------

        num_points = len(trajectory)
        collision_count = int(record["collision_count"])

        collision_fraction = (
            collision_count / num_points
            if num_points > 0
            else 0.0
        )

        results.append({
            "record_index": record_index,
            "num_points": num_points,
            "collision_count": collision_count,
            "collision_fraction": collision_fraction,
            "interior_mean_clearance": record[
                "interior_mean_clearance"
            ],
            "interior_minimum_clearance": record[
                "interior_minimum_clearance"
            ],
            "low_clearance_fraction": record[
                "interior_low_clearance_fraction"
            ],
        })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:

        print("\nCollision density for false negatives:")

        print(
            result_df[
                [
                    "record_index",
                    "num_points",
                    "collision_count",
                    "collision_fraction",
                    "interior_mean_clearance",
                    "interior_minimum_clearance",
                    "low_clearance_fraction",
                ]
            ].to_string(index=False)
        )

        print(
            "\nAverage collision fraction among false negatives: "
            f"{result_df['collision_fraction'].mean():.4f}"
        )


# ============================================================
# COMPARE FALSE NEGATIVES WITH CORRECT FAILURES
# ============================================================

def compare_with_correct_failures(df):

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES vs CORRECTLY PREDICTED FAILURES")
    print("=" * 70)

    correct_failures = df[
        (df["actual_failure"] == 1)
        & (df["predicted_failure"] == 1)
    ].copy()

    false_negatives = df[
        (df["actual_failure"] == 1)
        & (df["predicted_failure"] == 0)
    ].copy()

    if len(false_negatives) == 0:
        print("No false negatives available.")
        return

    features_to_compare = [
        "collision_count",
        "obstacle_collision_count",
        "interior_minimum_clearance",
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
    ]

    comparison = []

    for feature in features_to_compare:

        fn_mean = false_negatives[feature].mean()
        cf_mean = correct_failures[feature].mean()

        comparison.append({
            "feature": feature,
            "false_negative_mean": fn_mean,
            "correct_failure_mean": cf_mean,
            "difference": fn_mean - cf_mean,
        })

    comparison_df = pd.DataFrame(comparison)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# ENVIRONMENT ANALYSIS
# ============================================================

def analyze_environments(false_negative_df):

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES BY ENVIRONMENT")
    print("=" * 70)

    if len(false_negative_df) == 0:
        print("No false negatives.")
        return

    counts = (
        false_negative_df
        .groupby("environment_type")
        .size()
        .sort_values(ascending=False)
    )

    print(counts.to_string())


# ============================================================
# SOURCE ANALYSIS
# ============================================================

def analyze_sources(false_negative_df):

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES BY TRAJECTORY SOURCE")
    print("=" * 70)

    if len(false_negative_df) == 0:
        print("No false negatives.")
        return

    counts = (
        false_negative_df
        .groupby("source")
        .size()
        .sort_values(ascending=False)
    )

    print(counts.to_string())


# ============================================================
# IDENTIFY POTENTIAL LOCALIZED COLLISION PATTERN
# ============================================================

def analyze_patterns(false_negative_df):

    print("\n" + "=" * 70)
    print("PATTERN CHECK")
    print("=" * 70)

    if len(false_negative_df) == 0:
        print("No false negatives.")
        return

    # --------------------------------------------------------
    # Collision fraction
    # --------------------------------------------------------

    collision_fraction = (
        false_negative_df["collision_count"]
        / false_negative_df["num_points"]
    )

    print(
        f"\nMean collision fraction: "
        f"{collision_fraction.mean():.4f}"
    )

    print(
        f"Median collision fraction: "
        f"{collision_fraction.median():.4f}"
    )

    # --------------------------------------------------------
    # How many have very few collision configurations?
    # --------------------------------------------------------

    thresholds = [1, 3, 5, 10, 20]

    print("\nFalse negatives with collision count ≤ threshold:")

    for threshold in thresholds:

        count = (
            false_negative_df["collision_count"]
            <= threshold
        ).sum()

        print(
            f"  <= {threshold:2d} configurations: "
            f"{count}/{len(false_negative_df)}"
        )

    # --------------------------------------------------------
    # Low-clearance fraction
    # --------------------------------------------------------

    print(
        "\nLow-clearance fraction statistics:"
    )

    print(
        f"  Mean   : "
        f"{false_negative_df['interior_low_clearance_fraction'].mean():.4f}"
    )

    print(
        f"  Median : "
        f"{false_negative_df['interior_low_clearance_fraction'].median():.4f}"
    )

    # --------------------------------------------------------
    # Average clearance
    # --------------------------------------------------------

    print(
        "\nInterior mean clearance statistics:"
    )

    print(
        f"  Mean   : "
        f"{false_negative_df['interior_mean_clearance'].mean():.6f}"
    )

    print(
        f"  Median : "
        f"{false_negative_df['interior_mean_clearance'].median():.6f}"
    )

    # --------------------------------------------------------
    # Main diagnostic statement
    # --------------------------------------------------------

    print("\nInterpretation guide:")

    if collision_fraction.mean() < 0.05:
        print(
            "  → False negatives tend to have collisions in "
            "only a small fraction of trajectory configurations."
        )

    if (
        false_negative_df["interior_mean_clearance"].mean()
        > 0.05
    ):
        print(
            "  → Many false negatives still have relatively "
            "large average clearance."
        )

    if (
        false_negative_df["interior_low_clearance_fraction"].mean()
        < 0.20
    ):
        print(
            "  → Low-clearance regions may occupy only a "
            "small portion of the trajectory."
        )


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_results(df, false_negatives):

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "failure_prediction"
        / "analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions_path = (
        output_dir / "false_negative_analysis_all_predictions.csv"
    )

    false_negative_path = (
        output_dir / "false_negatives.csv"
    )

    df.to_csv(all_predictions_path, index=False)
    false_negatives.to_csv(false_negative_path, index=False)

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(all_predictions_path)
    print(false_negative_path)


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_records()

    predictor = load_predictor()

    df = generate_predictions(
        records,
        predictor
    )

    false_negatives = find_false_negatives(df)

    print_false_negatives(false_negatives)

    analyze_collision_localization(
        records,
        false_negatives
    )

    compare_with_correct_failures(df)

    analyze_environments(false_negatives)

    analyze_sources(false_negatives)

    analyze_patterns(false_negatives)

    save_results(
        df,
        false_negatives
    )

    print("\n" + "=" * 70)
    print("FALSE NEGATIVE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()