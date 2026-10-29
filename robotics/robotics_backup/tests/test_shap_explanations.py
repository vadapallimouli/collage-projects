import os
import sys
import pickle
import numpy as np

# Add project root to Python import path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )

from prediction.failure_predictor import FailurePredictor


RECORDS_PATH = "data/failure_prediction/trajectory_records.pkl"
MODEL_PATH = "data/failure_prediction/models/random_forest.joblib"
FEATURE_NAMES_PATH = "data/failure_prediction/models/feature_names.json"


def print_explanation(result):
    print("\nPrediction:")
    print(f"  Label              : {result['label']}")
    print(
        f"  Success probability: "
        f"{result['success_probability']:.4f}"
    )
    print(
        f"  Failure probability: "
        f"{result['failure_probability']:.4f}"
    )

    print("\nTop SHAP explanation factors:")

    for i, factor in enumerate(
        result["explanation"],
        start=1,
    ):
        direction = (
            "→ FAILURE"
            if factor["shap_value"] > 0
            else "→ SUCCESS"
        )

        print(
            f"\n{i}. {factor['feature']}"
        )

        print(
            f"   Value              : "
            f"{factor['value']:.6f}"
        )

        print(
            f"   SHAP value         : "
            f"{factor['shap_value']:.6f}"
        )

        print(
            f"   Explanation strength: "
            f"{factor['explanation_score']:.6f}"
        )

        print(
            f"   Global importance  : "
            f"{factor['model_importance']:.6f}"
        )

        print(
            f"   Direction          : "
            f"{direction}"
        )

        print(
            f"   Reason             : "
            f"{factor['reason']}"
        )


def find_examples(records, predictor):
    """
    Find:
      1. Correct success
      2. Correct failure
      3. False negative
    """

    correct_success = None
    correct_failure = None
    false_negative = None

    for index, record in enumerate(records):

        features = {
            feature: record[feature]
            for feature in predictor.ML_FEATURES
        }

        prediction = predictor.predict_from_features(
            features
        )

        actual = int(record["failure"])
        predicted = int(prediction["prediction"])

        # Correct success
        if (
            actual == 0
            and predicted == 0
            and correct_success is None
        ):
            correct_success = index

        # Correct failure
        if (
            actual == 1
            and predicted == 1
            and correct_failure is None
        ):
            correct_failure = index

        # False negative:
        # Actual failure, predicted success
        if (
            actual == 1
            and predicted == 0
            and false_negative is None
        ):
            false_negative = index

        if (
            correct_success is not None
            and correct_failure is not None
            and false_negative is not None
        ):
            break

    return (
        correct_success,
        correct_failure,
        false_negative,
    )


def analyze_record(
    records,
    index,
    predictor,
    title,
):
    record = records[index]

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    print(
        f"\nRecord index       : {index}"
    )

    print(
        f"Environment        : "
        f"{record['environment_type']}"
    )

    print(
        f"Problem index      : "
        f"{record['problem_index']}"
    )

    print(
        f"Source             : "
        f"{record['source']}"
    )

    print(
        f"Trajectory index   : "
        f"{record['trajectory_index']}"
    )

    print(
        f"Ground truth       : "
        f"{'FAILURE' if record['failure'] == 1 else 'SUCCESS'}"
    )

    print(
        f"Collision count    : "
        f"{record['collision_count']}"
    )

    print(
        f"Obstacle collisions: "
        f"{record['obstacle_collision_count']}"
    )

    print(
        f"Minimum clearance  : "
        f"{record['minimum_clearance']:.6f}"
    )

    print(
        f"Interior mean clearance: "
        f"{record['interior_mean_clearance']:.6f}"
    )

    print(
        f"Low-clearance fraction: "
        f"{record['interior_low_clearance_fraction']:.6f}"
    )

    print(
        f"Joint-space length : "
        f"{record['joint_space_length']:.6f}"
    )

    trajectory = np.asarray(
        record["_trajectory"],
        dtype=float,
    )

    print(
        f"Trajectory shape   : "
        f"{trajectory.shape}"
    )

    features = {
        feature: record[feature]
        for feature in predictor.ML_FEATURES
    }

    result = predictor.predict_and_explain(
        features,
        top_k=5,
    )

    print_explanation(result)

    return result


def main():

    print("=" * 75)
    print("SHAP EXPLAINABILITY VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        RECORDS_PATH,
        MODEL_PATH,
        FEATURE_NAMES_PATH,
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    print("\nLoading trajectory records...")

    with open(
        RECORDS_PATH,
        "rb",
    ) as f:

        records = pickle.load(f)

    print(
        f"Records loaded: {len(records)}"
    )

    # --------------------------------------------------------
    # Load predictor
    # --------------------------------------------------------

    print("\nLoading FailurePredictor...")

    predictor = FailurePredictor(
        model_path=MODEL_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
    )

    print(
        "Random Forest + SHAP loaded."
    )

    # --------------------------------------------------------
    # Find representative examples
    # --------------------------------------------------------

    print(
        "\nSearching for representative examples..."
    )

    (
        success_index,
        failure_index,
        false_negative_index,
    ) = find_examples(
        records,
        predictor,
    )

    print(
        f"\nCorrect success record : "
        f"{success_index}"
    )

    print(
        f"Correct failure record : "
        f"{failure_index}"
    )

    print(
        f"False negative record  : "
        f"{false_negative_index}"
    )

    if success_index is None:
        raise RuntimeError(
            "Could not find a correct success example."
        )

    if failure_index is None:
        raise RuntimeError(
            "Could not find a correct failure example."
        )

    if false_negative_index is None:
        raise RuntimeError(
            "Could not find a false-negative example."
        )

    # --------------------------------------------------------
    # 1. Correct success
    # --------------------------------------------------------

    success_result = analyze_record(
        records=records,
        index=success_index,
        predictor=predictor,
        title="CASE 1 — CORRECT SUCCESS PREDICTION",
    )

    # --------------------------------------------------------
    # 2. Correct failure
    # --------------------------------------------------------

    failure_result = analyze_record(
        records=records,
        index=failure_index,
        predictor=predictor,
        title="CASE 2 — CORRECT FAILURE PREDICTION",
    )

    # --------------------------------------------------------
    # 3. False negative
    # --------------------------------------------------------

    false_negative_result = analyze_record(
        records=records,
        index=false_negative_index,
        predictor=predictor,
        title="CASE 3 — FALSE NEGATIVE",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SHAP VALIDATION SUMMARY")
    print("=" * 75)

    print(
        "\nThree cases were analyzed:"
    )

    print(
        "1. Correctly predicted successful trajectory"
    )

    print(
        "2. Correctly predicted failed trajectory"
    )

    print(
        "3. Failed trajectory incorrectly predicted as success"
    )

    print(
        "\nSHAP explanations were generated successfully "
        "for all three cases."
    )

    print(
        "\nUse the false-negative case to understand "
        "where the current feature representation "
        "can miss localized collision behavior."
    )

    print("\n" + "=" * 75)
    print("SHAP EXPLAINABILITY TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()