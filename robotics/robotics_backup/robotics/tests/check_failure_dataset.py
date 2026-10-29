"""
Failure Prediction Dataset Quality Checker
===========================================

Checks the generated trajectory dataset before ML training.

Checks:
    1. File existence
    2. Number of samples
    3. Class distribution
    4. Missing values
    5. NaN / infinite values
    6. Duplicate rows
    7. Feature ranges
    8. Constant features
    9. ML feature / label leakage
    10. Distribution by environment
    11. Distribution by trajectory source
    12. Distribution by problem
    13. Feature statistics

This script DOES NOT train a model.
"""

import csv
import math
from pathlib import Path
from collections import Counter, defaultdict


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "trajectory_features.csv"
)


# ================================================================
# EXPECTED ML FEATURES
# ================================================================

ML_FEATURES = [
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
    "minimum_joint_limit_distance",
    "mean_joint_limit_distance",
    "minimum_normalized_joint_limit_distance",
    "mean_normalized_joint_limit_distance",
]


# ================================================================
# COLUMNS THAT MUST NOT BE ML INPUTS
# ================================================================

FORBIDDEN_ML_COLUMNS = [
    "failure",
    "collision",
    "collision_free",
    "collision_count",
    "self_collision_count",
    "obstacle_collision_count",
    "joint_limit_violation_count",
    "closest_obstacle_index",
    "closest_robot_link",
    "num_points",
    "environment_type",
    "subset",
    "problem_index",
    "source",
    "trajectory_index",
]


# ================================================================
# HELPERS
# ================================================================

def is_numeric(value):
    """
    Check whether a CSV value can safely be interpreted
    as a finite floating-point number.
    """

    try:
        number = float(value)

        return math.isfinite(number)

    except (
        TypeError,
        ValueError,
    ):
        return False


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# LOAD CSV
# ================================================================

def load_csv():

    print_section(
        "LOADING DATASET"
    )

    print(
        f"Dataset:\n{DATASET_PATH}"
    )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

        fieldnames = reader.fieldnames

    if fieldnames is None:

        raise RuntimeError(
            "CSV does not contain a header."
        )

    print(
        f"Rows loaded: {len(rows)}"
    )

    print(
        f"Columns: {len(fieldnames)}"
    )

    return rows, fieldnames


# ================================================================
# CHECK REQUIRED COLUMNS
# ================================================================

def check_columns(fieldnames):

    print_section(
        "1. COLUMN CHECK"
    )

    missing_features = [
        feature
        for feature in ML_FEATURES
        if feature not in fieldnames
    ]

    if missing_features:

        print(
            "FAIL: Missing ML features:"
        )

        for feature in missing_features:
            print(
                f"  - {feature}"
            )

    else:

        print(
            "PASS: All 16 ML features are present."
        )

    if "failure" not in fieldnames:

        print(
            "FAIL: 'failure' label is missing."
        )

    else:

        print(
            "PASS: Failure label is present."
        )

    print(
        "\nML feature count:",
        len(ML_FEATURES),
    )


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

def check_class_distribution(rows):

    print_section(
        "2. CLASS DISTRIBUTION"
    )

    labels = []

    invalid_labels = []

    for index, row in enumerate(rows):

        value = row.get(
            "failure",
            "",
        )

        try:

            label = int(value)

            if label not in (
                0,
                1,
            ):

                invalid_labels.append(
                    (
                        index,
                        value,
                    )
                )

            else:

                labels.append(
                    label
                )

        except ValueError:

            invalid_labels.append(
                (
                    index,
                    value,
                )
            )

    counter = Counter(
        labels
    )

    successful = counter[0]

    failed = counter[1]

    total = successful + failed

    print(
        f"Successful (0): {successful}"
    )

    print(
        f"Failed     (1): {failed}"
    )

    print(
        f"Total         : {total}"
    )

    if total > 0:

        print(
            f"Success ratio : "
            f"{successful / total:.3f}"
        )

        print(
            f"Failure ratio : "
            f"{failed / total:.3f}"
        )

    if invalid_labels:

        print(
            "\nFAIL: Invalid labels found:"
        )

        for item in invalid_labels[:10]:
            print(
                f"  row {item[0]} -> "
                f"{item[1]}"
            )

    else:

        print(
            "\nPASS: All labels are 0 or 1."
        )


# ================================================================
# MISSING VALUES
# ================================================================

def check_missing_values(
    rows,
    fieldnames,
):

    print_section(
        "3. MISSING VALUE CHECK"
    )

    missing_counts = Counter()

    for row in rows:

        for field in fieldnames:

            value = row.get(
                field,
                "",
            )

            if value is None or value.strip() == "":

                missing_counts[
                    field
                ] += 1

    if not missing_counts:

        print(
            "PASS: No missing values found."
        )

        return

    print(
        "Missing values:"
    )

    for field, count in (
        missing_counts.items()
    ):

        print(
            f"  {field}: {count}"
        )


# ================================================================
# NAN / INFINITY
# ================================================================

def check_nan_and_infinity(
    rows,
):

    print_section(
        "4. NaN / INFINITY CHECK"
    )

    problems = []

    for row_index, row in enumerate(
        rows
    ):

        for feature in ML_FEATURES:

            value = row.get(
                feature,
                "",
            )

            if not is_numeric(
                value
            ):

                problems.append(
                    (
                        row_index,
                        feature,
                        value,
                    )
                )

    if not problems:

        print(
            "PASS: All ML feature values are finite."
        )

        return

    print(
        f"FAIL: {len(problems)} invalid values found."
    )

    for row_index, feature, value in (
        problems[:20]
    ):

        print(
            f"  row={row_index}, "
            f"feature={feature}, "
            f"value={value}"
        )


# ================================================================
# DUPLICATE ROWS
# ================================================================

def check_duplicates(
    rows,
):

    print_section(
        "5. DUPLICATE CHECK"
    )

    signatures = {}

    duplicate_rows = []

    for index, row in enumerate(
        rows
    ):

        signature = tuple(
            row.get(
                feature,
                "",
            )
            for feature in ML_FEATURES
        )

        if signature in signatures:

            duplicate_rows.append(
                (
                    index,
                    signatures[
                        signature
                    ],
                )
            )

        else:

            signatures[
                signature
            ] = index

    print(
        f"Unique feature vectors: "
        f"{len(signatures)}"
    )

    print(
        f"Duplicate rows: "
        f"{len(duplicate_rows)}"
    )

    if duplicate_rows:

        print(
            "\nWARNING: Duplicate ML feature vectors detected."
        )

        for current, original in (
            duplicate_rows[:10]
        ):

            print(
                f"  row {current} "
                f"duplicates row {original}"
            )

    else:

        print(
            "PASS: No duplicate ML feature vectors."
        )


# ================================================================
# FEATURE STATISTICS
# ================================================================

def check_feature_statistics(
    rows,
):

    print_section(
        "6. FEATURE STATISTICS"
    )

    for feature in ML_FEATURES:

        values = []

        for row in rows:

            try:

                value = float(
                    row[feature]
                )

                if math.isfinite(
                    value
                ):

                    values.append(
                        value
                    )

            except (
                ValueError,
                KeyError,
            ):

                pass

        if not values:

            print(
                f"{feature:45s} "
                f"NO VALID VALUES"
            )

            continue

        minimum = min(values)
        maximum = max(values)

        mean = (
            sum(values)
            / len(values)
        )

        # Population variance.
        variance = (
            sum(
                (
                    value
                    - mean
                ) ** 2
                for value in values
            )
            / len(values)
        )

        std = math.sqrt(
            variance
        )

        print(
            f"{feature:45s} "
            f"min={minimum: .6f} "
            f"max={maximum: .6f} "
            f"mean={mean: .6f} "
            f"std={std: .6f}"
        )


# ================================================================
# CONSTANT FEATURES
# ================================================================

def check_constant_features(
    rows,
):

    print_section(
        "7. CONSTANT FEATURE CHECK"
    )

    constant_features = []

    for feature in ML_FEATURES:

        values = set()

        for row in rows:

            try:

                value = float(
                    row[feature]
                )

                if math.isfinite(
                    value
                ):

                    values.add(
                        value
                    )

            except (
                ValueError,
                KeyError,
            ):

                pass

        if len(values) <= 1:

            constant_features.append(
                feature
            )

    if not constant_features:

        print(
            "PASS: No constant ML features."
        )

    else:

        print(
            "WARNING: Constant features:"
        )

        for feature in constant_features:

            print(
                f"  - {feature}"
            )


# ================================================================
# LEAKAGE CHECK
# ================================================================

def check_feature_leakage(
    fieldnames,
):

    print_section(
        "8. TARGET LEAKAGE CHECK"
    )

    forbidden_present = [
        field
        for field in FORBIDDEN_ML_COLUMNS
        if field in ML_FEATURES
    ]

    if forbidden_present:

        print(
            "FAIL: Forbidden columns "
            "appear in ML feature list:"
        )

        for field in forbidden_present:

            print(
                f"  - {field}"
            )

    else:

        print(
            "PASS: No collision-outcome "
            "or metadata fields are ML inputs."
        )

    print(
        "\nActual ML features:"
    )

    for index, feature in enumerate(
        ML_FEATURES,
        start=1,
    ):

        print(
            f"  {index:2d}. {feature}"
        )


# ================================================================
# SOURCE DISTRIBUTION
# ================================================================

def check_source_distribution(
    rows,
):

    print_section(
        "9. TRAJECTORY SOURCE DISTRIBUTION"
    )

    source_stats = defaultdict(
        lambda: Counter()
    )

    for row in rows:

        source = row.get(
            "source",
            "unknown",
        )

        try:

            label = int(
                row["failure"]
            )

        except ValueError:

            continue

        if label == 0:

            source_stats[
                source
            ]["success"] += 1

        else:

            source_stats[
                source
            ]["failure"] += 1

    print(
        f"{'Source':15s}"
        f"{'Total':>10s}"
        f"{'Success':>10s}"
        f"{'Failure':>10s}"
        f"{'Fail %':>10s}"
    )

    print(
        "-" * 55
    )

    for source in sorted(
        source_stats
    ):

        success = source_stats[
            source
        ]["success"]

        failure = source_stats[
            source
        ]["failure"]

        total = (
            success
            + failure
        )

        failure_ratio = (
            failure / total
            if total
            else 0
        )

        print(
            f"{source:15s}"
            f"{total:10d}"
            f"{success:10d}"
            f"{failure:10d}"
            f"{failure_ratio:10.3f}"
        )


# ================================================================
# ENVIRONMENT DISTRIBUTION
# ================================================================

def check_environment_distribution(
    rows,
):

    print_section(
        "10. ENVIRONMENT DISTRIBUTION"
    )

    environment_stats = defaultdict(
        lambda: Counter()
    )

    for row in rows:

        environment = row.get(
            "environment_type",
            "unknown",
        )

        try:

            label = int(
                row["failure"]
            )

        except ValueError:

            continue

        if label == 0:

            environment_stats[
                environment
            ]["success"] += 1

        else:

            environment_stats[
                environment
            ]["failure"] += 1

    print(
        f"{'Environment':18s}"
        f"{'Total':>10s}"
        f"{'Success':>10s}"
        f"{'Failure':>10s}"
        f"{'Fail %':>10s}"
    )

    print(
        "-" * 65
    )

    for environment in sorted(
        environment_stats
    ):

        success = environment_stats[
            environment
        ]["success"]

        failure = environment_stats[
            environment
        ]["failure"]

        total = (
            success
            + failure
        )

        failure_ratio = (
            failure / total
            if total
            else 0
        )

        print(
            f"{environment:18s}"
            f"{total:10d}"
            f"{success:10d}"
            f"{failure:10d}"
            f"{failure_ratio:10.3f}"
        )


# ================================================================
# PROBLEM DISTRIBUTION
# ================================================================

def check_problem_distribution(
    rows,
):

    print_section(
        "11. PROBLEM-LEVEL DISTRIBUTION"
    )

    problem_stats = defaultdict(
        lambda: Counter()
    )

    for row in rows:

        environment = row.get(
            "environment_type",
            "unknown",
        )

        problem = row.get(
            "problem_index",
            "unknown",
        )

        key = (
            environment,
            problem,
        )

        try:

            label = int(
                row["failure"]
            )

        except ValueError:

            continue

        if label == 0:

            problem_stats[
                key
            ]["success"] += 1

        else:

            problem_stats[
                key
            ]["failure"] += 1

    print(
        f"{'Environment':16s}"
        f"{'Problem':>10s}"
        f"{'Success':>10s}"
        f"{'Failure':>10s}"
    )

    print(
        "-" * 50
    )

    for (
        environment,
        problem,
    ) in sorted(
        problem_stats
    ):

        success = problem_stats[
            (
                environment,
                problem,
            )
        ]["success"]

        failure = problem_stats[
            (
                environment,
                problem,
            )
        ]["failure"]

        print(
            f"{environment:16s}"
            f"{problem:10s}"
            f"{success:10d}"
            f"{failure:10d}"
        )


# ================================================================
# RANGE SANITY CHECK
# ================================================================

def check_feature_sanity(
    rows,
):

    print_section(
        "12. FEATURE SANITY CHECK"
    )

    problems = []

    for row_index, row in enumerate(
        rows
    ):

        # ------------------------------------------------------------
        # Clearance
        # ------------------------------------------------------------

        clearance_features = [
            "interior_minimum_clearance",
            "interior_mean_clearance",
            "interior_clearance_std",
            "interior_maximum_clearance",
        ]

        for feature in (
            clearance_features
        ):

            try:

                value = float(
                    row[feature]
                )

            except (
                ValueError,
                KeyError,
            ):

                continue

            if not math.isfinite(
                value
            ):

                problems.append(
                    (
                        row_index,
                        feature,
                        "non-finite",
                    )
                )

        # ------------------------------------------------------------
        # Fractions should be [0, 1]
        # ------------------------------------------------------------

        try:

            fraction = float(
                row[
                    "interior_low_clearance_fraction"
                ]
            )

            if not (
                0.0
                <= fraction
                <= 1.0
            ):

                problems.append(
                    (
                        row_index,
                        "interior_low_clearance_fraction",
                        fraction,
                    )
                )

        except (
            ValueError,
            KeyError,
        ):

            pass

        # ------------------------------------------------------------
        # Standard deviation cannot be negative.
        # ------------------------------------------------------------

        try:

            std = float(
                row[
                    "interior_clearance_std"
                ]
            )

            if std < 0:

                problems.append(
                    (
                        row_index,
                        "interior_clearance_std",
                        std,
                    )
                )

        except (
            ValueError,
            KeyError,
        ):

            pass

        # ------------------------------------------------------------
        # Counts cannot be negative.
        # ------------------------------------------------------------

        count_features = [
            "interior_low_clearance_count",
        ]

        for feature in count_features:

            try:

                value = float(
                    row[feature]
                )

                if value < 0:

                    problems.append(
                        (
                            row_index,
                            feature,
                            value,
                        )
                    )

            except (
                ValueError,
                KeyError,
            ):

                pass

    if not problems:

        print(
            "PASS: No obvious feature-range violations."
        )

    else:

        print(
            f"WARNING: "
            f"{len(problems)} "
            f"potential feature issues."
        )

        for problem in problems[:20]:

            print(
                f"  row={problem[0]}, "
                f"feature={problem[1]}, "
                f"value={problem[2]}"
            )


# ================================================================
# FINAL VERDICT
# ================================================================

def final_verdict(
    rows,
):

    print_section(
        "FINAL DATASET CHECK"
    )

    checks = []

    # ------------------------------------------------------------
    # Enough samples
    # ------------------------------------------------------------

    checks.append(
        (
            "At least 100 trajectories",
            len(rows) >= 100,
        )
    )

    # ------------------------------------------------------------
    # Both classes
    # ------------------------------------------------------------

    labels = set()

    for row in rows:

        try:

            labels.add(
                int(
                    row["failure"]
                )
            )

        except (
            ValueError,
            KeyError,
        ):

            pass

    checks.append(
        (
            "Both failure classes present",
            labels == {0, 1},
        )
    )

    # ------------------------------------------------------------
    # All ML features present
    # ------------------------------------------------------------

    # This is checked through the first row.
    if rows:

        available = set(
            rows[0].keys()
        )

        all_features_present = all(
            feature in available
            for feature in ML_FEATURES
        )

    else:

        all_features_present = False

    checks.append(
        (
            "All ML features present",
            all_features_present,
        )
    )

    # ------------------------------------------------------------
    # Invalid numeric values
    # ------------------------------------------------------------

    invalid_numeric = False

    for row in rows:

        for feature in ML_FEATURES:

            if not is_numeric(
                row.get(
                    feature,
                    "",
                )
            ):

                invalid_numeric = True

                break

        if invalid_numeric:
            break

    checks.append(
        (
            "No invalid ML feature values",
            not invalid_numeric,
        )
    )

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    passed = 0

    for name, result in checks:

        if result:

            print(
                f"PASS  {name}"
            )

            passed += 1

        else:

            print(
                f"FAIL  {name}"
            )

    print()

    if passed == len(checks):

        print(
            "DATASET BASIC CHECK: PASS"
        )

        print(
            "The dataset is ready for the "
            "next detailed ML-preparation step."
        )

    else:

        print(
            "DATASET BASIC CHECK: REVIEW REQUIRED"
        )

        print(
            "Fix the failed checks before training."
        )


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "ML FAILURE-PREDICTION DATASET "
        "QUALITY CHECK"
    )

    print(
        "=" * 70
    )

    rows, fieldnames = (
        load_csv()
    )

    check_columns(
        fieldnames
    )

    check_class_distribution(
        rows
    )

    check_missing_values(
        rows,
        fieldnames,
    )

    check_nan_and_infinity(
        rows
    )

    check_duplicates(
        rows
    )

    check_feature_statistics(
        rows
    )

    check_constant_features(
        rows
    )

    check_feature_leakage(
        fieldnames
    )

    check_source_distribution(
        rows
    )

    check_environment_distribution(
        rows
    )

    check_problem_distribution(
        rows
    )

    check_feature_sanity(
        rows
    )

    final_verdict(
        rows
    )

    print()
    print(
        "=" * 70
    )

    print(
        "DATASET QUALITY CHECK COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()