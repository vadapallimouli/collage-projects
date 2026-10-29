"""
ML Dataset Preparation
======================

Creates an environment-aware grouped train/test split.

Rules:
    - A complete planning problem stays in ONE split.
    - No planning problem appears in both train and test.
    - Every environment with usable problems is represented in both
      train and test whenever possible.
    - Approximately 20% of each environment's samples are used for
      testing.
"""

import csv
import random
from pathlib import Path
from collections import Counter, defaultdict


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "trajectory_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
)


# ================================================================
# SETTINGS
# ================================================================

RANDOM_SEED = 42

TEST_SIZE = 0.20


# ================================================================
# FINAL ML FEATURES
# ================================================================

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


# ================================================================
# METADATA
# ================================================================

METADATA_COLUMNS = [
    "environment_type",
    "subset",
    "problem_index",
    "source",
    "trajectory_index",
]


# ================================================================
# LOAD DATASET
# ================================================================

def load_dataset():

    print("=" * 70)
    print("LOADING FAILURE-PREDICTION DATASET")
    print("=" * 70)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

        fieldnames = reader.fieldnames

    if fieldnames is None:

        raise RuntimeError(
            "CSV header could not be read."
        )

    print(
        f"Input file:\n{INPUT_PATH}"
    )

    print(
        f"Rows: {len(rows)}"
    )

    print(
        f"Columns: {len(fieldnames)}"
    )

    return rows, fieldnames


# ================================================================
# CHECK REQUIRED COLUMNS
# ================================================================

def check_required_columns(
    fieldnames,
):

    required = (
        ML_FEATURES
        + METADATA_COLUMNS
        + ["failure"]
    )

    missing = [
        column
        for column in required
        if column not in fieldnames
    ]

    if missing:

        print(
            "\nERROR: Missing required columns:"
        )

        for column in missing:

            print(
                f"  - {column}"
            )

        raise RuntimeError(
            "Dataset is missing required columns."
        )

    print(
        "\nPASS: All required columns exist."
    )


# ================================================================
# VALIDATE ROWS
# ================================================================

def validate_rows(
    rows,
):

    print()
    print("=" * 70)
    print("VALIDATING ML FEATURES")
    print("=" * 70)

    valid_rows = []

    for index, row in enumerate(rows):

        try:

            failure = int(
                row["failure"]
            )

            if failure not in (0, 1):

                raise ValueError(
                    "failure must be 0 or 1"
                )

            feature_values = {}

            for feature in ML_FEATURES:

                value = float(
                    row[feature]
                )

                if not (
                    value == value
                ):
                    raise ValueError(
                        f"{feature} is NaN"
                    )

                if value in (
                    float("inf"),
                    float("-inf"),
                ):
                    raise ValueError(
                        f"{feature} is infinite"
                    )

                feature_values[
                    feature
                ] = value

            environment = row[
                "environment_type"
            ]

            problem_index = row[
                "problem_index"
            ]

            group_id = (
                f"{environment}__"
                f"{problem_index}"
            )

            valid_rows.append(
                {
                    "features": feature_values,
                    "failure": failure,
                    "environment_type": environment,
                    "subset": row["subset"],
                    "problem_index": problem_index,
                    "source": row["source"],
                    "trajectory_index": row[
                        "trajectory_index"
                    ],
                    "group_id": group_id,
                }
            )

        except Exception as error:

            print(
                f"ERROR in row {index}: "
                f"{error}"
            )

            raise

    print(
        f"Valid rows: {len(valid_rows)}"
    )

    print(
        "PASS: All rows contain valid ML data."
    )

    return valid_rows


# ================================================================
# CREATE PROBLEM GROUPS
# ================================================================

def create_groups(
    rows,
):

    groups = defaultdict(list)

    for row in rows:

        groups[
            row["group_id"]
        ].append(row)

    return groups


# ================================================================
# GROUP BY ENVIRONMENT
# ================================================================

def create_environment_groups(
    groups,
):

    environment_groups = defaultdict(list)

    for group_id in groups:

        environment = group_id.split(
            "__",
            1,
        )[0]

        environment_groups[
            environment
        ].append(group_id)

    return environment_groups


# ================================================================
# PRINT GROUP SUMMARY
# ================================================================

def print_group_summary(
    groups,
):

    print()
    print("=" * 70)
    print("PLANNING PROBLEM GROUPS")
    print("=" * 70)

    print(
        f"Total groups: {len(groups)}"
    )

    for group_id in sorted(groups):

        rows = groups[
            group_id
        ]

        labels = Counter(
            row["failure"]
            for row in rows
        )

        print(
            f"{group_id:30s} "
            f"total={len(rows):3d} "
            f"success={labels[0]:2d} "
            f"failure={labels[1]:2d}"
        )


# ================================================================
# ENVIRONMENT-AWARE GROUPED SPLIT
# ================================================================

def create_grouped_split(
    groups,
):

    print()
    print("=" * 70)
    print("CREATING ENVIRONMENT-AWARE GROUPED SPLIT")
    print("=" * 70)

    environment_groups = (
        create_environment_groups(
            groups
        )
    )

    random.seed(
        RANDOM_SEED
    )

    train_groups = []
    test_groups = []

    # ------------------------------------------------------------
    # Process every environment independently.
    # ------------------------------------------------------------

    for environment in sorted(
        environment_groups
    ):

        group_ids = list(
            environment_groups[
                environment
            ]
        )

        random.shuffle(
            group_ids
        )

        print()
        print(
            f"Environment: {environment}"
        )

        print(
            f"Available planning problems: "
            f"{len(group_ids)}"
        )

        # --------------------------------------------------------
        # At least one test problem when possible.
        # --------------------------------------------------------

        if len(group_ids) >= 2:

            test_group_count = max(
                1,
                round(
                    len(group_ids)
                    * TEST_SIZE
                ),
            )

            # Never put every group into test.
            test_group_count = min(
                test_group_count,
                len(group_ids) - 1,
            )

        else:

            # If an environment had only one problem,
            # it must remain in training.
            test_group_count = 0

        selected_test = (
            group_ids[
                :test_group_count
            ]
        )

        selected_train = (
            group_ids[
                test_group_count:
            ]
        )

        test_groups.extend(
            selected_test
        )

        train_groups.extend(
            selected_train
        )

        print(
            f"Train groups: "
            f"{len(selected_train)}"
        )

        print(
            f"Test groups : "
            f"{len(selected_test)}"
        )

        if selected_test:

            print(
                "Test problems:"
            )

            for group_id in sorted(
                selected_test
            ):

                print(
                    f"  {group_id}"
                )

    # ------------------------------------------------------------
    # Convert groups to rows.
    # ------------------------------------------------------------

    train_group_set = set(
        train_groups
    )

    test_group_set = set(
        test_groups
    )

    train_rows = []

    test_rows = []

    for group_id in train_groups:

        train_rows.extend(
            groups[group_id]
        )

    for group_id in test_groups:

        test_rows.extend(
            groups[group_id]
        )

    # ------------------------------------------------------------
    # Sanity check.
    # ------------------------------------------------------------

    overlap = (
        train_group_set
        & test_group_set
    )

    if overlap:

        raise RuntimeError(
            "Group leakage detected."
        )

    print()
    print(
        "FINAL SPLIT"
    )

    print(
        f"Training groups : "
        f"{len(train_groups)}"
    )

    print(
        f"Testing groups  : "
        f"{len(test_groups)}"
    )

    print(
        f"Training rows   : "
        f"{len(train_rows)}"
    )

    print(
        f"Testing rows    : "
        f"{len(test_rows)}"
    )

    return (
        train_rows,
        test_rows,
        train_groups,
        test_groups,
    )


# ================================================================
# GROUP LEAKAGE CHECK
# ================================================================

def check_group_leakage(
    train_rows,
    test_rows,
):

    print()
    print("=" * 70)
    print("GROUP LEAKAGE CHECK")
    print("=" * 70)

    train_groups = {
        row["group_id"]
        for row in train_rows
    }

    test_groups = {
        row["group_id"]
        for row in test_rows
    }

    overlap = (
        train_groups
        & test_groups
    )

    if overlap:

        print(
            "FAIL: Group leakage detected."
        )

        for group in sorted(
            overlap
        ):

            print(
                f"  {group}"
            )

        raise RuntimeError(
            "Same planning problem exists "
            "in both train and test."
        )

    print(
        "PASS: No planning problem appears "
        "in both train and test."
    )


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

def print_class_distribution(
    name,
    rows,
):

    counter = Counter(
        row["failure"]
        for row in rows
    )

    total = len(rows)

    success = counter[0]

    failure = counter[1]

    print()
    print(
        f"{name.upper()} CLASS DISTRIBUTION"
    )

    print(
        f"Total     : {total}"
    )

    print(
        f"Success   : {success}"
    )

    print(
        f"Failure   : {failure}"
    )

    if total:

        print(
            f"Success % : "
            f"{100 * success / total:.2f}%"
        )

        print(
            f"Failure % : "
            f"{100 * failure / total:.2f}%"
        )


# ================================================================
# ENVIRONMENT DISTRIBUTION
# ================================================================

def print_environment_distribution(
    name,
    rows,
):

    print()
    print(
        f"{name.upper()} ENVIRONMENT DISTRIBUTION"
    )

    stats = defaultdict(
        Counter
    )

    for row in rows:

        stats[
            row["environment_type"]
        ][
            row["failure"]
        ] += 1

    print(
        f"{'Environment':18s}"
        f"{'Total':>10s}"
        f"{'Success':>10s}"
        f"{'Failure':>10s}"
    )

    print(
        "-" * 50
    )

    for environment in sorted(
        stats
    ):

        success = stats[
            environment
        ][0]

        failure = stats[
            environment
        ][1]

        print(
            f"{environment:18s}"
            f"{success + failure:10d}"
            f"{success:10d}"
            f"{failure:10d}"
        )


# ================================================================
# SAVE DATASET
# ================================================================

def save_dataset(
    filename,
    rows,
):

    path = (
        OUTPUT_DIR
        / filename
    )

    columns = (
        METADATA_COLUMNS
        + ML_FEATURES
        + ["failure"]
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:

            output = {}

            for column in columns:

                if column in ML_FEATURES:

                    output[
                        column
                    ] = row[
                        "features"
                    ][column]

                elif column == "failure":

                    output[
                        column
                    ] = row[
                        "failure"
                    ]

                else:

                    output[
                        column
                    ] = row[
                        column
                    ]

            writer.writerow(
                output
            )

    print(
        f"Saved: {path}"
    )


# ================================================================
# SAVE LABELS
# ================================================================

def save_labels(
    filename,
    rows,
):

    path = (
        OUTPUT_DIR
        / filename
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            ["failure"]
        )

        for row in rows:

            writer.writerow(
                [
                    row["failure"]
                ]
            )

    print(
        f"Saved: {path}"
    )


# ================================================================
# SAVE SPLIT METADATA
# ================================================================

def save_split_metadata(
    train_groups,
    test_groups,
):

    path = (
        OUTPUT_DIR
        / "split_metadata.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "group_id",
                "environment_type",
                "problem_index",
                "split",
            ]
        )

        for group_id in sorted(
            train_groups
        ):

            environment, problem = (
                group_id.split(
                    "__",
                    1,
                )
            )

            writer.writerow(
                [
                    group_id,
                    environment,
                    problem,
                    "train",
                ]
            )

        for group_id in sorted(
            test_groups
        ):

            environment, problem = (
                group_id.split(
                    "__",
                    1,
                )
            )

            writer.writerow(
                [
                    group_id,
                    environment,
                    problem,
                    "test",
                ]
            )

    print(
        f"Saved: {path}"
    )


# ================================================================
# VERIFY OUTPUT
# ================================================================

def verify_output(
    filename,
    expected_rows,
):

    path = (
        OUTPUT_DIR
        / filename
    )

    if not path.exists():

        raise RuntimeError(
            f"Missing output file:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

    if len(rows) != expected_rows:

        raise RuntimeError(
            f"{filename}: expected "
            f"{expected_rows} rows, found "
            f"{len(rows)}."
        )

    print(
        f"PASS: {filename} "
        f"contains {len(rows)} rows."
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 70)
    print("ML DATASET PREPARATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    rows, fieldnames = (
        load_dataset()
    )

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------

    check_required_columns(
        fieldnames
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    rows = validate_rows(
        rows
    )

    # ------------------------------------------------------------
    # Create groups
    # ------------------------------------------------------------

    groups = create_groups(
        rows
    )

    print_group_summary(
        groups
    )

    # ------------------------------------------------------------
    # Environment-aware split
    # ------------------------------------------------------------

    (
        train_rows,
        test_rows,
        train_groups,
        test_groups,
    ) = create_grouped_split(
        groups
    )

    # ------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------

    check_group_leakage(
        train_rows,
        test_rows
    )

    # ------------------------------------------------------------
    # Class distributions
    # ------------------------------------------------------------

    print_class_distribution(
        "Train",
        train_rows
    )

    print_class_distribution(
        "Test",
        test_rows
    )

    # ------------------------------------------------------------
    # Environment distributions
    # ------------------------------------------------------------

    print_environment_distribution(
        "Train",
        train_rows
    )

    print_environment_distribution(
        "Test",
        test_rows
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING PREPARED DATASETS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dataset(
        "X_train.csv",
        train_rows,
    )

    save_dataset(
        "X_test.csv",
        test_rows,
    )

    save_labels(
        "y_train.csv",
        train_rows,
    )

    save_labels(
        "y_test.csv",
        test_rows,
    )

    save_split_metadata(
        train_groups,
        test_groups,
    )

    # ------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("VERIFYING OUTPUT FILES")
    print("=" * 70)

    verify_output(
        "X_train.csv",
        len(train_rows),
    )

    verify_output(
        "X_test.csv",
        len(test_rows),
    )

    verify_output(
        "y_train.csv",
        len(train_rows),
    )

    verify_output(
        "y_test.csv",
        len(test_rows),
    )

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"Training samples: {len(train_rows)}"
    )

    print(
        f"Testing samples : {len(test_rows)}"
    )

    print(
        f"Training groups : {len(train_groups)}"
    )

    print(
        f"Testing groups  : {len(test_groups)}"
    )

    print()
    print(
        "Original trajectory_features.csv "
        "was NOT modified."
    )


if __name__ == "__main__":
    main()