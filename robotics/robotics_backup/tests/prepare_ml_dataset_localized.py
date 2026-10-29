from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "failure_prediction"

INPUT_CSV = DATA_DIR / "trajectory_features_localized.csv"

X_TRAIN_OUT = DATA_DIR / "X_train_localized.csv"
X_TEST_OUT = DATA_DIR / "X_test_localized.csv"
Y_TRAIN_OUT = DATA_DIR / "y_train_localized.csv"
Y_TEST_OUT = DATA_DIR / "y_test_localized.csv"
META_OUT = DATA_DIR / "split_metadata_localized.csv"

ML_FEATURE_NAMES = [
    "interior_mean_clearance",
    "interior_clearance_std",
    "interior_maximum_clearance",
    "interior_low_clearance_count",
    "interior_low_clearance_fraction",
    "number_of_low_clearance_runs",
    "longest_low_clearance_run",
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

TEST_GROUPS = {
    "cubby": [2, 7, 8],
    "dresser": [9],
    "merged_cubby": [6, 9],
    "tabletop": [1, 7],
}

def make_group(row):
    return f"{row['environment_type']}__{int(row['problem_index'])}"

def main():
    print("=" * 70)
    print("PREPARING LOCALIZED ML DATASET")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Localized dataset not found:\n{INPUT_CSV}\n"
            "Run tests/build_failure_dataset_localized.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"Input dataset: {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required = ["environment_type", "problem_index", "failure"] + ML_FEATURE_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required columns:\n" + "\n".join(f"  - {x}" for x in missing))

    if len(ML_FEATURE_NAMES) != 17:
        raise RuntimeError(f"Expected 17 ML features, got {len(ML_FEATURE_NAMES)}.")

    X = df[ML_FEATURE_NAMES]
    if X.isnull().any().any():
        raise RuntimeError("Missing values found in ML features.")

    if not X.apply(lambda c: pd.api.types.is_numeric_dtype(c)).all():
        raise RuntimeError("One or more ML features are non-numeric.")

    y = df["failure"].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise RuntimeError(f"Failure label must contain only 0/1. Found: {sorted(y.unique())}")

    df = df.copy()
    df["group"] = df.apply(make_group, axis=1)

    all_groups = sorted(df["group"].unique())
    test_groups = sorted(
        f"{env}__{idx}"
        for env, indices in TEST_GROUPS.items()
        for idx in indices
    )

    missing_groups = [g for g in test_groups if g not in all_groups]
    if missing_groups:
        raise RuntimeError("Expected test groups are missing:\n" + "\n".join(missing_groups))

    train_groups = sorted(set(all_groups) - set(test_groups))

    print(f"Unique planning-problem groups: {len(all_groups)}")
    print(f"Train groups: {len(train_groups)}")
    print(f"Test groups : {len(test_groups)}")

    print("\nTEST GROUPS:")
    for g in test_groups:
        print(f"  {g}")

    train_df = df[df["group"].isin(train_groups)].copy()
    test_df = df[df["group"].isin(test_groups)].copy()

    train_df = train_df.sort_values(
        ["environment_type", "problem_index", "source", "trajectory_index"],
        kind="stable"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["environment_type", "problem_index", "source", "trajectory_index"],
        kind="stable"
    ).reset_index(drop=True)

    if set(train_df["group"]) & set(test_df["group"]):
        raise RuntimeError("GROUP LEAKAGE DETECTED.")

    print("\nGROUP LEAKAGE CHECK: PASS")

    X_train = train_df[ML_FEATURE_NAMES].copy()
    X_test = test_df[ML_FEATURE_NAMES].copy()
    y_train = train_df[["failure"]].copy()
    y_test = test_df[["failure"]].copy()

    def show_dist(name, frame):
        total = len(frame)
        success = int((frame["failure"] == 0).sum())
        failure = int((frame["failure"] == 1).sum())
        print(
            f"{name}: {total} rows | "
            f"{success} success ({success/total:.2%}) | "
            f"{failure} failure ({failure/total:.2%})"
        )

    print("\nCLASS DISTRIBUTION")
    show_dist("Train", train_df)
    show_dist("Test ", test_df)

    print("\nTRAIN ENVIRONMENT DISTRIBUTION")
    for env, g in train_df.groupby("environment_type"):
        failure = int(g["failure"].sum())
        print(f"  {env:15s}: {len(g):3d} total | {len(g)-failure:3d} success | {failure:3d} failure")

    print("\nTEST ENVIRONMENT DISTRIBUTION")
    for env, g in test_df.groupby("environment_type"):
        failure = int(g["failure"].sum())
        print(f"  {env:15s}: {len(g):3d} total | {len(g)-failure:3d} success | {failure:3d} failure")

    X_train.to_csv(X_TRAIN_OUT, index=False)
    X_test.to_csv(X_TEST_OUT, index=False)
    y_train.to_csv(Y_TRAIN_OUT, index=False)
    y_test.to_csv(Y_TEST_OUT, index=False)

    metadata = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True
    )
    metadata[
        ["split", "environment_type", "problem_index", "group",
         "source", "trajectory_index", "failure"]
    ].to_csv(META_OUT, index=False)

    assert X_train.shape[1] == 17
    assert X_test.shape[1] == 17
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    print("\n" + "=" * 70)
    print("FINAL SPLIT SUMMARY")
    print("=" * 70)
    print(f"Train rows  : {len(X_train)}")
    print(f"Test rows   : {len(X_test)}")
    print(f"Train groups: {len(train_groups)}")
    print(f"Test groups : {len(test_groups)}")
    print(f"ML features : {len(ML_FEATURE_NAMES)}")

    print("\nFILES SAVED:")
    for p in [X_TRAIN_OUT, X_TEST_OUT, Y_TRAIN_OUT, Y_TEST_OUT, META_OUT]:
        print(f"  {p}")

    print("\nDataset preparation PASSED.")

if __name__ == "__main__":
    main()
