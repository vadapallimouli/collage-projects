from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FILE PATHS
# ============================================================

RECORDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "trajectory_records.pkl"
)

FALSE_NEGATIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "analysis"
    / "false_negatives.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
    / "analysis"
)


# ============================================================
# SETTINGS
# ============================================================

# Same clearance threshold already used by the
# ClearanceAnalyzer for "low clearance" analysis.
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


def load_false_negatives():

    print("\n" + "=" * 70)
    print("LOADING FALSE NEGATIVES")
    print("=" * 70)

    df = pd.read_csv(FALSE_NEGATIVE_PATH)

    print(f"False negatives loaded: {len(df)}")

    return df


# ============================================================
# RUN DETECTION
# ============================================================

def find_runs(mask):
    """
    Find contiguous True regions in a boolean mask.

    Returns a list of dictionaries containing:
        start_index
        end_index
        length
    """

    runs = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        elif not value and start is not None:

            end = i - 1

            runs.append({
                "start_index": start,
                "end_index": end,
                "length": end - start + 1,
            })

            start = None

    # Handle a run that reaches the end
    if start is not None:

        end = len(mask) - 1

        runs.append({
            "start_index": start,
            "end_index": end,
            "length": end - start + 1,
        })

    return runs


# ============================================================
# ANALYZE ONE TRAJECTORY
# ============================================================

def analyze_trajectory(record_index, record):

    trajectory = np.asarray(record["_trajectory"])

    num_points = len(trajectory)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The trajectory_records file does not contain the complete
    # per-configuration clearance sequence.
    #
    # Therefore, the clearance sequence must be reconstructed
    # using the project's existing ClearanceAnalyzer.
    # --------------------------------------------------------

    from robot.panda_robot import PandaRobot
    from environment.environment import Environment
    from analysis.clearance_analyzer import ClearanceAnalyzer
    import pybullet as p
    import pybullet_data


    # --------------------------------------------------------
    # Reconstruct the original environment
    # --------------------------------------------------------

    environment_type = record["environment_type"]
    subset = record["subset"]
    problem_index = int(record["problem_index"])

    # Load MπNets dataset
    mpinets_path = (
        PROJECT_ROOT
        / "data"
        / "mpinets"
        / "both_solvable_problems.pkl"
    )

    with open(mpinets_path, "rb") as f:
        dataset = pickle.load(f)

    problem = dataset[environment_type][subset][problem_index]

    # --------------------------------------------------------
    # Create PyBullet world
    # --------------------------------------------------------

    client_id = p.connect(p.DIRECT)

    try:

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.loadURDF(
            "plane.urdf"
        )

        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True
        )

        robot = PandaRobot(robot_id)

        environment = Environment.from_mpinets_problem(
            problem
        )

        obstacle_ids = environment.create_pybullet_obstacles(
            client_id
        )

        analyzer = ClearanceAnalyzer(
            panda_robot=robot,
            environment=environment,
            client_id=client_id
        )

        # ----------------------------------------------------
        # Calculate clearance for EVERY trajectory point
        # ----------------------------------------------------

        clearances = []

        for q in trajectory:

            result = analyzer.calculate_configuration_clearance(
                q
            )

            clearances.append(
                float(result["minimum_clearance"])
            )

        clearances = np.asarray(
            clearances,
            dtype=float
        )

        # ----------------------------------------------------
        # Collision information
        # ----------------------------------------------------

        collision_mask = clearances < 0.0

        low_clearance_mask = (
            clearances <= LOW_CLEARANCE_THRESHOLD
        )

        collision_indices = np.where(
            collision_mask
        )[0]

        low_clearance_indices = np.where(
            low_clearance_mask
        )[0]

        # ----------------------------------------------------
        # Contiguous low-clearance runs
        # ----------------------------------------------------

        low_clearance_runs = find_runs(
            low_clearance_mask
        )

        collision_runs = find_runs(
            collision_mask
        )

        # ----------------------------------------------------
        # Run statistics
        # ----------------------------------------------------

        if low_clearance_runs:

            longest_low_clearance_run = max(
                run["length"]
                for run in low_clearance_runs
            )

        else:

            longest_low_clearance_run = 0

        if collision_runs:

            longest_collision_run = max(
                run["length"]
                for run in collision_runs
            )

        else:

            longest_collision_run = 0

        # ----------------------------------------------------
        # Check whether collisions occur inside
        # low-clearance regions
        # ----------------------------------------------------

        collision_inside_low_clearance = 0

        for index in collision_indices:

            if low_clearance_mask[index]:
                collision_inside_low_clearance += 1

        if len(collision_indices) > 0:

            collision_low_clearance_overlap = (
                collision_inside_low_clearance
                / len(collision_indices)
            )

        else:

            collision_low_clearance_overlap = 0.0

        # ----------------------------------------------------
        # Distance from collision to nearest low-clearance
        # region
        # ----------------------------------------------------

        distances_to_low_clearance = []

        if len(low_clearance_indices) > 0:

            for collision_index in collision_indices:

                distance = np.min(
                    np.abs(
                        low_clearance_indices
                        - collision_index
                    )
                )

                distances_to_low_clearance.append(
                    int(distance)
                )

        if distances_to_low_clearance:

            minimum_collision_to_low_clearance_distance = min(
                distances_to_low_clearance
            )

        else:

            minimum_collision_to_low_clearance_distance = -1

        # ----------------------------------------------------
        # Basic sequence statistics
        # ----------------------------------------------------

        result = {

            "record_index": record_index,

            "environment_type": environment_type,
            "source": record["source"],
            "subset": subset,
            "problem_index": problem_index,

            "num_points": num_points,

            "actual_collision_count": int(
                np.sum(collision_mask)
            ),

            "stored_collision_count": int(
                record["collision_count"]
            ),

            "collision_fraction": float(
                np.mean(collision_mask)
            ),

            "low_clearance_count": int(
                np.sum(low_clearance_mask)
            ),

            "low_clearance_fraction": float(
                np.mean(low_clearance_mask)
            ),

            "number_of_low_clearance_runs": len(
                low_clearance_runs
            ),

            "longest_low_clearance_run": (
                longest_low_clearance_run
            ),

            "number_of_collision_runs": len(
                collision_runs
            ),

            "longest_collision_run": (
                longest_collision_run
            ),

            "collision_low_clearance_overlap": (
                collision_low_clearance_overlap
            ),

            "minimum_collision_to_low_clearance_distance": (
                minimum_collision_to_low_clearance_distance
            ),

            "minimum_clearance": float(
                np.min(clearances)
            ),

            "mean_clearance": float(
                np.mean(clearances)
            ),

            "clearance_std": float(
                np.std(clearances)
            ),

            "maximum_clearance": float(
                np.max(clearances)
            ),

            "collision_indices": (
                collision_indices.tolist()
            ),

            "low_clearance_indices": (
                low_clearance_indices.tolist()
            ),

            "low_clearance_runs": (
                low_clearance_runs
            ),

            "collision_runs": (
                collision_runs
            ),
        }

        return result, clearances

    finally:

        p.disconnect(client_id)


# ============================================================
# ANALYZE ALL FALSE NEGATIVES
# ============================================================

def analyze_all(records, false_negative_df):

    print("\n" + "=" * 70)
    print("ANALYZING CLEARANCE SEQUENCES")
    print("=" * 70)

    results = []

    clearance_sequences = {}

    for _, row in false_negative_df.iterrows():

        record_index = int(
            row["record_index"]
        )

        print(
            f"\nAnalyzing false negative "
            f"{record_index}..."
        )

        record = records[record_index]

        result, clearances = analyze_trajectory(
            record_index,
            record
        )

        results.append(result)

        clearance_sequences[record_index] = (
            clearances
        )

    return (
        pd.DataFrame(results),
        clearance_sequences
    )


# ============================================================
# PRINT INDIVIDUAL RESULTS
# ============================================================

def print_individual_results(results_df):

    print("\n" + "=" * 70)
    print("LOCALIZATION RESULTS")
    print("=" * 70)

    columns = [
        "record_index",
        "environment_type",
        "source",
        "actual_collision_count",
        "collision_fraction",
        "number_of_collision_runs",
        "longest_collision_run",
        "low_clearance_fraction",
        "number_of_low_clearance_runs",
        "longest_low_clearance_run",
        "collision_low_clearance_overlap",
        "minimum_collision_to_low_clearance_distance",
    ]

    print(
        results_df[columns].to_string(
            index=False
        )
    )


# ============================================================
# PRINT RUN DETAILS
# ============================================================

def print_run_details(results_df):

    print("\n" + "=" * 70)
    print("LOW-CLEARANCE RUN DETAILS")
    print("=" * 70)

    for _, row in results_df.iterrows():

        print(
            f"\nRecord {int(row['record_index'])}:"
        )

        print(
            "  Low-clearance runs:"
        )

        print(
            f"    {row['number_of_low_clearance_runs']}"
        )

        print(
            "  Longest low-clearance run:"
        )

        print(
            f"    {row['longest_low_clearance_run']}"
        )

        print(
            "  Collision runs:"
        )

        print(
            f"    {row['number_of_collision_runs']}"
        )

        print(
            "  Longest collision run:"
        )

        print(
            f"    {row['longest_collision_run']}"
        )


# ============================================================
# OVERALL PATTERN ANALYSIS
# ============================================================

def analyze_overall_pattern(results_df):

    print("\n" + "=" * 70)
    print("OVERALL LOCALIZATION PATTERN")
    print("=" * 70)

    print(
        f"\nNumber of false negatives: "
        f"{len(results_df)}"
    )

    print(
        f"\nAverage collision fraction: "
        f"{results_df['collision_fraction'].mean():.4f}"
    )

    print(
        f"Average number of collision runs: "
        f"{results_df['number_of_collision_runs'].mean():.2f}"
    )

    print(
        f"Average longest collision run: "
        f"{results_df['longest_collision_run'].mean():.2f}"
    )

    print(
        f"\nAverage low-clearance fraction: "
        f"{results_df['low_clearance_fraction'].mean():.4f}"
    )

    print(
        f"Average number of low-clearance runs: "
        f"{results_df['number_of_low_clearance_runs'].mean():.2f}"
    )

    print(
        f"Average longest low-clearance run: "
        f"{results_df['longest_low_clearance_run'].mean():.2f}"
    )

    print(
        f"\nAverage collision/low-clearance overlap: "
        f"{results_df['collision_low_clearance_overlap'].mean():.4f}"
    )

    # --------------------------------------------------------
    # How many false negatives have a localized collision?
    # --------------------------------------------------------

    localized = results_df[
        results_df["collision_fraction"] <= 0.05
    ]

    print(
        "\nFalse negatives with collision fraction "
        f"<= 5%: {len(localized)}/{len(results_df)}"
    )

    localized = results_df[
        results_df["collision_fraction"] <= 0.10
    ]

    print(
        "False negatives with collision fraction "
        f"<= 10%: {len(localized)}/{len(results_df)}"
    )

    # --------------------------------------------------------
    # How many have a collision inside a low-clearance run?
    # --------------------------------------------------------

    overlap = results_df[
        results_df["collision_low_clearance_overlap"] > 0
    ]

    print(
        "\nFalse negatives where collision points overlap "
        f"low-clearance regions: "
        f"{len(overlap)}/{len(results_df)}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    clearance_sequences
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_path = (
        OUTPUT_DIR
        / "false_negative_localization.csv"
    )

    sequences_path = (
        OUTPUT_DIR
        / "false_negative_clearance_sequences.pkl"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    with open(
        sequences_path,
        "wb"
    ) as f:

        pickle.dump(
            clearance_sequences,
            f
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(results_path)
    print(sequences_path)


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_records()

    false_negative_df = load_false_negatives()

    results_df, clearance_sequences = analyze_all(
        records,
        false_negative_df
    )

    print_individual_results(
        results_df
    )

    print_run_details(
        results_df
    )

    analyze_overall_pattern(
        results_df
    )

    save_results(
        results_df,
        clearance_sequences
    )

    print("\n" + "=" * 70)
    print("LOCALIZATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()