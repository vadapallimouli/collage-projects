"""
Balanced Failure-Prediction Dataset Generator
==============================================

This script generates a multi-problem dataset for ML-based robot
trajectory failure prediction.

Ground truth:
    PyBullet collision checking ONLY.

The generator:
    1. Loads multiple MπNets environments.
    2. Selects multiple problems from each environment.
    3. Finds a collision-free start and goal.
    4. Generates a large pool of candidate trajectories.
    5. Evaluates every candidate using PyBullet.
    6. Selects a more balanced set of successful and failed trajectories.
    7. Saves:
         - CSV containing ML features and labels
         - PKL containing complete trajectories

IMPORTANT:
The script does NOT manually assign success/failure labels.

A trajectory is:
    failure = 1
if and only if the collision checker reports failure.

This is a data-sampling strategy, not a hand-designed failure rule.
"""

import csv
import pickle
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import pybullet_data


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# PROJECT IMPORTS
# ================================================================

from environment.environment import Environment

from robot.panda_robot import PandaRobot

from planning.trajectory_generator import (
    TrajectoryGenerator,
)

from planning.goal_configuration import (
    GoalConfigurationGenerator,
)

from planning.obstacle_aware_planner import (
    ObstacleAwarePlanner,
)

from analysis.collision_checker import (
    CollisionChecker,
)

from analysis.clearance_analyzer import (
    ClearanceAnalyzer,
)

from analysis.feature_extractor import (
    TrajectoryFeatureExtractor,
)


# ================================================================
# DATASET CONFIGURATION
# ================================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "failure_prediction"
)


CSV_PATH = (
    OUTPUT_DIR
    / "trajectory_features.csv"
)


PKL_PATH = (
    OUTPUT_DIR
    / "trajectory_records.pkl"
)


# ------------------------------------------------
# Problems
# ------------------------------------------------

# We use more problems because some scenes do not
# provide a collision-free IK goal with our current
# IK strategy.

PROBLEMS_PER_ENVIRONMENT = 15


# ------------------------------------------------
# Final number of trajectories retained per problem
# ------------------------------------------------

MAX_SUCCESS_PER_PROBLEM = 12
MAX_FAILURE_PER_PROBLEM = 12


# ------------------------------------------------
# Candidate generation
# ------------------------------------------------

NUM_PLANNED_CANDIDATES = 5

NUM_PERTURBED_CANDIDATES = 25

NUM_WAYPOINT_CANDIDATES = 15


# ------------------------------------------------
# Trajectory resolution
# ------------------------------------------------

NUM_POINTS = 200


# ------------------------------------------------
# Planner
# ------------------------------------------------

PLANNER_SAMPLES = 350

PLANNER_K_NEIGHBORS = 12

PLANNER_EDGE_POINTS = 20


# ------------------------------------------------
# Random seed
# ------------------------------------------------

RANDOM_SEED = 42


# ================================================================
# LOAD MπNETS DATASET
# ================================================================

def load_dataset():

    print()
    print("=" * 70)
    print("LOADING MπNets DATASET")
    print("=" * 70)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"MπNets dataset not found:\n"
            f"{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "rb",
    ) as f:

        dataset = pickle.load(f)

    print(
        "Available environments:",
        list(dataset.keys()),
    )

    return dataset


# ================================================================
# RESET PYBULLET
# ================================================================

def reset_pybullet(client_id):

    p.resetSimulation(
        physicsClientId=client_id
    )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath(),
        physicsClientId=client_id,
    )

    # Ground
    p.loadURDF(
        "plane.urdf",
        physicsClientId=client_id,
    )

    # Panda
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
        physicsClientId=client_id,
    )

    return robot_id


# ================================================================
# CREATE PROBLEM CONTEXT
# ================================================================

def create_problem_context(
    problem,
    client_id,
):

    robot_id = reset_pybullet(
        client_id
    )

    panda_robot = PandaRobot(
        robot_id
    )

    environment = (
        Environment.from_mpinets_problem(
            problem
        )
    )

    environment.create_pybullet_obstacles(
        client_id
    )

    trajectory_generator = (
        TrajectoryGenerator(
            panda_robot
        )
    )

    collision_checker = (
        CollisionChecker(
            panda_robot,
            environment,
            client_id,
        )
    )

    clearance_analyzer = (
        ClearanceAnalyzer(
            panda_robot,
            environment,
            client_id,
        )
    )

    feature_extractor = (
        TrajectoryFeatureExtractor(
            collision_checker,
            clearance_analyzer,
            trajectory_generator,
        )
    )

    return (
        panda_robot,
        environment,
        trajectory_generator,
        collision_checker,
        clearance_analyzer,
        feature_extractor,
    )


# ================================================================
# FIND COLLISION-FREE GOAL
# ================================================================

def find_collision_free_goal(
    panda_robot,
    environment,
    collision_checker,
    client_id,
    seed,
):

    goal_generator = (
        GoalConfigurationGenerator(
            panda_robot=panda_robot,
            environment=environment,
            client_id=client_id,
        )
    )

    solutions = (
        goal_generator.solve_multiple(
            num_solutions=30,
            num_rest_poses=30,
            seed=seed,
        )
    )

    for solution in solutions:

        q_goal = np.asarray(
            solution["configuration"],
            dtype=float,
        )

        result = (
            collision_checker
            .check_configuration(
                q_goal
            )
        )

        if not result["collision"]:

            return q_goal

    return None


# ================================================================
# PERTURBED TRAJECTORY
# ================================================================

def generate_perturbed_trajectory(
    trajectory_generator,
    start,
    goal,
    rng,
    scale,
):

    trajectory = (
        trajectory_generator
        .generate_joint_trajectory(
            start,
            goal,
            num_points=NUM_POINTS,
        )
    ).copy()

    n = len(trajectory)

    # Perturb only the middle portion.
    start_index = int(
        0.20 * n
    )

    end_index = int(
        0.80 * n
    )

    noise = rng.normal(
        loc=0.0,
        scale=scale,
        size=(
            end_index - start_index,
            7,
        ),
    )

    trajectory[
        start_index:end_index
    ] += noise

    # Smooth the perturbation.
    kernel_size = 9

    kernel = (
        np.ones(kernel_size)
        / kernel_size
    )

    for joint in range(7):

        segment = trajectory[
            start_index:end_index,
            joint,
        ]

        smoothed = np.convolve(
            segment,
            kernel,
            mode="same",
        )

        trajectory[
            start_index:end_index,
            joint,
        ] = smoothed

    # Enforce joint limits.
    trajectory = np.clip(
        trajectory,
        trajectory_generator.lower_limits,
        trajectory_generator.upper_limits,
    )

    # Preserve exact endpoints.
    trajectory[0] = start
    trajectory[-1] = goal

    return trajectory


# ================================================================
# WAYPOINT TRAJECTORY
# ================================================================

def generate_waypoint_trajectory(
    trajectory_generator,
    start,
    goal,
    rng,
    scale,
):

    midpoint = (
        0.5 * (start + goal)
    )

    waypoint = (
        midpoint
        + rng.normal(
            loc=0.0,
            scale=scale,
            size=7,
        )
    )

    waypoint = np.clip(
        waypoint,
        trajectory_generator.lower_limits,
        trajectory_generator.upper_limits,
    )

    trajectory = (
        trajectory_generator
        .generate_waypoint_trajectory(
            start,
            [waypoint],
            goal,
            num_points=NUM_POINTS,
        )
    )

    return trajectory


# ================================================================
# TRAJECTORY SIGNATURE
# ================================================================

def trajectory_signature(
    trajectory
):
    """
    Create a compact signature so that duplicate trajectories
    can be avoided.

    We round the trajectory before converting it to bytes.
    """

    trajectory = np.asarray(
        trajectory,
        dtype=np.float32,
    )

    rounded = np.round(
        trajectory,
        decimals=5,
    )

    return rounded.tobytes()


# ================================================================
# EVALUATE TRAJECTORY
# ================================================================

def evaluate_candidate(
    trajectory,
    feature_extractor,
):

    features = (
        feature_extractor.extract(
            trajectory,
            include_label=True,
        )
    )

    return features


# ================================================================
# ADD CANDIDATE
# ================================================================

def add_candidate(
    candidates,
    signatures,
    trajectory,
    source,
    feature_extractor,
):

    signature = trajectory_signature(
        trajectory
    )

    # Prevent duplicate trajectories.
    if signature in signatures:
        return False

    signatures.add(
        signature
    )

    features = evaluate_candidate(
        trajectory,
        feature_extractor,
    )

    candidate = {

        "trajectory": np.asarray(
            trajectory,
            dtype=np.float32,
        ),

        "source": source,

        "features": features,
    }

    candidates.append(
        candidate
    )

    return True


# ================================================================
# SELECT BALANCED CANDIDATES
# ================================================================

def select_balanced_candidates(
    candidates,
    rng,
):
    """
    Select a balanced subset from the generated candidate pool.

    Labels are already determined by PyBullet.

    This function does NOT create labels.

    It only samples from the already-labelled candidate pool.
    """

    successful = [
        candidate
        for candidate in candidates
        if candidate["features"]["failure"] == 0
    ]

    failed = [
        candidate
        for candidate in candidates
        if candidate["features"]["failure"] == 1
    ]

    # Shuffle within each class.
    rng.shuffle(
        successful
    )

    rng.shuffle(
        failed
    )

    selected = []

    # ------------------------------------------------------------
    # Prefer source diversity.
    #
    # We first attempt to select candidates from each source.
    # ------------------------------------------------------------

    def select_diverse(
        pool,
        maximum,
    ):

        if not pool:
            return []

        selected_local = []

        sources = sorted(
            set(
                candidate["source"]
                for candidate in pool
            )
        )

        # Round-robin through sources.
        while (
            pool
            and len(selected_local)
            < maximum
        ):

            progress = False

            for source in sources:

                if len(selected_local) >= maximum:
                    break

                for index, candidate in enumerate(pool):

                    if candidate[
                        "source"
                    ] == source:

                        selected_local.append(
                            candidate
                        )

                        pool.pop(
                            index
                        )

                        progress = True

                        break

            if not progress:
                break

        return selected_local

    successful_pool = list(
        successful
    )

    failed_pool = list(
        failed
    )

    selected_success = select_diverse(
        successful_pool,
        MAX_SUCCESS_PER_PROBLEM,
    )

    selected_failure = select_diverse(
        failed_pool,
        MAX_FAILURE_PER_PROBLEM,
    )

    selected.extend(
        selected_success
    )

    selected.extend(
        selected_failure
    )

    rng.shuffle(
        selected
    )

    return (
        selected,
        len(successful),
        len(failed),
    )


# ================================================================
# ADD FINAL RECORD
# ================================================================

def make_final_record(
    candidate,
    environment_type,
    subset,
    problem_index,
):

    record = dict(
        candidate["features"]
    )

    record["environment_type"] = (
        environment_type
    )

    record["subset"] = (
        subset
    )

    record["problem_index"] = (
        int(problem_index)
    )

    record["source"] = (
        candidate["source"]
    )

    record["_trajectory"] = (
        candidate["trajectory"]
    )

    return record


# ================================================================
# BUILD DATASET
# ================================================================

def build_dataset():

    dataset = load_dataset()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    client_id = p.connect(
        p.DIRECT
    )

    if client_id < 0:

        raise RuntimeError(
            "Could not connect to PyBullet."
        )

    all_records = []

    skipped_problems = 0

    problem_counter = 0

    try:

        environment_names = [
            "tabletop",
            "cubby",
            "merged_cubby",
            "dresser",
        ]

        for environment_type in (
            environment_names
        ):

            if environment_type not in dataset:
                continue

            if (
                "task_oriented"
                not in dataset[
                    environment_type
                ]
            ):
                continue

            problems = dataset[
                environment_type
            ]["task_oriented"]

            number_of_problems = min(
                PROBLEMS_PER_ENVIRONMENT,
                len(problems),
            )

            print()
            print(
                "=" * 70
            )

            print(
                f"ENVIRONMENT: "
                f"{environment_type}"
            )

            print(
                f"Problems selected: "
                f"{number_of_problems}"
            )

            print(
                "=" * 70
            )

            for local_index in range(
                number_of_problems
            ):

                problem_counter += 1

                print()
                print(
                    f"[{problem_counter}] "
                    f"{environment_type} "
                    f"/ problem "
                    f"{local_index}"
                )

                try:

                    problem = problems[
                        local_index
                    ]

                    (
                        panda_robot,
                        environment,
                        trajectory_generator,
                        collision_checker,
                        clearance_analyzer,
                        feature_extractor,
                    ) = create_problem_context(
                        problem,
                        client_id,
                    )

                    start = np.asarray(
                        problem.q0,
                        dtype=float,
                    )

                    # ==================================================
                    # START CHECK
                    # ==================================================

                    start_result = (
                        collision_checker
                        .check_configuration(
                            start
                        )
                    )

                    if start_result[
                        "collision"
                    ]:

                        print(
                            "  SKIP: "
                            "start configuration "
                            "is in collision."
                        )

                        skipped_problems += 1

                        continue

                    # ==================================================
                    # GOAL
                    # ==================================================

                    goal = (
                        find_collision_free_goal(
                            panda_robot,
                            environment,
                            collision_checker,
                            client_id,
                            RANDOM_SEED
                            + problem_counter,
                        )
                    )

                    if goal is None:

                        print(
                            "  SKIP: "
                            "no collision-free "
                            "goal found."
                        )

                        skipped_problems += 1

                        continue

                    print(
                        "  Start: collision-free"
                    )

                    print(
                        "  Goal: collision-free"
                    )

                    # ==================================================
                    # CANDIDATE POOL
                    # ==================================================

                    candidates = []

                    signatures = set()

                    # ==================================================
                    # 1. DIRECT
                    # ==================================================

                    direct = (
                        trajectory_generator
                        .generate_joint_trajectory(
                            start,
                            goal,
                            num_points=NUM_POINTS,
                        )
                    )

                    add_candidate(
                        candidates,
                        signatures,
                        direct,
                        "direct",
                        feature_extractor,
                    )

                    # ==================================================
                    # 2. PLANNED
                    # ==================================================

                    print(
                        "  Generating planned "
                        "trajectories..."
                    )

                    planner = (
                        ObstacleAwarePlanner(
                            panda_robot=panda_robot,
                            collision_checker=collision_checker,
                            trajectory_generator=trajectory_generator,
                            num_samples=PLANNER_SAMPLES,
                            k_neighbors=PLANNER_K_NEIGHBORS,
                            edge_points=PLANNER_EDGE_POINTS,
                            seed=(
                                RANDOM_SEED
                                + problem_counter
                            ),
                        )
                    )

                    try:

                        planned_results = (
                            planner.plan_multiple(
                                start,
                                goal,
                                num_candidates=(
                                    NUM_PLANNED_CANDIDATES
                                ),
                            )
                        )

                    except Exception as exc:

                        print(
                            "  Planner error:",
                            type(exc).__name__,
                            str(exc),
                        )

                        planned_results = []

                    for result in (
                        planned_results
                    ):

                        trajectory = (
                            result.get(
                                "trajectory"
                            )
                        )

                        if trajectory is None:
                            continue

                        trajectory = np.asarray(
                            trajectory,
                            dtype=float,
                        )

                        if (
                            trajectory.ndim != 2
                            or trajectory.shape[1] != 7
                        ):
                            continue

                        add_candidate(
                            candidates,
                            signatures,
                            trajectory,
                            "planned",
                            feature_extractor,
                        )

                    # ==================================================
                    # 3. PERTURBATIONS
                    # ==================================================

                    print(
                        "  Generating "
                        "perturbed trajectories..."
                    )

                    # Small → medium → large.
                    #
                    # The purpose is to create a continuum
                    # of difficulty rather than making almost
                    # every perturbed trajectory fail.

                    perturbation_scales = (
                        np.geomspace(
                            0.01,
                            0.45,
                            NUM_PERTURBED_CANDIDATES,
                        )
                    )

                    rng.shuffle(
                        perturbation_scales
                    )

                    for scale in (
                        perturbation_scales
                    ):

                        trajectory = (
                            generate_perturbed_trajectory(
                                trajectory_generator,
                                start,
                                goal,
                                rng,
                                float(scale),
                            )
                        )

                        add_candidate(
                            candidates,
                            signatures,
                            trajectory,
                            "perturbed",
                            feature_extractor,
                        )

                    # ==================================================
                    # 4. WAYPOINTS
                    # ==================================================

                    print(
                        "  Generating "
                        "waypoint trajectories..."
                    )

                    waypoint_scales = (
                        np.geomspace(
                            0.01,
                            0.50,
                            NUM_WAYPOINT_CANDIDATES,
                        )
                    )

                    rng.shuffle(
                        waypoint_scales
                    )

                    for scale in (
                        waypoint_scales
                    ):

                        trajectory = (
                            generate_waypoint_trajectory(
                                trajectory_generator,
                                start,
                                goal,
                                rng,
                                float(scale),
                            )
                        )

                        add_candidate(
                            candidates,
                            signatures,
                            trajectory,
                            "waypoint",
                            feature_extractor,
                        )

                    # ==================================================
                    # CANDIDATE POOL STATISTICS
                    # ==================================================

                    pool_successes = sum(
                        1
                        for candidate
                        in candidates
                        if candidate[
                            "features"
                        ]["failure"] == 0
                    )

                    pool_failures = (
                        len(candidates)
                        - pool_successes
                    )

                    print(
                        f"  Candidate pool: "
                        f"{len(candidates)}"
                    )

                    print(
                        f"  Pool success: "
                        f"{pool_successes}"
                    )

                    print(
                        f"  Pool failure: "
                        f"{pool_failures}"
                    )

                    # ==================================================
                    # BALANCED SELECTION
                    # ==================================================

                    (
                        selected,
                        available_successes,
                        available_failures,
                    ) = (
                        select_balanced_candidates(
                            candidates,
                            rng,
                        )
                    )

                    selected_successes = sum(
                        1
                        for candidate
                        in selected
                        if candidate[
                            "features"
                        ]["failure"] == 0
                    )

                    selected_failures = (
                        len(selected)
                        - selected_successes
                    )

                    # ==================================================
                    # FINAL RECORDS
                    # ==================================================

                    for candidate in selected:

                        record = (
                            make_final_record(
                                candidate,
                                environment_type,
                                "task_oriented",
                                local_index,
                            )
                        )

                        all_records.append(
                            record
                        )

                    print(
                        f"  Available: "
                        f"{available_successes} "
                        f"success / "
                        f"{available_failures} "
                        f"failure"
                    )

                    print(
                        f"  Selected: "
                        f"{selected_successes} "
                        f"success / "
                        f"{selected_failures} "
                        f"failure"
                    )

                except Exception as exc:

                    skipped_problems += 1

                    print(
                        "  ERROR:",
                        type(exc).__name__,
                        str(exc),
                    )

        # ============================================================
        # CHECK DATASET
        # ============================================================

        if not all_records:

            raise RuntimeError(
                "No records were generated."
            )

        # ============================================================
        # TRAJECTORY INDEX
        # ============================================================

        for index, record in enumerate(
            all_records
        ):

            record[
                "trajectory_index"
            ] = index

        # ============================================================
        # ML FEATURES
        # ============================================================

        feature_names = [
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

        # ============================================================
        # CSV COLUMNS
        # ============================================================

        metadata_fields = [
            "environment_type",
            "subset",
            "problem_index",
            "source",
            "trajectory_index",
        ]

        label_fields = [
            "failure"
        ]

        diagnostic_fields = [
            "collision",
            "collision_free",
            "collision_count",
            "self_collision_count",
            "obstacle_collision_count",
            "joint_limit_violation_count",
            "minimum_clearance",
            "closest_obstacle_index",
            "closest_robot_link",
            "num_points",
        ]

        csv_fields = (
            metadata_fields
            + feature_names
            + label_fields
            + diagnostic_fields
        )

        # ============================================================
        # SAVE CSV
        # ============================================================

        print()
        print(
            "=" * 70
        )

        print(
            "SAVING CSV"
        )

        print(
            "=" * 70
        )

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=csv_fields,
            )

            writer.writeheader()

            for record in all_records:

                row = {}

                for field in csv_fields:

                    value = record.get(
                        field,
                        "",
                    )

                    if isinstance(
                        value,
                        np.generic,
                    ):

                        value = value.item()

                    if value is None:

                        value = ""

                    row[field] = value

                writer.writerow(
                    row
                )

        # ============================================================
        # SAVE TRAJECTORIES
        # ============================================================

        print(
            "Saving trajectory archive..."
        )

        with open(
            PKL_PATH,
            "wb",
        ) as f:

            pickle.dump(
                all_records,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # ============================================================
        # FINAL STATISTICS
        # ============================================================

        total = len(
            all_records
        )

        failures = sum(
            int(record["failure"])
            for record in all_records
        )

        successes = (
            total - failures
        )

        print()
        print(
            "=" * 70
        )

        print(
            "FINAL DATASET SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            f"Total trajectories : "
            f"{total}"
        )

        print(
            f"Successful         : "
            f"{successes}"
        )

        print(
            f"Failed             : "
            f"{failures}"
        )

        print(
            f"Success ratio      : "
            f"{successes / total:.3f}"
        )

        print(
            f"Failure ratio      : "
            f"{failures / total:.3f}"
        )

        print(
            f"Skipped problems   : "
            f"{skipped_problems}"
        )

        print()
        print(
            f"CSV saved to:\n"
            f"{CSV_PATH}"
        )

        print()
        print(
            f"PKL saved to:\n"
            f"{PKL_PATH}"
        )

        # ============================================================
        # SOURCE BREAKDOWN
        # ============================================================

        print()
        print(
            "-" * 70
        )

        print(
            "BREAKDOWN BY SOURCE"
        )

        print(
            "-" * 70
        )

        sources = sorted(
            set(
                record["source"]
                for record in all_records
            )
        )

        for source in sources:

            source_records = [
                record
                for record in all_records
                if record["source"]
                == source
            ]

            source_failures = sum(
                int(record["failure"])
                for record in source_records
            )

            source_successes = (
                len(source_records)
                - source_failures
            )

            print(
                f"{source:12s}: "
                f"{len(source_records):4d} total | "
                f"{source_successes:4d} success | "
                f"{source_failures:4d} failure"
            )

        # ============================================================
        # ENVIRONMENT BREAKDOWN
        # ============================================================

        print()
        print(
            "-" * 70
        )

        print(
            "BREAKDOWN BY ENVIRONMENT"
        )

        print(
            "-" * 70
        )

        environments = sorted(
            set(
                record[
                    "environment_type"
                ]
                for record in all_records
            )
        )

        for environment in environments:

            environment_records = [
                record
                for record in all_records
                if record[
                    "environment_type"
                ] == environment
            ]

            environment_failures = sum(
                int(record["failure"])
                for record in environment_records
            )

            environment_successes = (
                len(environment_records)
                - environment_failures
            )

            print(
                f"{environment:16s}: "
                f"{len(environment_records):4d} total | "
                f"{environment_successes:4d} success | "
                f"{environment_failures:4d} failure"
            )

        # ============================================================
        # PROBLEM-LEVEL BALANCE
        # ============================================================

        print()
        print(
            "-" * 70
        )

        print(
            "PROBLEM-LEVEL BALANCE"
        )

        print(
            "-" * 70
        )

        problem_keys = sorted(
            set(
                (
                    record[
                        "environment_type"
                    ],
                    record[
                        "problem_index"
                    ],
                )

                for record in all_records
            )
        )

        for environment, problem_index in (
            problem_keys
        ):

            problem_records = [
                record
                for record in all_records
                if (
                    record[
                        "environment_type"
                    ]
                    == environment
                    and
                    record[
                        "problem_index"
                    ]
                    == problem_index
                )
            ]

            problem_failures = sum(
                int(record["failure"])
                for record in problem_records
            )

            problem_successes = (
                len(problem_records)
                - problem_failures
            )

            print(
                f"{environment:15s} "
                f"{problem_index:2d}: "
                f"{problem_successes:2d} success / "
                f"{problem_failures:2d} failure"
            )

    finally:

        p.disconnect(
            client_id
        )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    build_dataset()