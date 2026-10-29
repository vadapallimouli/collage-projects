"""
Test trajectory feature extraction.

Pipeline:

MπNets problem
    ↓
PyBullet Panda
    ↓
Environment
    ↓
Goal configuration
    ↓
Multiple candidate paths
    ↓
Clearance analysis
    ↓
Feature extraction
"""

import pickle
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment

from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer
from analysis.trajectory_evaluator import TrajectoryEvaluator

from planning.goal_configuration import (
    GoalConfigurationGenerator
)

from planning.trajectory_generator import (
    TrajectoryGenerator
)

from planning.obstacle_aware_planner import (
    ObstacleAwarePlanner
)

from analysis.feature_extractor import (
    TrajectoryFeatureExtractor
)


DATASET_PATH = (
    "data/mpinets/both_solvable_problems.pkl"
)

NUM_CANDIDATES = 10

NUM_POINTS = 200

NUM_SAMPLES = 700

K_NEIGHBORS = 15

EDGE_POINTS = 25

SEED = 42


def main():

    print("=" * 60)
    print("TRAJECTORY FEATURE EXTRACTION TEST")
    print("=" * 60)

    # =========================================================
    # LOAD DATASET
    # =========================================================

    print("\nLoading MπNets dataset...")

    with open(
        DATASET_PATH,
        "rb"
    ) as file:

        dataset = pickle.load(
            file
        )

    print("Dataset loaded.")

    problem = (
        dataset[
            "tabletop"
        ][
            "task_oriented"
        ][0]
    )

    print(
        "Selected tabletop/task_oriented "
        "problem 0."
    )

    # =========================================================
    # PYBULLET
    # =========================================================

    physics_client = p.connect(
        p.DIRECT
    )

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

    panda_robot = PandaRobot(
        robot_id
    )

    print("\nPanda loaded.")

    print(
        "DOF:",
        panda_robot.DOF
    )

    # =========================================================
    # ENVIRONMENT
    # =========================================================

    environment = (
        Environment.from_mpinets_problem(
            problem
        )
    )

    environment.create_pybullet_obstacles(
        physics_client
    )

    print("\nEnvironment created.")

    print(
        "Number of obstacles:",
        environment.num_obstacles
    )

    # =========================================================
    # COLLISION CHECKER
    # =========================================================

    collision_checker = (
        CollisionChecker(
            panda_robot,
            environment,
            physics_client
        )
    )

    # =========================================================
    # CLEARANCE ANALYZER
    # =========================================================

    clearance_analyzer = (
        ClearanceAnalyzer(
            panda_robot,
            environment,
            physics_client
        )
    )

    print(
        "Clearance analyzer created."
    )

    # =========================================================
    # TRAJECTORY GENERATOR
    # =========================================================

    trajectory_generator = (
        TrajectoryGenerator(
            panda_robot
        )
    )

    # =========================================================
    # START CONFIGURATION
    # =========================================================

    start_configuration = problem.q0

    print(
        "\nSTART CONFIGURATION:"
    )

    print(
        start_configuration
    )

    start_result = (
        collision_checker
        .check_configuration(
            start_configuration
        )
    )

    print(
        "Start collision:",
        start_result["collision"]
    )

    # =========================================================
    # GOAL CONFIGURATION
    # =========================================================

    print(
        "\nGenerating goal configuration..."
    )

    goal_generator = (
        GoalConfigurationGenerator(
            panda_robot=panda_robot,
            environment=environment,
            client_id=physics_client
        )
    )

    goal_result = (
        goal_generator.solve()
    )

    if goal_result is None:

        raise RuntimeError(
            "Could not generate goal configuration."
        )

    goal_configuration = (
        goal_result[
            "configuration"
        ]
    )

    print(
        "\nGOAL CONFIGURATION:"
    )

    print(
        goal_configuration
    )

    goal_collision = (
        collision_checker
        .check_configuration(
            goal_configuration
        )
    )

    print(
        "Goal collision:",
        goal_collision["collision"]
    )

    # =========================================================
    # PLANNER
    # =========================================================

    planner = (
        ObstacleAwarePlanner(
            panda_robot=panda_robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=NUM_SAMPLES,
            k_neighbors=K_NEIGHBORS,
            edge_points=EDGE_POINTS,
            seed=SEED
        )
    )

    # =========================================================
    # GENERATE CANDIDATES
    # =========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "GENERATING CANDIDATE TRAJECTORIES"
    )

    print(
        "=" * 60
    )

    candidates = (
        planner.plan_multiple(
            start_configuration,
            goal_configuration,
            num_candidates=NUM_CANDIDATES,
            num_points=NUM_POINTS
        )
    )

    print(
        "\nGenerated candidates:",
        len(candidates)
    )

    if len(candidates) == 0:

        raise RuntimeError(
            "No candidate trajectories were generated."
        )

    # =========================================================
    # FEATURE EXTRACTOR
    # =========================================================

    feature_extractor = (
        TrajectoryFeatureExtractor(
            collision_checker=collision_checker,
            clearance_analyzer=clearance_analyzer,
            trajectory_generator=trajectory_generator
        )
    )

    print(
        "\nFeature extractor created."
    )

    # =========================================================
    # EXTRACT FEATURES
    # =========================================================

    trajectories = [
        candidate[
            "trajectory"
        ]
        for candidate in candidates
    ]

    feature_rows = (
        feature_extractor
        .extract_multiple(
            trajectories,
            include_label=True
        )
    )

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EXTRACTED FEATURES"
    )

    print(
        "=" * 60
    )

    for row in feature_rows:

        print(
            "\nCandidate",
            row[
                "trajectory_index"
            ] + 1
        )

        print(
            "----------------------------------------"
        )

        print(
            "Collision:",
            row["collision"]
        )

        print(
            "Collision count:",
            row["collision_count"]
        )

        print(
            "Obstacle collisions:",
            row[
                "obstacle_collision_count"
            ]
        )

        print(
            "Self collisions:",
            row[
                "self_collision_count"
            ]
        )

        print(
            "Joint-limit violations:",
            row[
                "joint_limit_violation_count"
            ]
        )

        print(
            "Minimum clearance:",
            f"{row['minimum_clearance']:.6f} m"
        )

        print(
            "Closest obstacle index:",
            row[
                "closest_obstacle_index"
            ]
        )

        print(
            "Joint-space length:",
            f"{row['joint_space_length']:.6f}"
        )

        print(
            "Maximum joint step:",
            f"{row['max_joint_step']:.6f}"
        )

        print(
            "Maximum joint velocity:",
            f"{row['max_joint_velocity']:.6f}"
        )

        print(
            "Mean joint velocity:",
            f"{row['mean_joint_velocity']:.6f}"
        )

        print(
            "Maximum joint acceleration:",
            f"{row['max_joint_acceleration']:.6f}"
        )

        print(
            "Mean joint acceleration:",
            f"{row['mean_joint_acceleration']:.6f}"
        )

        print(
            "Number of trajectory points:",
            row["num_points"]
        )

        print(
            "Failure label:",
            row["failure"]
        )

    # =========================================================
    # ML FEATURE VECTOR
    # =========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "ML FEATURE VECTORS"
    )

    print(
        "=" * 60
    )

    feature_names = [
        "minimum_clearance",
        "joint_space_length",
        "max_joint_step",
        "max_joint_velocity",
        "mean_joint_velocity",
        "max_joint_acceleration",
        "mean_joint_acceleration",
        "num_points",
    ]

    print(
        "\nFeature order:"
    )

    for i, name in enumerate(
        feature_names
    ):

        print(
            f"{i}: {name}"
        )

    print(
        "\nFeature matrix:"
    )

    for row in feature_rows:

        vector = [
            row[name]
            for name in feature_names
        ]

        print(
            f"Candidate "
            f"{row['trajectory_index'] + 1}:"
        )

        print(
            vector
        )

    # =========================================================
    # LABEL DISTRIBUTION
    # =========================================================

    labels = [
        row["failure"]
        for row in feature_rows
    ]

    failures = sum(
        labels
    )

    successes = (
        len(labels)
        - failures
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "LABEL SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        "Total trajectories:",
        len(labels)
    )

    print(
        "Successful:",
        successes
    )

    print(
        "Failed:",
        failures
    )

    # =========================================================
    # CLEANUP
    # =========================================================

    p.disconnect(
        physics_client
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE EXTRACTION TEST COMPLETED."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()