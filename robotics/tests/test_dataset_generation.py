# tests/test_dataset_generation.py

import pickle
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment

from planning.goal_configuration import (
    GoalConfigurationGenerator
)

from planning.trajectory_generator import (
    TrajectoryGenerator
)

from planning.obstacle_aware_planner import (
    ObstacleAwarePlanner
)

from analysis.collision_checker import (
    CollisionChecker
)

from analysis.clearance_analyzer import (
    ClearanceAnalyzer
)

from analysis.feature_extractor import (
    TrajectoryFeatureExtractor
)


# ================================================================
# PATHS
# ================================================================

DATASET_PATH = Path(
    r"C:\Users\sharm\OneDrive\Desktop\s3\robotics\data\mpinets\both_solvable_problems.pkl"
)


# ================================================================
# TEST CONFIGURATION
# ================================================================

NUM_PLANNED = 5
NUM_DIRECT = 5
NUM_PERTURBED = 10

NUM_POINTS = 200

RANDOM_SEED = 42


# ================================================================
# LOAD MπNets DATASET
# ================================================================

def load_mpinets_dataset():

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    print("Dataset loaded.")

    return dataset


# ================================================================
# PYBULLET SETUP
# ================================================================

def setup_pybullet():

    physics_client = p.connect(
        p.DIRECT
    )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81
    )

    # ------------------------------------------------------------
    # Ground
    # ------------------------------------------------------------

    p.loadURDF(
        "plane.urdf"
    )

    # ------------------------------------------------------------
    # Panda
    # ------------------------------------------------------------

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True
    )

    robot = PandaRobot(
        robot_id
    )

    return physics_client, robot


# ================================================================
# CREATE ENVIRONMENT
# ================================================================

def create_environment(
    problem,
    client_id
):

    environment = (
        Environment.from_mpinets_problem(
            problem
        )
    )

    environment.create_pybullet_obstacles(
        client_id
    )

    return environment


# ================================================================
# GENERATE COLLISION-FREE GOAL
# ================================================================

def generate_goal(
    robot,
    environment,
    collision_checker,
    client_id
):
    """
    Generate a valid collision-free goal.

    A candidate goal must:
        1. Be a valid IK solution.
        2. Respect Panda joint limits.
        3. Satisfy the target position constraint.
        4. Be collision-free in PyBullet.

    Multiple IK solutions are checked because the first
    valid IK solution may still be in collision.
    """

    print()
    print("Generating goal...")

    goal_generator = GoalConfigurationGenerator(
        panda_robot=robot,
        environment=environment,
        client_id=client_id
    )

    # ------------------------------------------------------------
    # Generate multiple IK solutions.
    # ------------------------------------------------------------

    solutions = (
        goal_generator.solve_multiple(
            num_solutions=20,
            num_rest_poses=20,
            seed=42
        )
    )

    if not solutions:

        raise RuntimeError(
            "No valid IK goal configurations were generated."
        )

    print(
        f"IK solutions generated: "
        f"{len(solutions)}"
    )

    # ------------------------------------------------------------
    # Test every IK solution for collision.
    # ------------------------------------------------------------

    for index, solution in enumerate(
        solutions
    ):

        goal = np.asarray(
            solution["configuration"],
            dtype=float
        )

        collision_result = (
            collision_checker.check_configuration(
                goal
            )
        )

        is_collision = bool(
            collision_result["collision"]
        )

        print(
            f"Goal candidate {index + 1}: "
            f"collision = {is_collision}"
        )

        # --------------------------------------------------------
        # Select first collision-free solution.
        # --------------------------------------------------------

        if not is_collision:

            print(
                f"Selected collision-free goal: "
                f"{index + 1}"
            )

            return goal

    # ------------------------------------------------------------
    # No collision-free goal found.
    # ------------------------------------------------------------

    raise RuntimeError(
        "IK solutions were generated, but none "
        "were collision-free."
    )


# ================================================================
# GENERATE DIRECT TRAJECTORY
# ================================================================

def generate_direct_trajectory(
    trajectory_generator,
    start,
    goal
):

    trajectory = (
        trajectory_generator
        .generate_joint_trajectory(
            start,
            goal,
            num_points=NUM_POINTS
        )
    )

    return trajectory


# ================================================================
# GENERATE PERTURBED TRAJECTORY
# ================================================================

def generate_perturbed_trajectory(
    trajectory_generator,
    start,
    goal,
    rng,
    perturbation_scale
):
    """
    Generate a trajectory by perturbing the interior of
    a direct joint-space trajectory.

    The start and goal configurations remain unchanged.
    """

    trajectory = (
        trajectory_generator
        .generate_joint_trajectory(
            start,
            goal,
            num_points=NUM_POINTS
        )
    )

    # ------------------------------------------------------------
    # Interior region.
    # ------------------------------------------------------------

    start_index = int(
        NUM_POINTS * 0.25
    )

    end_index = int(
        NUM_POINTS * 0.75
    )

    indices = np.arange(
        start_index,
        end_index
    )

    # ------------------------------------------------------------
    # Random perturbation.
    # ------------------------------------------------------------

    perturbation = rng.normal(
        loc=0.0,
        scale=perturbation_scale,
        size=(len(indices), 7)
    )

    # ------------------------------------------------------------
    # Smooth perturbation.
    # ------------------------------------------------------------

    for joint in range(7):

        values = perturbation[
            :,
            joint
        ]

        kernel_size = 9

        kernel = (
            np.ones(kernel_size)
            / kernel_size
        )

        smoothed = np.convolve(
            values,
            kernel,
            mode="same"
        )

        perturbation[
            :,
            joint
        ] = smoothed

    # ------------------------------------------------------------
    # Apply perturbation.
    # ------------------------------------------------------------

    trajectory[
        indices
    ] += perturbation

    # ------------------------------------------------------------
    # Enforce Panda joint limits.
    # ------------------------------------------------------------

    limits = np.asarray(
        trajectory_generator.joint_limits,
        dtype=float
    )

    lower = limits[:, 0]
    upper = limits[:, 1]

    trajectory = np.clip(
        trajectory,
        lower,
        upper
    )

    # ------------------------------------------------------------
    # Keep exact start and goal.
    # ------------------------------------------------------------

    trajectory[0] = start
    trajectory[-1] = goal

    return trajectory


# ================================================================
# GENERATE CANDIDATE TRAJECTORIES
# ================================================================

def generate_candidate_set(
    robot,
    environment,
    collision_checker,
    trajectory_generator,
    start,
    goal,
    client_id
):

    candidates = []

    # ============================================================
    # 1. OBSTACLE-AWARE TRAJECTORIES
    # ============================================================

    print()
    print(
        "Generating obstacle-aware trajectories..."
    )

    planner = ObstacleAwarePlanner(
        panda_robot=robot,
        collision_checker=collision_checker,
        trajectory_generator=trajectory_generator,

        num_samples=500,

        k_neighbors=12,

        edge_points=20,

        seed=RANDOM_SEED
    )

    planned_results = (
        planner.plan_multiple(
            start_configuration=start,
            goal_configuration=goal,
            num_candidates=NUM_PLANNED
        )
    )

    planned_count = 0

    for result in planned_results:

        if result.get(
            "success",
            False
        ):

            candidates.append(
                (
                    "planned",
                    result["trajectory"]
                )
            )

            planned_count += 1

    print(
        "Successful planned trajectories: "
        f"{planned_count}"
    )

    # ============================================================
    # 2. DIRECT TRAJECTORIES
    # ============================================================

    print()
    print(
        "Generating direct trajectories..."
    )

    for _ in range(
        NUM_DIRECT
    ):

        trajectory = (
            generate_direct_trajectory(
                trajectory_generator,
                start,
                goal
            )
        )

        candidates.append(
            (
                "direct",
                trajectory
            )
        )

    # ============================================================
    # 3. PERTURBED TRAJECTORIES
    # ============================================================

    print()
    print(
        "Generating perturbed trajectories..."
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    perturbation_scales = [
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        1.00,
        1.15,
    ]

    for scale in perturbation_scales:

        trajectory = (
            generate_perturbed_trajectory(
                trajectory_generator,
                start,
                goal,
                rng,
                scale
            )
        )

        candidates.append(
            (
                "perturbed",
                trajectory
            )
        )

    return candidates


# ================================================================
# PRINT TRAJECTORY RESULT
# ================================================================

def print_trajectory_result(
    record
):

    print()
    print(
        f"Trajectory "
        f"{record['trajectory_index'] + 1}"
    )

    print(
        f"Source: "
        f"{record['source']}"
    )

    print(
        f"Failure: "
        f"{record['failure']}"
    )

    print(
        f"Collision count: "
        f"{record['collision_count']}"
    )

    print(
        f"Obstacle collisions: "
        f"{record['obstacle_collision_count']}"
    )

    print(
        f"Self collisions: "
        f"{record['self_collision_count']}"
    )

    print(
        f"Joint-limit violations: "
        f"{record['joint_limit_violation_count']}"
    )

    print(
        f"Minimum clearance: "
        f"{record['minimum_clearance']:.6f} m"
    )

    print(
        f"Interior minimum clearance: "
        f"{record['interior_minimum_clearance']:.6f} m"
    )

    print(
        f"Interior mean clearance: "
        f"{record['interior_mean_clearance']:.6f} m"
    )

    print(
        f"Joint-space length: "
        f"{record['joint_space_length']:.6f}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print(
        "ML FAILURE-PREDICTION DATASET GENERATION TEST"
    )
    print("=" * 70)

    # ============================================================
    # LOAD DATASET
    # ============================================================

    dataset = (
        load_mpinets_dataset()
    )

    problem = (
        dataset[
            "tabletop"
        ][
            "task_oriented"
        ][0]
    )

    print()
    print(
        "Selected: "
        "tabletop/task_oriented/0"
    )

    # ============================================================
    # PYBULLET
    # ============================================================

    client_id, robot = (
        setup_pybullet()
    )

    print()
    print("Panda loaded.")

    print(
        f"DOF: {robot.DOF}"
    )

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    environment = (
        create_environment(
            problem,
            client_id
        )
    )

    print()
    print(
        "Environment created."
    )

    print(
        f"Obstacles: "
        f"{environment.num_obstacles}"
    )

    # ============================================================
    # COLLISION CHECKER
    # ============================================================

    collision_checker = (
        CollisionChecker(
            panda_robot=robot,
            environment=environment,
            client_id=client_id
        )
    )

    # ============================================================
    # CLEARANCE ANALYZER
    # ============================================================

    clearance_analyzer = (
        ClearanceAnalyzer(
            panda_robot=robot,
            environment=environment,
            client_id=client_id
        )
    )

    # ============================================================
    # TRAJECTORY GENERATOR
    # ============================================================

    trajectory_generator = (
        TrajectoryGenerator(
            panda_robot=robot
        )
    )

    # ============================================================
    # FEATURE EXTRACTOR
    # ============================================================

    feature_extractor = (
        TrajectoryFeatureExtractor(
            collision_checker=collision_checker,
            clearance_analyzer=clearance_analyzer,
            trajectory_generator=trajectory_generator
        )
    )

    # ============================================================
    # START CONFIGURATION
    # ============================================================

    start = np.asarray(
        problem.q0,
        dtype=float
    )

    print()
    print(
        "Start configuration:"
    )

    print(start)

    # ============================================================
    # CHECK START
    # ============================================================

    start_result = (
        collision_checker.check_configuration(
            start
        )
    )

    print()
    print(
        "Start collision: "
        f"{start_result['collision']}"
    )

    if start_result["collision"]:

        raise RuntimeError(
            "MπNets start configuration is in collision."
        )

    # ============================================================
    # GENERATE GOAL
    # ============================================================

    goal = generate_goal(
        robot=robot,
        environment=environment,
        collision_checker=collision_checker,
        client_id=client_id
    )

    print()
    print(
        "Goal:"
    )

    print(goal)

    # ============================================================
    # CHECK GOAL
    # ============================================================

    goal_result = (
        collision_checker.check_configuration(
            goal
        )
    )

    print()
    print(
        "Goal collision: "
        f"{goal_result['collision']}"
    )

    if goal_result["collision"]:

        raise RuntimeError(
            "Selected goal is unexpectedly in collision."
        )

    # ============================================================
    # GENERATE CANDIDATES
    # ============================================================

    candidates = (
        generate_candidate_set(
            robot=robot,
            environment=environment,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            start=start,
            goal=goal,
            client_id=client_id
        )
    )

    print()
    print(
        f"Total candidate trajectories: "
        f"{len(candidates)}"
    )

    if not candidates:

        raise RuntimeError(
            "No candidate trajectories were generated."
        )

    # ============================================================
    # EXTRACT FEATURES
    # ============================================================

    print()
    print("=" * 70)
    print(
        "EVALUATING TRAJECTORIES"
    )
    print("=" * 70)

    trajectories = [
        trajectory
        for _, trajectory in candidates
    ]

    records = (
        feature_extractor.extract_multiple(
            trajectories,
            include_label=True
        )
    )

    # ============================================================
    # ADD SOURCE INFORMATION
    # ============================================================

    for record, (
        source,
        _
    ) in zip(
        records,
        candidates
    ):

        record["source"] = source

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    successful = 0
    failed = 0

    for record in records:

        if record["failure"] == 0:

            successful += 1

        else:

            failed += 1

        print_trajectory_result(
            record
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print(
        "DATASET SUMMARY"
    )
    print("=" * 70)

    print(
        f"Total trajectories: "
        f"{len(records)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print()

    # ============================================================
    # CLASS BALANCE CHECK
    # ============================================================

    if (
        successful > 0
        and failed > 0
    ):

        print(
            "SUCCESS: Both failure classes are present."
        )

        print(
            "The trajectory generation strategy "
            "produced both successful and failed examples."
        )

        print()
        print(
            "The next stage can build the actual "
            "ML training dataset."
        )

    elif failed == 0:

        print(
            "WARNING: No failed trajectories were generated."
        )

        print(
            "The current perturbations did not produce "
            "collisions."
        )

        print(
            "We will need stronger or different "
            "failure-generation strategies."
        )

    elif successful == 0:

        print(
            "WARNING: No successful trajectories were generated."
        )

        print(
            "The current candidate-generation strategy "
            "needs adjustment."
        )

    # ============================================================
    # CLOSE PYBULLET
    # ============================================================

    p.disconnect(
        client_id
    )

    print()
    print("=" * 70)
    print(
        "DATASET GENERATION TEST COMPLETED"
    )
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()