"""
Test Recoverability Analyzer

This test verifies the complete recovery workflow:

    10 feasible planner trajectories
                +
        1 deliberately failed trajectory
                |
                v
        RecoverabilityAnalyzer
                |
        +-------+-------+
        |               |
      FAILED         FEASIBLE
        |               |
        +-------> recovery
                    |
                    v
              feasible alternative
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from robot.panda_robot import PandaRobot
from environment.environment import Environment

from analysis.collision_checker import CollisionChecker

from planning.trajectory_generator import TrajectoryGenerator
from planning.goal_configuration import GoalConfigurationGenerator
from planning.obstacle_aware_planner import ObstacleAwarePlanner

from analysis.recoverability_analyzer import (
    RecoverabilityAnalyzer,
)


# =====================================================================
# DATASET
# =====================================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)


# =====================================================================
# TEST CONFIGURATION
# =====================================================================

ENVIRONMENT_NAME = "tabletop"
PROBLEM_SET_NAME = "task_oriented"
PROBLEM_INDEX = 0

NUM_GOAL_SOLUTIONS = 100
NUM_REST_POSES = 100

NUM_CANDIDATE_PATHS = 10

PLANNER_SAMPLES = 700
PLANNER_K_NEIGHBORS = 15
PLANNER_EDGE_POINTS = 25

RANDOM_SEED = 42


# =====================================================================
# LOAD DATASET
# =====================================================================

def load_problem():

    print(
        "\nLoading MπNets dataset..."
    )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "rb",
    ) as file:

        dataset = pickle.load(file)

    problem = dataset[
        ENVIRONMENT_NAME
    ][
        PROBLEM_SET_NAME
    ][
        PROBLEM_INDEX
    ]

    print(
        f"Environment : {ENVIRONMENT_NAME}"
    )

    print(
        f"Problem set : {PROBLEM_SET_NAME}"
    )

    print(
        f"Problem      : {PROBLEM_INDEX}"
    )

    return problem


# =====================================================================
# CREATE PYBULLET SIMULATION
# =====================================================================

def create_simulation():

    physics_client = p.connect(
        p.DIRECT
    )

    if physics_client < 0:

        raise RuntimeError(
            "Could not connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81,
    )

    p.loadURDF(
        "plane.urdf"
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
    )

    panda_robot = PandaRobot(
        robot_id
    )

    return (
        physics_client,
        panda_robot,
    )


# =====================================================================
# FIND COLLISION-FREE GOAL
# =====================================================================

def find_collision_free_goal(
    panda_robot,
    environment,
    collision_checker,
    physics_client,
):

    print(
        "\nGenerating goal configurations..."
    )

    goal_generator = GoalConfigurationGenerator(
        panda_robot=panda_robot,
        environment=environment,
        target_position=environment.target_position,
        target_half_extents=environment.target_half_extents,
        client_id=physics_client,
    )

    solutions = goal_generator.solve_multiple(
        num_solutions=NUM_GOAL_SOLUTIONS,
        num_rest_poses=NUM_REST_POSES,
        seed=RANDOM_SEED,
    )

    print(
        f"IK solutions generated: "
        f"{len(solutions)}"
    )

    if len(solutions) == 0:

        raise RuntimeError(
            "No valid IK solutions were generated."
        )

    for solution_index, solution in enumerate(
        solutions
    ):

        configuration = np.asarray(
            solution["configuration"],
            dtype=float,
        )

        result = collision_checker.check_configuration(
            configuration
        )

        if not result["collision"]:

            print(
                f"  Goal {solution_index + 1}: "
                f"COLLISION-FREE"
            )

            print(
                "\nSelected collision-free goal:"
            )

            print(
                np.array2string(
                    configuration,
                    precision=6,
                    suppress_small=True,
                )
            )

            return configuration

        print(
            f"  Goal {solution_index + 1}: "
            f"collision"
        )

    raise RuntimeError(
        "No collision-free goal configuration found."
    )


# =====================================================================
# GENERATE VALID CANDIDATES
# =====================================================================

def generate_candidates(
    planner,
    start,
    goal,
):

    print(
        "\nGenerating candidate trajectories..."
    )

    planned_results = planner.plan_multiple(
        start,
        goal,
        NUM_CANDIDATE_PATHS,
    )

    trajectories = []

    for result in planned_results:

        trajectory = result.get(
            "trajectory"
        )

        if trajectory is None:
            continue

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if trajectory.ndim != 2:
            continue

        if trajectory.shape[1] != 7:
            continue

        trajectories.append(
            trajectory
        )

    print(
        f"Candidate trajectories obtained: "
        f"{len(trajectories)}"
    )

    if len(trajectories) < 2:

        raise RuntimeError(
            "Fewer than two candidate trajectories "
            "were generated."
        )

    return trajectories


# =====================================================================
# CREATE DELIBERATELY FAILED TRAJECTORY
# =====================================================================

def create_failed_trajectory(
    valid_trajectory,
    collision_checker,
    panda_robot,
):
    """
    Create a deliberately invalid trajectory.

    We keep the original trajectory unchanged and search for a
    configuration that violates the robot's joint limits.

    CollisionChecker treats joint-limit violations as failures.

    The returned trajectory therefore provides a deterministic
    failure case for testing the recovery mechanism.
    """

    print(
        "\nCreating deliberate failure case..."
    )

    failed_trajectory = np.asarray(
        valid_trajectory,
        dtype=float,
    ).copy()

    # -------------------------------------------------------------
    # Try several increasingly large violations.
    # -------------------------------------------------------------

    test_values = [
        10.0,
        20.0,
        -10.0,
        -20.0,
    ]

    original_value = failed_trajectory[
        len(failed_trajectory) // 2,
        0,
    ]

    for test_value in test_values:

        failed_trajectory[
            len(failed_trajectory) // 2,
            0,
        ] = test_value

        result = collision_checker.check_trajectory(
            failed_trajectory
        )

        if not result["collision_free"]:

            print(
                "Deliberate failure created."
            )

            print(
                f"  Modified joint: J1"
            )

            print(
                f"  Original value: "
                f"{original_value:.6f}"
            )

            print(
                f"  Test value    : "
                f"{test_value:.6f}"
            )

            print(
                f"  Collision count: "
                f"{result['collision_count']}"
            )

            return failed_trajectory

    # Restore the original value if the search somehow failed.
    failed_trajectory[
        len(failed_trajectory) // 2,
        0,
    ] = original_value

    raise RuntimeError(
        "Could not create a deliberately failed trajectory."
    )


# =====================================================================
# PRINT CANDIDATE VALIDATION
# =====================================================================

def print_candidate_results(
    analysis_result,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CANDIDATE VALIDATION"
    )

    print(
        "=" * 70
    )

    for candidate in analysis_result[
        "candidate_results"
    ]:

        index = candidate[
            "candidate_index"
        ]

        validation = candidate[
            "validation"
        ]

        status = (
            "FEASIBLE"
            if candidate["feasible"]
            else "FAILED"
        )

        print(
            f"\nCandidate {index + 1}"
        )

        print(
            f"  Status                  : "
            f"{status}"
        )

        print(
            f"  Collision count        : "
            f"{validation['collision_count']}"
        )

        print(
            f"  Obstacle collisions    : "
            f"{validation['obstacle_collision_count']}"
        )

        print(
            f"  Self collisions        : "
            f"{validation['self_collision_count']}"
        )

        print(
            f"  Joint-limit violations : "
            f"{validation['joint_limit_violation_count']}"
        )


# =====================================================================
# PRINT RECOVERY RESULTS
# =====================================================================

def print_recovery_results(
    analyzer,
    analysis_result,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RECOVERY ANALYSIS"
    )

    print(
        "=" * 70
    )

    summary = analysis_result[
        "summary"
    ]

    print(
        f"\nTotal candidates        : "
        f"{summary['total_candidates']}"
    )

    print(
        f"Feasible candidates     : "
        f"{summary['feasible_candidates']}"
    )

    print(
        f"Failed candidates       : "
        f"{summary['failed_candidates']}"
    )

    print(
        f"Recoverable failures    : "
        f"{summary['recoverable_failed_candidates']}"
    )

    print(
        f"Unrecoverable failures  : "
        f"{summary['unrecoverable_failed_candidates']}"
    )

    print(
        f"Recoverability rate     : "
        f"{summary['recoverability_rate']:.2%}"
    )

    print(
        "\n"
        + "-" * 70
    )

    print(
        "RECOVERY OPTIONS"
    )

    print(
        "-" * 70
    )

    for failed_candidate in analysis_result[
        "failed_candidates"
    ]:

        failed_index = failed_candidate[
            "candidate_index"
        ]

        alternatives = analyzer.find_alternatives(
            failed_candidate_index=failed_index,
            candidate_results=analysis_result[
                "candidate_results"
            ],
            max_alternatives=3,
        )

        print(
            f"\nFailed candidate {failed_index + 1}:"
        )

        if not alternatives:

            print(
                "  NO FEASIBLE RECOVERY"
            )

            continue

        print(
            f"  Feasible alternatives: "
            f"{len(alternatives)}"
        )

        for alternative in alternatives:

            print(
                f"    -> Candidate "
                f"{alternative['candidate_index'] + 1}"
                f" | distance = "
                f"{alternative['trajectory_distance']:.6f}"
            )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TESTING RECOVERABILITY ANALYZER"
    )

    print(
        "=" * 70
    )

    physics_client = None

    try:

        # ---------------------------------------------------------
        # Load problem
        # ---------------------------------------------------------

        problem = load_problem()

        # ---------------------------------------------------------
        # Create PyBullet
        # ---------------------------------------------------------

        (
            physics_client,
            panda_robot,
        ) = create_simulation()

        # ---------------------------------------------------------
        # Environment
        # ---------------------------------------------------------

        environment = (
            Environment.from_mpinets_problem(
                problem
            )
        )

        obstacle_ids = (
            environment.create_pybullet_obstacles(
                client_id=physics_client
            )
        )

        print(
            f"\nObstacles created: "
            f"{len(obstacle_ids)}"
        )

        # ---------------------------------------------------------
        # Collision checker
        # ---------------------------------------------------------

        collision_checker = CollisionChecker(
            panda_robot=panda_robot,
            environment=environment,
            client_id=physics_client,
        )

        # ---------------------------------------------------------
        # Trajectory generator
        # ---------------------------------------------------------

        trajectory_generator = (
            TrajectoryGenerator(
                panda_robot
            )
        )

        # ---------------------------------------------------------
        # Start
        # ---------------------------------------------------------

        start = np.asarray(
            problem.q0,
            dtype=float,
        )

        start_result = (
            collision_checker.check_configuration(
                start
            )
        )

        print(
            "\nStart configuration:"
        )

        print(
            f"  Collision: "
            f"{start_result['collision']}"
        )

        if start_result["collision"]:

            raise RuntimeError(
                "Start configuration is not collision-free."
            )

        # ---------------------------------------------------------
        # Goal
        # ---------------------------------------------------------

        goal = find_collision_free_goal(
            panda_robot=panda_robot,
            environment=environment,
            collision_checker=collision_checker,
            physics_client=physics_client,
        )

        # ---------------------------------------------------------
        # Planner
        # ---------------------------------------------------------

        planner = ObstacleAwarePlanner(
            panda_robot=panda_robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=PLANNER_SAMPLES,
            k_neighbors=PLANNER_K_NEIGHBORS,
            edge_points=PLANNER_EDGE_POINTS,
            seed=RANDOM_SEED,
        )

        # ---------------------------------------------------------
        # Generate 10 valid trajectories
        # ---------------------------------------------------------

        trajectories = generate_candidates(
            planner=planner,
            start=start,
            goal=goal,
        )

        # ---------------------------------------------------------
        # Make one deliberate failure
        # ---------------------------------------------------------

        failed_trajectory = create_failed_trajectory(
            valid_trajectory=trajectories[0],
            collision_checker=collision_checker,
            panda_robot=panda_robot,
        )

        # ---------------------------------------------------------
        # Add the failed trajectory
        # ---------------------------------------------------------

        all_trajectories = (
            trajectories
            + [failed_trajectory]
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RECOVERABILITY TEST SET"
        )

        print(
            "=" * 70
        )

        print(
            f"Valid planner trajectories : "
            f"{len(trajectories)}"
        )

        print(
            "Deliberate failed trajectory: 1"
        )

        print(
            f"Total trajectories          : "
            f"{len(all_trajectories)}"
        )

        # ---------------------------------------------------------
        # Recoverability analyzer
        # ---------------------------------------------------------

        analyzer = RecoverabilityAnalyzer(
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
        )

        # ---------------------------------------------------------
        # Analyze
        # ---------------------------------------------------------

        analysis_result = (
            analyzer.analyze_recoverability(
                all_trajectories
            )
        )

        # ---------------------------------------------------------
        # Candidate validation
        # ---------------------------------------------------------

        print_candidate_results(
            analysis_result
        )

        # ---------------------------------------------------------
        # Recovery analysis
        # ---------------------------------------------------------

        print_recovery_results(
            analyzer,
            analysis_result,
        )

        # ---------------------------------------------------------
        # Verify that exactly one candidate failed
        # ---------------------------------------------------------

        failed_candidates = (
            analysis_result[
                "failed_candidates"
            ]
        )

        assert len(
            failed_candidates
        ) == 1, (
            "Expected exactly one failed candidate, "
            f"but found {len(failed_candidates)}."
        )

        failed_index = failed_candidates[0][
            "candidate_index"
        ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RECOVERY SELECTION TEST"
        )

        print(
            "=" * 70
        )

        print(
            f"Failed candidate index: "
            f"{failed_index}"
        )

        # ---------------------------------------------------------
        # Select recovery
        # ---------------------------------------------------------

        recovery = (
            analyzer.select_recovery_trajectory(
                recoverability_result=analysis_result,
                failed_candidate_index=failed_index,
            )
        )

        assert recovery is not None, (
            "The failed candidate should have "
            "at least one feasible recovery."
        )

        recovery_index = recovery[
            "candidate_index"
        ]

        print(
            f"Selected recovery candidate: "
            f"{recovery_index}"
        )

        print(
            f"Trajectory distance: "
            f"{recovery['trajectory_distance']:.6f}"
        )

        # ---------------------------------------------------------
        # Verify recovery is physically feasible
        # ---------------------------------------------------------

        recovery_validation = (
            recovery[
                "validation"
            ]
        )

        assert recovery_validation[
            "collision_free"
        ], (
            "Selected recovery trajectory "
            "is not collision-free."
        )

        print(
            "Recovery trajectory: "
            "PHYSICALLY FEASIBLE"
        )

        # ---------------------------------------------------------
        # Verify recovery statistics
        # ---------------------------------------------------------

        summary = analysis_result[
            "summary"
        ]

        assert (
            summary[
                "total_candidates"
            ]
            == 11
        )

        assert (
            summary[
                "feasible_candidates"
            ]
            == 10
        )

        assert (
            summary[
                "failed_candidates"
            ]
            == 1
        )

        assert (
            summary[
                "recoverable_failed_candidates"
            ]
            == 1
        )

        assert (
            summary[
                "unrecoverable_failed_candidates"
            ]
            == 0
        )

        assert (
            summary[
                "recoverability_rate"
            ]
            == 1.0
        )

        # ---------------------------------------------------------
        # PASS
        # ---------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RECOVERABILITY ANALYZER TEST PASSED"
        )

        print(
            "=" * 70
        )

        print(
            "\nVerified:"
        )

        print(
            "  [PASS] Multiple candidate trajectories"
        )

        print(
            "  [PASS] Failed trajectory detection"
        )

        print(
            "  [PASS] Feasible alternative detection"
        )

        print(
            "  [PASS] Recovery trajectory selection"
        )

        print(
            "  [PASS] Physical validation of recovery"
        )

    finally:

        if physics_client is not None:

            p.disconnect(
                physics_client
            )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()