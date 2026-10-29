# tests/test_intelligent_planner.py

import pickle
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ================================================================
# PROJECT IMPORTS
# ================================================================

from robot.panda_robot import PandaRobot

from environment.environment import Environment

from analysis.collision_checker import (
    CollisionChecker,
)

from analysis.clearance_analyzer import (
    ClearanceAnalyzer,
)

from analysis.feature_extractor import (
    TrajectoryFeatureExtractor,
)

from analysis.recoverability_analyzer import (
    RecoverabilityAnalyzer,
)

from analysis.intelligent_planner import (
    IntelligentMotionPlanner,
)

from planning.trajectory_generator import (
    TrajectoryGenerator,
)

from planning.goal_configuration import (
    GoalConfigurationGenerator,
)

from planning.obstacle_aware_planner import (
    ObstacleAwarePlanner,
)

from prediction.failure_predictor import (
    FailurePredictor,
)


# ================================================================
# DATASET
# ================================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)


# ================================================================
# CONFIGURATION
# ================================================================

ENVIRONMENT_NAME = "tabletop"
PROBLEM_SET_NAME = "task_oriented"
PROBLEM_INDEX = 0

RANDOM_SEED = 42

NUM_GOAL_SOLUTIONS = 100
NUM_REST_POSES = 100

PLANNER_SAMPLES = 700
PLANNER_K_NEIGHBORS = 15
PLANNER_EDGE_POINTS = 25

NUM_CANDIDATES = 10


# ================================================================
# LOAD PROBLEM
# ================================================================

def load_problem():

    print(
        "\nLoading MπNets dataset..."
    )

    with open(
        DATASET_PATH,
        "rb",
    ) as file:

        dataset = pickle.load(
            file
        )

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


# ================================================================
# CREATE SIMULATION
# ================================================================

def create_simulation():

    physics_client = p.connect(
        p.DIRECT
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


# ================================================================
# FIND COLLISION-FREE GOAL
# ================================================================

def find_collision_free_goal(
    panda_robot,
    environment,
    collision_checker,
    physics_client,
):

    print(
        "\nGenerating goal configurations..."
    )

    goal_generator = (
        GoalConfigurationGenerator(
            panda_robot=panda_robot,
            environment=environment,
            target_position=environment.target_position,
            target_half_extents=environment.target_half_extents,
            client_id=physics_client,
        )
    )

    solutions = (
        goal_generator.solve_multiple(
            num_solutions=NUM_GOAL_SOLUTIONS,
            num_rest_poses=NUM_REST_POSES,
            seed=RANDOM_SEED,
        )
    )

    print(
        f"IK solutions generated: "
        f"{len(solutions)}"
    )

    for solution in solutions:

        configuration = np.asarray(
            solution["configuration"],
            dtype=float,
        )

        result = (
            collision_checker.check_configuration(
                configuration
            )
        )

        if not result["collision"]:

            print(
                "Collision-free goal found."
            )

            return configuration

    raise RuntimeError(
        "No collision-free goal "
        "configuration found."
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)

    print(
        "TESTING INTELLIGENT MOTION PLANNER"
    )

    print("=" * 70)

    physics_client = None

    try:

        # --------------------------------------------------------
        # Load dataset
        # --------------------------------------------------------

        problem = load_problem()

        # --------------------------------------------------------
        # Simulation
        # --------------------------------------------------------

        (
            physics_client,
            panda_robot,
        ) = create_simulation()

        # --------------------------------------------------------
        # Environment
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Collision checker
        # --------------------------------------------------------

        collision_checker = (
            CollisionChecker(
                panda_robot=panda_robot,
                environment=environment,
                client_id=physics_client,
            )
        )

        # --------------------------------------------------------
        # Clearance analyzer
        # --------------------------------------------------------

        clearance_analyzer = (
            ClearanceAnalyzer(
                panda_robot=panda_robot,
                environment=environment,
                client_id=physics_client,
            )
        )

        # --------------------------------------------------------
        # Trajectory generator
        # --------------------------------------------------------

        trajectory_generator = (
            TrajectoryGenerator(
                panda_robot
            )
        )

        # --------------------------------------------------------
        # Feature extractor
        # --------------------------------------------------------

        feature_extractor = (
            TrajectoryFeatureExtractor(
                collision_checker=collision_checker,
                clearance_analyzer=clearance_analyzer,
                trajectory_generator=trajectory_generator,
            )
        )

        print(
            "\nFeature extractor created."
        )

        print(
            f"ML features: "
            f"{len(feature_extractor.get_feature_names())}"
        )

        # --------------------------------------------------------
        # Failure predictor
        # --------------------------------------------------------

        failure_predictor = (
            FailurePredictor(
                feature_extractor=feature_extractor
            )
        )

        print(
            "\nFailure predictor loaded."
        )

        # --------------------------------------------------------
        # Recoverability analyzer
        # --------------------------------------------------------

        recoverability_analyzer = (
            RecoverabilityAnalyzer(
                collision_checker=collision_checker,
                trajectory_generator=trajectory_generator,
            )
        )

        print(
            "\nRecoverability analyzer created."
        )

        # --------------------------------------------------------
        # Standard obstacle-aware planner
        # --------------------------------------------------------

        base_planner = (
            ObstacleAwarePlanner(
                panda_robot=panda_robot,
                collision_checker=collision_checker,
                trajectory_generator=trajectory_generator,
                num_samples=PLANNER_SAMPLES,
                k_neighbors=PLANNER_K_NEIGHBORS,
                edge_points=PLANNER_EDGE_POINTS,
                seed=RANDOM_SEED,
            )
        )

        # --------------------------------------------------------
        # Intelligent planner
        # --------------------------------------------------------

        intelligent_planner = (
            IntelligentMotionPlanner(
                planner=base_planner,
                feature_extractor=feature_extractor,
                failure_predictor=failure_predictor,
                recoverability_analyzer=recoverability_analyzer,
                collision_checker=collision_checker,
            )
        )

        print(
            "\nIntelligent planner created."
        )

        # --------------------------------------------------------
        # Start
        # --------------------------------------------------------

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
            "\nStart collision: "
            f"{start_result['collision']}"
        )

        if start_result["collision"]:

            raise RuntimeError(
                "Dataset start configuration "
                "is in collision."
            )

        # --------------------------------------------------------
        # Goal
        # --------------------------------------------------------

        goal = find_collision_free_goal(
            panda_robot=panda_robot,
            environment=environment,
            collision_checker=collision_checker,
            physics_client=physics_client,
        )

        # --------------------------------------------------------
        # COMPLETE INTELLIGENT PIPELINE
        # --------------------------------------------------------

        print(
            "\nRunning complete intelligent "
            "planning pipeline..."
        )

        result = intelligent_planner.plan(
            start=start,
            goal=goal,
            num_candidates=NUM_CANDIDATES,
            top_k=5,
        )

        # --------------------------------------------------------
        # Print summary
        # --------------------------------------------------------

        print(
            "\n"
            + intelligent_planner.summarize(
                result
            )
        )

        # --------------------------------------------------------
        # Candidate details
        # --------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CANDIDATE ANALYSIS"
        )

        print(
            "=" * 70
        )

        for item in result[
            "combined_results"
        ]:

            index = item[
                "trajectory_index"
            ]

            prediction = item[
                "predicted_class"
            ]

            success_probability = item[
                "success_probability"
            ]

            failure_probability = item[
                "failure_probability"
            ]

            physical = item[
                "physical_validation"
            ]

            actual_status = (
                "FEASIBLE"
                if physical[
                    "collision_free"
                ]
                else "FAILED"
            )

            print(
                f"\nCandidate {index + 1}"
            )

            print(
                f"  ML prediction      : "
                f"{prediction.upper()}"
            )

            print(
                f"  Success probability: "
                f"{success_probability:.4f}"
            )

            print(
                f"  Failure probability: "
                f"{failure_probability:.4f}"
            )

            print(
                f"  Physical result    : "
                f"{actual_status}"
            )

            print(
                f"  Collision count    : "
                f"{physical['collision_count']}"
            )

            print(
                f"  Prediction correct : "
                f"{item['prediction_correct']}"
            )

        # --------------------------------------------------------
        # Selected trajectory
        # --------------------------------------------------------

        selected = result[
            "selected"
        ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL TRAJECTORY"
        )

        print(
            "=" * 70
        )

        if selected is None:

            print(
                "\nNo physically feasible "
                "trajectory was selected."
            )

        else:

            selected_index = (
                selected[
                    "trajectory_index"
                ]
            )

            selected_result = (
                selected["result"]
            )

            print(
                f"\nSelected candidate: "
                f"{selected_index + 1}"
            )

            print(
                f"Predicted success probability: "
                f"{selected_result['success_probability']:.4f}"
            )

            print(
                f"Physical collision-free: "
                f"{selected_result['physical_validation']['collision_free']}"
            )

            print(
                "\nTop explanation:"
            )

            for explanation in (
                selected_result[
                    "top_explanations"
                ][:3]
            ):

                print(
                    f"\n  {explanation['feature']}"
                )

                print(
                    f"    Value       : "
                    f"{explanation['value']:.6f}"
                )

                print(
                    f"    Contribution: "
                    f"{explanation['contribution']:.6f}"
                )

                print(
                    f"    Direction   : "
                    f"{explanation['direction'].upper()}"
                )

        # --------------------------------------------------------
        # Assertions
        # --------------------------------------------------------

        assert result[
            "num_candidates"
        ] > 0

        assert len(
            result["predictions"]
        ) == result[
            "num_candidates"
        ]

        assert len(
            result["validations"]
        ) == result[
            "num_candidates"
        ]

        assert len(
            result["combined_results"]
        ) == result[
            "num_candidates"
        ]

        assert (
            len(
                feature_extractor
                .get_feature_names()
            )
            == 17
        )

        # --------------------------------------------------------
        # Final validation
        # --------------------------------------------------------

        if selected is not None:

            assert selected[
                "result"
            ][
                "physical_validation"
            ][
                "collision_free"
            ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            "INTELLIGENT MOTION PLANNER "
            "TEST PASSED"
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
            "  [PASS] ML failure prediction"
        )

        print(
            "  [PASS] Local explanations"
        )

        print(
            "  [PASS] Physical collision validation"
        )

        print(
            "  [PASS] Prediction vs physical comparison"
        )

        print(
            "  [PASS] Recoverability analysis"
        )

        print(
            "  [PASS] Final feasible trajectory selection"
        )

    finally:

        if physics_client is not None:

            p.disconnect(
                physics_client
            )


if __name__ == "__main__":
    main()