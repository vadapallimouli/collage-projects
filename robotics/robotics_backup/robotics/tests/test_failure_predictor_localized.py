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

from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer
from analysis.feature_extractor import (
    TrajectoryFeatureExtractor,
)

from planning.trajectory_generator import TrajectoryGenerator
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

    for solution in solutions:

        configuration = np.asarray(
            solution["configuration"],
            dtype=float,
        )

        result = collision_checker.check_configuration(
            configuration
        )

        if not result["collision"]:

            print(
                "Collision-free goal found."
            )

            return configuration

    raise RuntimeError(
        "No collision-free goal configuration found."
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TESTING LOCALIZED FAILURE PREDICTOR"
    )

    print(
        "=" * 70
    )

    physics_client = None

    try:

        # --------------------------------------------------------
        # Load dataset
        # --------------------------------------------------------

        problem = load_problem()

        # --------------------------------------------------------
        # PyBullet
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

        collision_checker = CollisionChecker(
            panda_robot=panda_robot,
            environment=environment,
            client_id=physics_client,
        )

        # --------------------------------------------------------
        # Clearance analyzer
        # --------------------------------------------------------

        clearance_analyzer = ClearanceAnalyzer(
            panda_robot=panda_robot,
            environment=environment,
            client_id=physics_client,
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

        predictor = FailurePredictor(
            feature_extractor=feature_extractor
        )

        print(
            "\nFailure predictor loaded."
        )

        print(
            f"Model: "
            f"{type(predictor.model).__name__}"
        )

        print(
            f"Features: "
            f"{len(predictor.feature_names)}"
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
        # Planner
        # --------------------------------------------------------

        planner = ObstacleAwarePlanner(
            panda_robot=panda_robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=PLANNER_SAMPLES,
            k_neighbors=PLANNER_K_NEIGHBORS,
            edge_points=PLANNER_EDGE_POINTS,
            seed=RANDOM_SEED,
        )

        # --------------------------------------------------------
        # Generate candidates
        # --------------------------------------------------------

        print(
            "\nGenerating candidate trajectories..."
        )

        planned_results = planner.plan_multiple(
            start,
            goal,
            5,
        )

        trajectories = []

        for result in planned_results:

            trajectory = result.get(
                "trajectory"
            )

            if trajectory is None:
                continue

            trajectories.append(
                np.asarray(
                    trajectory,
                    dtype=float,
                )
            )

        if len(trajectories) == 0:

            raise RuntimeError(
                "No candidate trajectories were generated."
            )

        print(
            f"Candidate trajectories: "
            f"{len(trajectories)}"
        )

        # --------------------------------------------------------
        # Predict all candidates
        # --------------------------------------------------------

        predictions = (
            predictor.predict_multiple(
                trajectories=trajectories,
                top_k=5,
            )
        )

        # --------------------------------------------------------
        # Print predictions
        # --------------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TRAJECTORY PREDICTIONS"
        )

        print(
            "=" * 70
        )

        for result in predictions:

            index = result[
                "trajectory_index"
            ]

            predicted_class = result[
                "predicted_class"
            ]

            success_probability = result[
                "success_probability"
            ]

            failure_probability = result[
                "failure_probability"
            ]

            print(
                f"\nCandidate {index + 1}"
            )

            print(
                f"  Prediction          : "
                f"{predicted_class.upper()}"
            )

            print(
                f"  Success probability : "
                f"{success_probability:.4f}"
            )

            print(
                f"  Failure probability : "
                f"{failure_probability:.4f}"
            )

        # --------------------------------------------------------
        # Detailed explanation for first candidate
        # --------------------------------------------------------

        first_result = predictions[0]

        print(
            "\n"
            + "=" * 70
        )

        print(
            "SHAP-STYLE LOCAL EXPLANATION"
        )

        print(
            "=" * 70
        )

        print(
            f"\nCandidate "
            f"{first_result['trajectory_index'] + 1}"
        )

        print(
            f"Prediction: "
            f"{first_result['predicted_class'].upper()}"
        )

        print(
            f"Failure probability: "
            f"{first_result['failure_probability']:.4f}"
        )

        print(
            "\nTop contributing features:"
        )

        for rank, explanation in enumerate(
            first_result[
                "top_explanations"
            ],
            start=1,
        ):

            print(
                f"\n{rank}. "
                f"{explanation['feature']}"
            )

            print(
                f"   Value       : "
                f"{explanation['value']:.6f}"
            )

            print(
                f"   Contribution: "
                f"{explanation['shap_value']:.6f}"
            )

            print(
                f"   Direction   : "
                f"{explanation['direction'].upper()}"
            )

            print(
                f"   Reason      : "
                f"{explanation['reason']}"
            )

        # --------------------------------------------------------
        # Feature validation
        # --------------------------------------------------------

        first_features = first_result[
            "features"
        ]

        ml_features = (
            predictor.feature_names
        )

        missing = [
            feature
            for feature in ml_features
            if feature not in first_features
        ]

        if missing:

            raise AssertionError(
                "Missing features after extraction:\n"
                + "\n".join(missing)
            )

        # --------------------------------------------------------
        # Probability validation
        # --------------------------------------------------------

        for result in predictions:

            assert 0.0 <= (
                result["success_probability"]
            ) <= 1.0

            assert 0.0 <= (
                result["failure_probability"]
            ) <= 1.0

            assert np.isclose(
                result["success_probability"]
                + result["failure_probability"],
                1.0,
                atol=1e-6,
            )

        # --------------------------------------------------------
        # Final checks
        # --------------------------------------------------------

        assert len(
            predictor.feature_names
        ) == 17

        assert (
            type(
                predictor.model
            ).__name__
            == "XGBClassifier"
        )

        assert len(
            predictions
        ) == len(
            trajectories
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "LOCALIZED FAILURE PREDICTOR TEST PASSED"
        )

        print(
            "=" * 70
        )

        print(
            "\nVerified:"
        )

        print(
            "  [PASS] Localized XGBoost loaded"
        )

        print(
            "  [PASS] 17 features extracted"
        )

        print(
            "  [PASS] Multiple trajectories predicted"
        )

        print(
            "  [PASS] Failure probabilities valid"
        )

        print(
            "  [PASS] Local feature contributions generated"
        )

        print(
            "  [PASS] Human-readable explanations generated"
        )

    finally:

        if physics_client is not None:

            p.disconnect(
                physics_client
            )


if __name__ == "__main__":
    main()