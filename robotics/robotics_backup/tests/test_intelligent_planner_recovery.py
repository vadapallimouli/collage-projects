"""
INTELLIGENT PLANNER FAILURE + RECOVERY TEST

Pipeline
--------
MπNets problem
    ↓
PyBullet
    ↓
Panda robot
    ↓
Environment + obstacles
    ↓
Collision checker
    ↓
Goal configuration
    ↓
Many candidate trajectories
    ↓
ML failure prediction
    ↓
Physical validation
    ↓
Deliberate failure
    ↓
Recoverability analysis
    ↓
Recovery trajectory selection
    ↓
Final physical verification
"""

import os
import sys
import pickle

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from robot.panda_robot import PandaRobot

from environment.environment import Environment

from analysis.collision_checker import CollisionChecker

from analysis.clearance_analyzer import ClearanceAnalyzer

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


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "mpinets",
    "both_solvable_problems.pkl",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "failure_prediction",
    "models",
    "xgboost_localized.joblib",
)

FEATURE_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "failure_prediction",
    "models",
    "feature_names_localized.json",
)


# ============================================================
# CONFIGURATION
# ============================================================

ENVIRONMENT_NAME = "tabletop"

PROBLEM_SET_NAME = "task_oriented"

PROBLEM_INDEX = 0

NUM_CANDIDATES = 10

PLANNER_NUM_SAMPLES = 700

PLANNER_K_NEIGHBORS = 15

PLANNER_EDGE_POINTS = 25

RANDOM_SEED = 42


# ============================================================
# PRINT HELPERS
# ============================================================

def header(title):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def section(number, title):

    print()
    print(
        f"STEP {number}: {title}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_problem():

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            "MπNets dataset not found:\n"
            f"{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "rb",
    ) as file:

        dataset = pickle.load(
            file
        )

    print(
        "Dataset loaded successfully."
    )

    print(
        "Environments:",
        list(dataset.keys()),
    )

    problem = (
        dataset[
            ENVIRONMENT_NAME
        ][
            PROBLEM_SET_NAME
        ][
            PROBLEM_INDEX
        ]
    )

    print(
        "Environment:",
        ENVIRONMENT_NAME,
    )

    print(
        "Problem type:",
        type(problem).__name__,
    )

    return problem


# ============================================================
# CREATE PYBULLET
# ============================================================

def create_pybullet():

    client_id = p.connect(
        p.DIRECT
    )

    if client_id < 0:

        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81,
        physicsClientId=client_id,
    )

    p.loadURDF(
        "plane.urdf",
        physicsClientId=client_id,
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
        physicsClientId=client_id,
    )

    panda_robot = PandaRobot(
        robot_id
    )

    return (
        client_id,
        panda_robot,
    )


# ============================================================
# FIND GOAL
# ============================================================

def find_goal_configuration(
    panda_robot,
    environment,
    collision_checker,
    client_id,
):

    print(
        "\nGenerating IK goal configurations..."
    )

    goal_generator = GoalConfigurationGenerator(
        panda_robot=panda_robot,
        environment=environment,
        target_position=environment.target_position,
        target_half_extents=environment.target_half_extents,
        client_id=client_id,
    )

    # --------------------------------------------------------
    # Try progressively more IK solutions.
    # --------------------------------------------------------

    search_settings = [
        (30, 30),
        (60, 60),
        (100, 100),
    ]

    all_solutions = []

    for num_solutions, num_rest_poses in (
        search_settings
    ):

        print(
            f"Trying {num_solutions} IK solutions "
            f"from {num_rest_poses} rest poses..."
        )

        solutions = (
            goal_generator.solve_multiple(
                num_solutions=num_solutions,
                num_rest_poses=num_rest_poses,
                seed=RANDOM_SEED,
            )
        )

        all_solutions.extend(
            solutions
        )

        print(
            "Generated:",
            len(solutions),
        )

        # ----------------------------------------------------
        # Check every solution physically.
        # ----------------------------------------------------

        for index, solution in enumerate(
            solutions,
            start=1,
        ):

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
                    "\nCollision-free goal found."
                )

                print(
                    "Goal configuration:"
                )

                print(
                    np.round(
                        configuration,
                        6,
                    )
                )

                return configuration

    # --------------------------------------------------------
    # If the sampled IK solutions did not produce a safe goal,
    # report the actual situation instead of silently creating
    # an invalid goal.
    # --------------------------------------------------------

    print(
        "\nWARNING: No collision-free IK goal "
        "was found from the sampled rest poses."
    )

    print(
        "Total IK solutions checked:",
        len(all_solutions),
    )

    # --------------------------------------------------------
    # We can still choose the solution with the smallest
    # position error, but clearly mark it as a fallback.
    # --------------------------------------------------------

    if not all_solutions:

        raise RuntimeError(
            "No IK solutions were generated."
        )

    all_solutions.sort(
        key=lambda item: float(
            item.get(
                "position_error",
                np.inf,
            )
        )
    )

    fallback_goal = np.asarray(
        all_solutions[0]["configuration"],
        dtype=float,
    )

    fallback_validation = (
        collision_checker.check_configuration(
            fallback_goal
        )
    )

    print(
        "\nFallback IK goal:"
    )

    print(
        np.round(
            fallback_goal,
            6,
        )
    )

    print(
        "Fallback collision:",
        fallback_validation[
            "collision"
        ],
    )

    # --------------------------------------------------------
    # The recovery test requires a usable planning target.
    # If the fallback itself is in collision, fail here with
    # a useful diagnostic rather than later inside PRM.
    # --------------------------------------------------------

    if fallback_validation[
        "collision"
    ]:

        raise RuntimeError(
            "\nCould not obtain a collision-free "
            "goal configuration for this scene.\n"
            "The current IK sampling strategy does not "
            "produce a safe goal for this particular "
            "MπNets problem."
        )

    return fallback_goal


# ============================================================
# MAIN TEST
# ============================================================

def main():

    header(
        "INTELLIGENT PLANNER FAILURE + RECOVERY TEST"
    )

    physics_client = None

    try:

        # ====================================================
        # STEP 1
        # ====================================================

        section(
            1,
            "LOADING MπNets DATASET",
        )

        problem = load_problem()

        # ====================================================
        # STEP 2
        # ====================================================

        section(
            2,
            "STARTING PYBULLET",
        )

        (
            physics_client,
            panda_robot,
        ) = create_pybullet()

        print(
            "PyBullet client:",
            physics_client,
        )

        # ====================================================
        # STEP 3
        # ====================================================

        section(
            3,
            "LOADING PANDA ROBOT",
        )

        print(
            "Robot ID:",
            panda_robot.robot_id,
        )

        print(
            "DOF:",
            panda_robot.DOF,
        )

        # ====================================================
        # STEP 4
        # ====================================================

        section(
            4,
            "CREATING ENVIRONMENT",
        )

        # IMPORTANT:
        #
        # Environment.from_mpinets_problem()
        # accepts ONLY the MπNets problem.
        #
        # Obstacles are created separately.

        environment = (
            Environment.from_mpinets_problem(
                problem
            )
        )

        obstacle_ids = (
            environment.create_pybullet_obstacles(
                physics_client
            )
        )

        print(
            "Environment created."
        )

        print(
            "Start configuration:",
            np.round(
                environment.start_configuration,
                4,
            ),
        )

        print(
            "Target position:",
            np.round(
                environment.target_position,
                4,
            ),
        )

        print(
            "Number of obstacles:",
            len(obstacle_ids),
        )

        # ====================================================
        # STEP 5
        # ====================================================

        section(
            5,
            "CREATING COLLISION CHECKER",
        )

        collision_checker = CollisionChecker(
            panda_robot,
            environment,
            physics_client,
        )

        print(
            "Collision checker ready."
        )

        # ====================================================
        # STEP 6
        # ====================================================

        section(
            6,
            "CREATING PLANNING MODULES",
        )

        trajectory_generator = (
            TrajectoryGenerator(
                panda_robot
            )
        )

        clearance_analyzer = (
            ClearanceAnalyzer(
                panda_robot,
                environment,
                physics_client,
            )
        )

        feature_extractor = (
            TrajectoryFeatureExtractor(
                collision_checker,
                clearance_analyzer,
                trajectory_generator,
            )
        )

        recoverability_analyzer = (
            RecoverabilityAnalyzer(
                collision_checker,
                trajectory_generator,
            )
        )

        planner = ObstacleAwarePlanner(
            panda_robot=panda_robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=PLANNER_NUM_SAMPLES,
            k_neighbors=PLANNER_K_NEIGHBORS,
            edge_points=PLANNER_EDGE_POINTS,
            seed=RANDOM_SEED,
        )

        failure_predictor = FailurePredictor(
            model_path=MODEL_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
        )

        intelligent_planner = (
            IntelligentMotionPlanner(
                planner=planner,
                feature_extractor=feature_extractor,
                failure_predictor=failure_predictor,
                recoverability_analyzer=recoverability_analyzer,
                collision_checker=collision_checker,
            )
        )

        print(
            "All planning modules created."
        )

        # ====================================================
        # STEP 7
        # ====================================================

        section(
            7,
            "CHECKING START CONFIGURATION",
        )

        start = np.asarray(
            problem.q0,
            dtype=float,
        )

        start_validation = (
            collision_checker.check_configuration(
                start
            )
        )

        print(
            "Start collision:",
            start_validation[
                "collision"
            ],
        )

        print(
            "Start joint-limit violation:",
            start_validation[
                "joint_limit_violation"
            ]
            if
            "joint_limit_violation"
            in start_validation
            else
            False,
        )

        if start_validation[
            "collision"
        ]:

            raise RuntimeError(
                "MπNets start configuration "
                "is in collision."
            )

        print(
            "[PASS] Start configuration is valid."
        )

        # ====================================================
        # STEP 8
        # ====================================================

        section(
            8,
            "GENERATING GOAL CONFIGURATION",
        )

        goal = find_goal_configuration(
            panda_robot=panda_robot,
            environment=environment,
            collision_checker=collision_checker,
            client_id=physics_client,
        )

        goal_validation = (
            collision_checker.check_configuration(
                goal
            )
        )

        print(
            "\nSelected goal collision:",
            goal_validation[
                "collision"
            ],
        )

        if goal_validation[
            "collision"
        ]:

            raise RuntimeError(
                "Selected goal is in collision."
            )

        print(
            "[PASS] Collision-free goal selected."
        )

        # ====================================================
        # STEP 9
        # ====================================================

        section(
            9,
            "GENERATING MANY CANDIDATE TRAJECTORIES",
        )

        generated = (
            intelligent_planner.generate_candidates(
                start,
                goal,
                NUM_CANDIDATES,
            )
        )

        # ----------------------------------------------------
        # Handle both possible return styles.
        # ----------------------------------------------------

        if isinstance(
            generated,
            dict,
        ):

            trajectories = generated.get(
                "trajectories",
                generated.get(
                    "candidates",
                    [],
                ),
            )

        else:

            trajectories = generated

        trajectories = [
            np.asarray(
                trajectory,
                dtype=float,
            )
            for trajectory in trajectories
        ]

        print(
            "Candidates generated:",
            len(trajectories),
        )

        if len(trajectories) == 0:

            raise RuntimeError(
                "No candidate trajectories were generated."
            )

        for index, trajectory in enumerate(
            trajectories,
            start=1,
        ):

            print(
                f"  Candidate {index}: "
                f"{trajectory.shape}"
            )

        # ====================================================
        # STEP 10
        # ====================================================

        section(
            10,
            "ML FAILURE PREDICTION",
        )

        predictions = (
            intelligent_planner.predict_candidates(
                trajectories
            )
        )

        if len(predictions) != len(
            trajectories
        ):

            raise RuntimeError(
                "Prediction count does not match "
                "trajectory count."
            )

        for index, prediction in enumerate(
            predictions,
            start=1,
        ):

            success_probability = float(
                prediction.get(
                    "success_probability",
                    0.0,
                )
            )

            failure_probability = float(
                prediction.get(
                    "failure_probability",
                    0.0,
                )
            )

            predicted_failure = prediction.get(
                "predicted_failure",
                prediction.get(
                    "failure",
                    0,
                ),
            )

            label = (
                "FAILURE"
                if int(
                    predicted_failure
                ) == 1
                else "SUCCESS"
            )

            print(
                f"Candidate {index}: "
                f"{label} | "
                f"success="
                f"{success_probability:.4f} | "
                f"failure="
                f"{failure_probability:.4f}"
            )

        print(
            "[PASS] ML predictions generated."
        )

        # ====================================================
        # STEP 11
        # ====================================================

        section(
            11,
            "PHYSICAL VALIDATION",
        )

        validations = (
            intelligent_planner.validate_candidates(
                trajectories
            )
        )

        if len(validations) != len(
            trajectories
        ):

            raise RuntimeError(
                "Validation count does not match "
                "trajectory count."
            )

        physically_feasible = 0

        for index, validation in enumerate(
            validations,
            start=1,
        ):

            valid = bool(
                validation.get(
                    "valid",
                    validation.get(
                        "collision_free",
                        False,
                    ),
                )
            )

            collision_count = int(
                validation.get(
                    "collision_count",
                    0,
                )
            )

            joint_limit_count = int(
                validation.get(
                    "joint_limit_violation_count",
                    0,
                )
            )

            if valid:

                physically_feasible += 1

            print(
                f"Candidate {index}: "
                f"{'SAFE' if valid else 'FAILED'} | "
                f"collisions={collision_count} | "
                f"joint-limit="
                f"{joint_limit_count}"
            )

        print(
            "\nPhysically feasible:",
            f"{physically_feasible}/"
            f"{len(trajectories)}",
        )

        if physically_feasible == 0:

            raise RuntimeError(
                "No physically feasible candidate "
                "trajectories were generated."
            )

        print(
            "[PASS] Physical validation completed."
        )

        # ====================================================
        # STEP 12
        # ====================================================

        section(
            12,
            "COMBINING ML + PHYSICAL RESULTS",
        )

        combined = (
            intelligent_planner.combine_results(
                trajectories,
                predictions,
                validations,
            )
        )

        print(
            "Combined candidates:",
            len(combined),
        )

        print(
            "[PASS] Results combined."
        )

        # ====================================================
        # STEP 13
        # ====================================================

        section(
            13,
            "RECOVERABILITY ANALYSIS",
        )

        recoverability_result = (
            recoverability_analyzer.analyze_recoverability(
                trajectories
            )
        )

        print(
            "Recoverability analysis completed."
        )

        if isinstance(
            recoverability_result,
            dict,
        ):

            for key in [
                "total_candidates",
                "feasible_candidates",
                "failed_candidates",
                "recoverable_candidates",
                "unrecoverable_candidates",
            ]:

                if key in recoverability_result:

                    print(
                        f"  {key}: "
                        f"{recoverability_result[key]}"
                    )

        print(
            "[PASS] Recoverability analysis completed."
        )

        # ====================================================
        # STEP 14
        # ====================================================

        section(
            14,
            "SELECTING FINAL TRAJECTORY",
        )

        selected = (
            intelligent_planner.select_final_trajectory(
                combined
            )
        )

        if not isinstance(
            selected,
            dict,
        ):

            raise RuntimeError(
                "Final selection did not return "
                "a dictionary."
            )

        if not selected.get(
            "success",
            False,
        ):

            raise RuntimeError(
                "No physically feasible trajectory "
                "could be selected."
            )

        selected_index = int(
            selected[
                "selected_index"
            ]
        )

        selected_trajectory = np.asarray(
            selected[
                "trajectory"
            ],
            dtype=float,
        )

        print(
            "Selected candidate:",
            selected_index + 1,
        )

        print(
            "ML success probability:",
            f"{float(selected['success_probability']):.4f}",
        )

        print(
            "ML failure probability:",
            f"{float(selected['failure_probability']):.4f}",
        )

        print(
            "[PASS] Final trajectory selected."
        )

        # ====================================================
        # STEP 15
        # ====================================================

        section(
            15,
            "FINAL PHYSICAL VERIFICATION",
        )

        final_validation = (
            collision_checker.check_trajectory(
                selected_trajectory
            )
        )

        print(
            "Collision-free         :",
            final_validation[
                "collision_free"
            ],
        )

        print(
            "Collision count        :",
            final_validation[
                "collision_count"
            ],
        )

        print(
            "Self-collision count   :",
            final_validation[
                "self_collision_count"
            ],
        )

        print(
            "Obstacle collisions    :",
            final_validation[
                "obstacle_collision_count"
            ],
        )

        print(
            "Joint-limit violations :",
            final_validation[
                "joint_limit_violation_count"
            ],
        )

        if not final_validation[
            "collision_free"
        ]:

            raise RuntimeError(
                "Selected trajectory failed "
                "final physical verification."
            )

        if final_validation[
            "joint_limit_violation_count"
        ] != 0:

            raise RuntimeError(
                "Selected trajectory violates "
                "joint limits."
            )

        print(
            "[PASS] Final trajectory physically verified."
        )

        # ====================================================
        # STEP 16
        # ====================================================
        #
        # INTENTIONAL FAILURE TEST
        #
        # This is the actual recovery demonstration.
        #
        # Take a valid trajectory and deliberately corrupt
        # one configuration by setting J1 = 10 radians.
        # ====================================================

        section(
            16,
            "CREATING DELIBERATE FAILED TRAJECTORY",
        )

        failed_candidate = (
            trajectories[0].copy()
        )

        midpoint = (
            len(
                failed_candidate
            )
            // 2
        )

        failed_candidate[
            midpoint,
            0,
        ] = 10.0

        print(
            "Modified trajectory midpoint:",
            midpoint,
        )

        print(
            "Modified J1:",
            failed_candidate[
                midpoint,
                0,
            ],
        )

        # ====================================================
        # STEP 17
        # ====================================================

        section(
            17,
            "VERIFYING DELIBERATE FAILURE",
        )

        failed_validation = (
            collision_checker.check_trajectory(
                failed_candidate
            )
        )

        print(
            "Collision-free:",
            failed_validation[
                "collision_free"
            ],
        )

        print(
            "Collision count:",
            failed_validation[
                "collision_count"
            ],
        )

        print(
            "Self-collision count:",
            failed_validation[
                "self_collision_count"
            ],
        )

        print(
            "Obstacle collision count:",
            failed_validation[
                "obstacle_collision_count"
            ],
        )

        print(
            "Joint-limit violation count:",
            failed_validation[
                "joint_limit_violation_count"
            ],
        )

        if (
            failed_validation[
                "collision_free"
            ]
            and
            failed_validation[
                "joint_limit_violation_count"
            ] == 0
        ):

            raise RuntimeError(
                "Deliberate failure was not detected."
            )

        print(
            "[PASS] Deliberate failure detected."
        )

        # ====================================================
        # STEP 18
        # ====================================================

        section(
            18,
            "RECOVERABILITY OF FAILED TRAJECTORY",
        )

        recovery_candidates = (
            list(trajectories)
            + [
                failed_candidate
            ]
        )

        recovery_analysis = (
            recoverability_analyzer.analyze_recoverability(
                recovery_candidates
            )
        )

        print(
            "Recovery analysis:"
        )

        print(
            recovery_analysis
        )

        # ====================================================
        # STEP 19
        # ====================================================

        section(
            19,
            "FINDING RECOVERY ALTERNATIVES",
        )

        failed_candidate_index = (
            len(
                recovery_candidates
            )
            - 1
        )

        alternatives = recoverability_analyzer.find_alternatives(
    failed_candidate_index=failed_candidate_index,
    candidate_results=recovery_analysis["candidate_results"],
)

        print(
            "Failed candidate index:",
            failed_candidate_index,
        )

        print(
            "Recovery alternatives:",
            len(alternatives),
        )

        for alternative in alternatives:

            print(
                alternative
            )

        if len(alternatives) == 0:

            raise RuntimeError(
                "No recovery alternatives found."
            )

        print(
            "[PASS] Recovery alternatives found."
        )

        # ====================================================
        # STEP 20
        # ====================================================

        section(
            20,
            "SELECTING RECOVERY TRAJECTORY",
        )

        recovery_result = (
            recoverability_analyzer.select_recovery_trajectory(
                recovery_analysis,
                failed_candidate_index,
            )
        )

        print(
            "Recovery selection:"
        )

        print(
            recovery_result
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # select_recovery_trajectory()
        # returns a dictionary containing:
        #
        # candidate_index
        # trajectory
        # validation
        # trajectory_distance
        #
        # It does NOT necessarily contain "success".
        # ----------------------------------------------------

        if not isinstance(
            recovery_result,
            dict,
        ):

            raise RuntimeError(
                "Recovery selection did not "
                "return a dictionary."
            )

        if "trajectory" not in (
            recovery_result
        ):

            raise RuntimeError(
                "No recovery trajectory was selected."
            )

        recovery_trajectory = np.asarray(
            recovery_result[
                "trajectory"
            ],
            dtype=float,
        )

        recovery_index = (
            recovery_result.get(
                "candidate_index"
            )
        )

        print(
            "\nSelected recovery candidate:",
            int(
                recovery_index
            ) + 1,
        )

        print(
            "Recovery trajectory shape:",
            recovery_trajectory.shape,
        )

        print(
            "Trajectory distance:",
            f"{float(recovery_result.get('trajectory_distance', 0.0)):.6f}",
        )

        print(
            "[PASS] Recovery trajectory selected."
        )

        # ====================================================
        # STEP 21
        # ====================================================

        section(
            21,
            "VERIFYING RECOVERY TRAJECTORY",
        )

        recovery_validation = (
            collision_checker.check_trajectory(
                recovery_trajectory
            )
        )

        print()

        print(
            "Collision-free         :",
            recovery_validation[
                "collision_free"
            ],
        )

        print(
            "Collision count        :",
            recovery_validation[
                "collision_count"
            ],
        )

        print(
            "Self-collision count   :",
            recovery_validation[
                "self_collision_count"
            ],
        )

        print(
            "Obstacle collisions    :",
            recovery_validation[
                "obstacle_collision_count"
            ],
        )

        print(
            "Joint-limit violations :",
            recovery_validation[
                "joint_limit_violation_count"
            ],
        )

        if not recovery_validation[
            "collision_free"
        ]:

            raise RuntimeError(
                "Selected recovery trajectory "
                "is not collision-free."
            )

        if recovery_validation[
            "joint_limit_violation_count"
        ] != 0:

            raise RuntimeError(
                "Selected recovery trajectory "
                "violates joint limits."
            )

        print(
            "[PASS] Recovery trajectory "
            "physically verified."
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        print()
        print("=" * 72)
        print("FINAL TEST RESULT")
        print("=" * 72)

        print(
            "[PASS] Dataset loaded"
        )

        print(
            "[PASS] PyBullet initialized"
        )

        print(
            "[PASS] Panda robot loaded"
        )

        print(
            "[PASS] Environment created"
        )

        print(
            "[PASS] Collision checker created"
        )

        print(
            "[PASS] Planning modules created"
        )

        print(
            "[PASS] Collision-free goal generated"
        )

        print(
            "[PASS] Normal candidates generated"
        )

        print(
            "[PASS] ML failure prediction completed"
        )

        print(
            "[PASS] Physical validation completed"
        )

        print(
            "[PASS] Deliberate failure detected"
        )

        print(
            "[PASS] Recoverability analysis completed"
        )

        print(
            "[PASS] Recovery alternatives found"
        )

        print(
            "[PASS] Recovery trajectory selected"
        )

        print(
            "[PASS] Recovery trajectory physically verified"
        )

        print()
        print(
            "INTELLIGENT PLANNER FAILURE + "
            "RECOVERY TEST PASSED"
        )

        print("=" * 72)

    finally:

        if (
            physics_client is not None
            and p.isConnected(
                physics_client
            )
        ):

            p.disconnect(
                physics_client
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()