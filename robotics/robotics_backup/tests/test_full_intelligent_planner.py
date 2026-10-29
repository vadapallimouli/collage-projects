"""
Full Intelligent Motion Planner Integration Test
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

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ============================================================
# IMPORTS
# ============================================================

from robot.panda_robot import PandaRobot
from environment.environment import Environment

from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer
from analysis.feature_extractor import TrajectoryFeatureExtractor
from analysis.recoverability_analyzer import RecoverabilityAnalyzer
from analysis.intelligent_planner import IntelligentMotionPlanner

from planning.trajectory_generator import TrajectoryGenerator
from planning.goal_configuration import GoalConfigurationGenerator
from planning.obstacle_aware_planner import ObstacleAwarePlanner

from prediction.failure_predictor import FailurePredictor


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = os.path.join(
    ROOT,
    "data",
    "mpinets",
    "both_solvable_problems.pkl",
)

NUM_CANDIDATES = 10


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FULL INTELLIGENT MOTION PLANNING TEST")
    print("=" * 70)

    client = None

    try:

        # ----------------------------------------------------
        # 1. LOAD DATASET
        # ----------------------------------------------------

        print("\n[1] Loading MπNets dataset...")

        with open(
            DATASET_PATH,
            "rb",
        ) as f:

            dataset = pickle.load(f)

        problem = (
            dataset[
                "tabletop"
            ][
                "task_oriented"
            ][
                0
            ]
        )

        print(
            "    Environment : tabletop"
        )

        print(
            "    Problem set : task_oriented"
        )

        print(
            "    Problem     : 0"
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 2. PYBULLET
        # ----------------------------------------------------

        print("\n[2] Starting PyBullet...")

        client = p.connect(
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

        robot = PandaRobot(
            robot_id
        )

        print(
            f"    Robot DOF: {robot.DOF}"
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 3. ENVIRONMENT
        # ----------------------------------------------------

        print(
            "\n[3] Creating environment..."
        )

        environment = (
            Environment.from_mpinets_problem(
                problem
            )
        )

        obstacle_ids = (
            environment.create_pybullet_obstacles(
                client
            )
        )

        print(
            f"    Obstacles: {len(obstacle_ids)}"
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 4. MODULES
        # ----------------------------------------------------

        print(
            "\n[4] Initializing modules..."
        )

        collision_checker = CollisionChecker(
            panda_robot=robot,
            environment=environment,
            client_id=client,
        )

        clearance_analyzer = ClearanceAnalyzer(
            panda_robot=robot,
            environment=environment,
            client_id=client,
        )

        trajectory_generator = TrajectoryGenerator(
            panda_robot=robot
        )

        feature_extractor = (
            TrajectoryFeatureExtractor(
                collision_checker=collision_checker,
                clearance_analyzer=clearance_analyzer,
                trajectory_generator=trajectory_generator,
            )
        )

        recoverability_analyzer = (
            RecoverabilityAnalyzer(
                collision_checker=collision_checker,
                trajectory_generator=trajectory_generator,
            )
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 5. START
        # ----------------------------------------------------

        print(
            "\n[5] Checking start configuration..."
        )

        start = np.asarray(
            environment.start_configuration,
            dtype=float,
        )

        start_result = (
            collision_checker.check_configuration(
                start
            )
        )

        print(
            f"    Collision: "
            f"{start_result['collision']}"
        )

        if start_result["collision"]:
            raise RuntimeError(
                "Start configuration is in collision."
            )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 6. GOAL
        # ----------------------------------------------------

        print(
            "\n[6] Generating goal configuration..."
        )

        goal_generator = (
            GoalConfigurationGenerator(
                panda_robot=robot,
                environment=environment,
                target_position=environment.target_position,
                target_half_extents=environment.target_half_extents,
                client_id=client,
            )
        )

        solutions = (
            goal_generator.solve_multiple(
                num_solutions=10,
                num_rest_poses=20,
                seed=42,
            )
        )

        goal = None

        for solution in solutions:

            q = np.asarray(
                solution["configuration"],
                dtype=float,
            )

            result = (
                collision_checker.check_configuration(
                    q
                )
            )

            if not result["collision"]:

                goal = q

                break

        if goal is None:
            raise RuntimeError(
                "Could not find collision-free goal."
            )

        print(
            "    Collision-free goal found."
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 7. OBSTACLE-AWARE PLANNER
        # ----------------------------------------------------

        print(
            "\n[7] Creating obstacle-aware planner..."
        )

        planner = ObstacleAwarePlanner(
            panda_robot=robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=700,
            k_neighbors=15,
            edge_points=25,
            seed=42,
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 8. ML MODEL
        # ----------------------------------------------------

        print(
            "\n[8] Loading XGBoost failure predictor..."
        )

        predictor = FailurePredictor()

        info = predictor.get_model_info()

        print(
            f"    Features: "
            f"{len(info['feature_names'])}"
        )

        print(
            f"    Model: "
            f"{info.get('model_type', 'XGBoost')}"
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 9. INTELLIGENT PLANNER
        # ----------------------------------------------------

        print(
            "\n[9] Creating intelligent planner..."
        )

        intelligent = IntelligentMotionPlanner(
            planner=planner,
            feature_extractor=feature_extractor,
            failure_predictor=predictor,
            recoverability_analyzer=recoverability_analyzer,
            collision_checker=collision_checker,
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 10. CANDIDATES
        # ----------------------------------------------------

        print(
            "\n[10] Generating candidate trajectories..."
        )

        trajectories = (
            intelligent.generate_candidates(
                start,
                goal,
                NUM_CANDIDATES,
            )
        )

        if trajectories is None:
            raise RuntimeError(
                "generate_candidates returned None."
            )

        trajectories = list(
            trajectories
        )

        print(
            f"    Generated: "
            f"{len(trajectories)}"
        )

        if len(trajectories) == 0:
            raise RuntimeError(
                "No candidate trajectories generated."
            )

        for i, trajectory in enumerate(
            trajectories,
            start=1,
        ):

            print(
                f"    Candidate {i}: "
                f"{np.asarray(trajectory).shape}"
            )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 11. ML PREDICTION
        # ----------------------------------------------------

        print(
            "\n[11] Running ML failure prediction..."
        )

        predictions = (
            intelligent.predict_candidates(
                trajectories
            )
        )

        if len(predictions) != len(
            trajectories
        ):

            raise RuntimeError(
                "Prediction count mismatch."
            )

        for i, prediction in enumerate(
            predictions,
            start=1,
        ):

            print(
                f"    Candidate {i}: "
                f"{prediction['predicted_class']} | "
                f"failure="
                f"{float(prediction['failure_probability']):.4f}"
            )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 12. PHYSICAL VALIDATION
        # ----------------------------------------------------

        print(
            "\n[12] Physically validating candidates..."
        )

        validations = (
            intelligent.validate_candidates(
                trajectories
            )
        )

        feasible = 0

        for i, validation in enumerate(
            validations,
            start=1,
        ):

            valid = bool(
                validation["valid"]
            )

            if valid:
                feasible += 1

            print(
                f"    Candidate {i}: "
                f"{'SAFE' if valid else 'FAILED'}"
            )

        print(
            f"    Physically feasible: "
            f"{feasible}/{len(trajectories)}"
        )

        if feasible == 0:
            raise RuntimeError(
                "No physically feasible candidates."
            )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 13. COMBINE
        # ----------------------------------------------------

        print(
            "\n[13] Combining ML and physical results..."
        )

        combined = (
            intelligent.combine_results(
                trajectories=trajectories,
                predictions=predictions,
                validations=validations,
            )
        )

        print(
            f"    Combined results: "
            f"{len(combined)}"
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 14. RECOVERABILITY
        # ----------------------------------------------------

        print(
            "\n[14] Analyzing recoverability..."
        )

        recoverability = (
            intelligent.analyze_recoverability(
                trajectories
            )
        )

        print(
            "    Recoverability analysis complete."
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 15. FINAL SELECTION
        # ----------------------------------------------------

        print(
            "\n[15] Selecting final trajectory..."
        )

        selected = (
            intelligent.select_final_trajectory(
                combined
            )
        )

        if selected is None:
            raise RuntimeError(
                "No final trajectory selected."
            )

        selected_index = (
            selected["selected_index"]
        )

        selected_trajectory = (
            selected["trajectory"]
        )

        print(
            f"    Selected candidate: "
            f"{int(selected_index) + 1}"
        )

        print(
            f"    ML success probability: "
            f"{float(selected['success_probability']):.4f}"
        )

        print(
            f"    ML failure probability: "
            f"{float(selected['failure_probability']):.4f}"
        )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # 16. FINAL PHYSICAL CHECK
        # ----------------------------------------------------

        print(
            "\n[16] Final physical verification..."
        )

        final_check = (
            collision_checker.check_trajectory(
                selected_trajectory
            )
        )

        print(
            f"    Collision-free: "
            f"{final_check['collision_free']}"
        )

        print(
            f"    Collision count: "
            f"{final_check['collision_count']}"
        )

        print(
            f"    Joint-limit violations: "
            f"{final_check['joint_limit_violation_count']}"
        )

        if not final_check[
            "collision_free"
        ]:

            raise RuntimeError(
                "Final trajectory is not collision-free."
            )

        if final_check[
            "joint_limit_violation_count"
        ] != 0:

            raise RuntimeError(
                "Final trajectory violates joint limits."
            )

        print(
            "    [PASS]"
        )


        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "INTELLIGENT MOTION PLANNING TEST PASSED"
        )
        print("=" * 70)

        print(
            f"Candidates generated : "
            f"{len(trajectories)}"
        )

        print(
            f"Physically feasible  : "
            f"{feasible}"
        )

        print(
            f"Selected candidate   : "
            f"{int(selected_index) + 1}"
        )

        print(
            "Final trajectory safe: YES"
        )

        print()
        print(
            "ALL TESTS PASSED."
        )


    finally:

        if (
            client is not None
            and p.isConnected(client)
        ):

            p.disconnect(
                client
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()