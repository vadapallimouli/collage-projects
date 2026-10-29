"""
Test the ML failure predictor against PyBullet collision ground truth.

Pipeline:

MπNets problem
    ↓
Panda + environment
    ↓
Obstacle-aware planner
    ↓
5 candidate trajectories
    ↓
Feature extraction
    ↓
Random Forest prediction
    ↓
Compare against CollisionChecker ground truth
    ↓
Human-readable explanation
"""

from pathlib import Path
import sys
import pickle

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from robot.panda_robot import PandaRobot

from environment.environment import Environment

from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer
from analysis.feature_extractor import TrajectoryFeatureExtractor

from planning.trajectory_generator import TrajectoryGenerator
from planning.obstacle_aware_planner import ObstacleAwarePlanner

from prediction.failure_predictor import FailurePredictor


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAILURE PREDICTOR TEST")
    print("=" * 70)

    client_id = None

    try:

        # ====================================================
        # 1. LOAD DATASET
        # ====================================================

        if not DATASET_PATH.exists():

            raise FileNotFoundError(
                f"MπNets dataset not found:\n"
                f"{DATASET_PATH}"
            )

        print("\nLoading MπNets dataset...")

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
            ][0]
        )

        print(
            "Selected problem: "
            "tabletop / task_oriented / 0"
        )

        # ====================================================
        # 2. START PYBULLET
        # ====================================================

        client_id = p.connect(
            p.DIRECT
        )

        if client_id < 0:

            raise RuntimeError(
                "Could not connect to PyBullet."
            )

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        # Plane
        p.loadURDF(
            "plane.urdf"
        )

        # Panda
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
        )

        robot = PandaRobot(
            robot_id
        )

        # ====================================================
        # 3. ENVIRONMENT
        # ====================================================

        environment = (
            Environment.from_mpinets_problem(
                problem
            )
        )

        obstacle_ids = (
            environment.create_pybullet_obstacles(
                client_id
            )
        )

        print(
            f"Obstacles created: "
            f"{len(obstacle_ids)}"
        )

        # ====================================================
        # 4. ANALYSIS MODULES
        # ====================================================

        collision_checker = CollisionChecker(
            panda_robot=robot,
            environment=environment,
            client_id=client_id,
        )

        clearance_analyzer = ClearanceAnalyzer(
            panda_robot=robot,
            environment=environment,
            client_id=client_id,
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

        # ====================================================
        # 5. LOAD FAILURE PREDICTOR
        # ====================================================

        predictor = FailurePredictor(
            feature_extractor=feature_extractor
        )

        print(
            "\nRandom Forest model loaded."
        )

        # ====================================================
        # 6. START CONFIGURATION
        # ====================================================

        start = np.asarray(
            environment.start_configuration,
            dtype=float,
        )

        print(
            "\nStart configuration:"
        )

        print(
            np.array2string(
                start,
                precision=6,
                suppress_small=True,
            )
        )

        start_check = (
            collision_checker.check_configuration(
                start
            )
        )

        print(
            "\nStart collision: "
            f"{start_check['collision']}"
        )

        if start_check["collision"]:

            raise RuntimeError(
                "Dataset start configuration "
                "is already in collision."
            )

        # ====================================================
        # 7. KNOWN COLLISION-FREE GOAL
        # ====================================================

        goal = np.array(
            [
                0.969308251,
                0.776347574,
                -2.325633062,
                -2.457559241,
                -0.022686111,
                2.261069429,
                -2.341595461,
            ],
            dtype=float,
        )

        goal_check = (
            collision_checker.check_configuration(
                goal
            )
        )

        print(
            "\nKnown goal collision: "
            f"{goal_check['collision']}"
        )

        if goal_check["collision"]:

            raise RuntimeError(
                "Known goal configuration is "
                "not collision-free."
            )

        print(
            "Known collision-free goal verified."
        )

        # ====================================================
        # 8. OBSTACLE-AWARE PLANNER
        # ====================================================

        planner = ObstacleAwarePlanner(
            panda_robot=robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=700,
            k_neighbors=15,
            edge_points=25,
            seed=42,
        )

        # ====================================================
        # 9. GENERATE CANDIDATES
        # ====================================================

        print("\n")
        print("=" * 70)
        print("GENERATING CANDIDATE TRAJECTORIES")
        print("=" * 70)

        candidates = planner.plan_multiple(
            start,
            goal,
            5,
        )

        if not candidates:

            raise RuntimeError(
                "No candidate trajectories were generated."
            )

        print(
            f"\nGenerated "
            f"{len(candidates)} "
            f"candidate trajectories."
        )

        # ====================================================
        # 10. PREDICTION VS GROUND TRUTH
        # ====================================================

        print("\n")
        print("=" * 70)
        print("ML PREDICTION VS PYBULLET GROUND TRUTH")
        print("=" * 70)

        total = 0
        correct = 0

        actual_successes = 0
        actual_failures = 0

        predicted_successes = 0
        predicted_failures = 0

        summary = []

        # ====================================================
        # 11. TEST EACH CANDIDATE
        # ====================================================

        for index, candidate in enumerate(
            candidates
        ):

            trajectory = candidate[
                "trajectory"
            ]

            total += 1

            # ------------------------------------------------
            # ACTUAL PYBULLET GROUND TRUTH
            # ------------------------------------------------
            #
            # CollisionChecker is the ground truth.
            # We intentionally do NOT use the ML features
            # to determine this value.
            # ------------------------------------------------

            collision_result = (
                collision_checker.check_trajectory(
                    trajectory
                )
            )

            actual_failure = int(
                not collision_result[
                    "collision_free"
                ]
            )

            # ------------------------------------------------
            # CLEARANCE INFORMATION
            # ------------------------------------------------

            clearance_result = (
                clearance_analyzer.calculate_trajectory_clearance(
                    trajectory
                )
            )

            minimum_clearance = (
                clearance_result[
                    "minimum_clearance"
                ]
            )

            # ------------------------------------------------
            # ML PREDICTION
            # ------------------------------------------------

            prediction = (
                predictor.predict_trajectory(
                    trajectory,
                    top_k=3,
                )
            )

            predicted_failure = int(
                prediction["prediction"]
            )

            # ------------------------------------------------
            # CORRECTNESS
            # ------------------------------------------------

            is_correct = (
                predicted_failure
                == actual_failure
            )

            if is_correct:
                correct += 1

            # ------------------------------------------------
            # CLASS COUNTS
            # ------------------------------------------------

            if actual_failure:
                actual_failures += 1
            else:
                actual_successes += 1

            if predicted_failure:
                predicted_failures += 1
            else:
                predicted_successes += 1

            # =================================================
            # DISPLAY
            # =================================================

            print("\n")
            print("-" * 70)
            print(
                f"CANDIDATE {index + 1}"
            )
            print("-" * 70)

            print(
                "Actual outcome       : "
                f"{'FAILURE' if actual_failure else 'SUCCESS'}"
            )

            print(
                "ML prediction        : "
                f"{prediction['label']}"
            )

            print(
                "Prediction correct   : "
                f"{'YES' if is_correct else 'NO'}"
            )

            print(
                "Failure probability  : "
                f"{prediction['failure_probability']:.4f}"
            )

            print(
                "Success probability  : "
                f"{prediction['success_probability']:.4f}"
            )

            print(
                "Collision count      : "
                f"{collision_result['collision_count']}"
            )

            print(
                "Obstacle collisions  : "
                f"{collision_result['obstacle_collision_count']}"
            )

            print(
                "Self collisions      : "
                f"{collision_result['self_collision_count']}"
            )

            print(
                "Minimum clearance    : "
                f"{minimum_clearance:.6f}"
            )

            # ------------------------------------------------
            # TRAJECTORY LENGTH
            # ------------------------------------------------

            joint_space_length = (
                trajectory_generator.calculate_joint_space_length(
                    trajectory
                )
            )

            print(
                "Joint-space length   : "
                f"{joint_space_length:.6f}"
            )

            # =================================================
            # EXPLANATION
            # =================================================

            print(
                "\nTop explanation factors:"
            )

            for rank, reason in enumerate(
                prediction["explanation"],
                start=1,
            ):

                print(
                    f"\n{rank}. "
                    f"{reason['feature']}"
                )

                print(
                    f"   Value: "
                    f"{reason['value']:.6f}"
                )

                print(
                    f"   Global importance: "
                    f"{reason['model_importance']:.4f}"
                )

                print(
                    f"   Explanation score: "
                    f"{reason['explanation_score']:.6f}"
                )

                print(
                    f"   Reason: "
                    f"{reason['reason']}"
                )

            # =================================================
            # STORE SUMMARY
            # =================================================

            summary.append(
                {
                    "candidate": index + 1,
                    "actual_failure": actual_failure,
                    "predicted_failure": predicted_failure,
                    "prediction_correct": is_correct,
                    "failure_probability": (
                        prediction[
                            "failure_probability"
                        ]
                    ),
                    "success_probability": (
                        prediction[
                            "success_probability"
                        ]
                    ),
                    "collision_count": (
                        collision_result[
                            "collision_count"
                        ]
                    ),
                    "obstacle_collision_count": (
                        collision_result[
                            "obstacle_collision_count"
                        ]
                    ),
                    "self_collision_count": (
                        collision_result[
                            "self_collision_count"
                        ]
                    ),
                    "minimum_clearance": (
                        minimum_clearance
                    ),
                    "joint_space_length": (
                        joint_space_length
                    ),
                }
            )

        # ====================================================
        # 12. FINAL SUMMARY
        # ====================================================

        print("\n")
        print("=" * 70)
        print("FINAL PREDICTION SUMMARY")
        print("=" * 70)

        print(
            f"\nTotal candidates      : "
            f"{total}"
        )

        print(
            f"Correct predictions   : "
            f"{correct}"
        )

        if total > 0:

            print(
                f"Prediction agreement  : "
                f"{correct / total:.2%}"
            )

        print(
            f"\nActual successes      : "
            f"{actual_successes}"
        )

        print(
            f"Actual failures       : "
            f"{actual_failures}"
        )

        print(
            f"Predicted successes   : "
            f"{predicted_successes}"
        )

        print(
            f"Predicted failures    : "
            f"{predicted_failures}"
        )

        # ====================================================
        # 13. SUMMARY TABLE
        # ====================================================

        print("\n")
        print("=" * 70)
        print("CANDIDATE SUMMARY")
        print("=" * 70)

        print(
            f"{'Candidate':<12}"
            f"{'Actual':<12}"
            f"{'Predicted':<12}"
            f"{'Failure Prob.':<15}"
            f"{'Correct':<10}"
        )

        print("-" * 61)

        for row in summary:

            actual_label = (
                "Failure"
                if row["actual_failure"]
                else "Success"
            )

            predicted_label = (
                "Failure"
                if row["predicted_failure"]
                else "Success"
            )

            print(
                f"{row['candidate']:<12}"
                f"{actual_label:<12}"
                f"{predicted_label:<12}"
                f"{row['failure_probability']:<15.4f}"
                f"{str(row['prediction_correct']):<10}"
            )

        # ====================================================
        # 14. COMPLETE
        # ====================================================

        print("\n")
        print("=" * 70)
        print("FAILURE PREDICTOR TEST COMPLETE")
        print("=" * 70)

    finally:

        if client_id is not None:

            try:

                p.disconnect(
                    client_id
                )

            except Exception:

                pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()