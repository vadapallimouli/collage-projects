"""
planning/path_planner.py
========================

Three-Path Motion Planning and Evaluation Orchestrator
-------------------------------------------------------

Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

Purpose
-------
This module connects the candidate trajectory generator with the
already-tested analysis, prediction, and explainability modules.

The complete workflow is:

    start position
        +
    goal position
        +
    virtual obstacles
        |
        v
    Generate exactly 3 candidate trajectories
        |
        +-------------------+
        |                   |
        v                   v
    Analyze Path 1      Analyze Path 2      Analyze Path 3
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                    Failure prediction
                            |
                            v
                    Explanation engine
                            |
                            v
                    Compare the 3 paths
                            |
                            v
                    Best candidate path

IMPORTANT ARCHITECTURAL BOUNDARY
---------------------------------
This module is an ORCHESTRATOR.

It does NOT:
    - implement inverse kinematics
    - generate Cartesian paths itself
    - calculate Jacobians itself
    - calculate torque itself
    - calculate obstacle distances itself
    - calculate singularity metrics itself
    - calculate failure-risk formulas itself
    - generate explanations itself
    - perform recovery
    - modify trajectories
    - perform the final optimization/recovery stage

Instead, it calls the project's existing modules.

RESPONSIBILITIES
----------------
    trajectory_generator.py
        -> generates exactly 3 candidate trajectories

    robot/kinematics.py
        -> computes raw kinematic measurements

    robot/dynamics.py
        -> computes raw dynamic measurements

    analysis/kinematic_analysis.py
        -> interprets kinematic measurements against configured limits

    analysis/dynamic_analysis.py
        -> interprets dynamic measurements against configured limits

    analysis/obstacle_analysis.py
        -> evaluates geometric obstacle clearance

    analysis/singularity_analysis.py
        -> evaluates singularity proximity

    prediction/failure_predictor.py
        -> computes estimated failure risk

    explainability/explanation_engine.py
        -> explains the detected problems

    this module
        -> runs that complete pipeline independently for all 3 paths
           and identifies the lowest-risk candidate

The final optimization/recovery modules can later consume the resulting
PathEvaluation objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.dynamic_analysis import DynamicSafetyAnalyzer
from analysis.kinematic_analysis import KinematicSafetyAnalyzer
from analysis.obstacle_analysis import (
    CircularObstacle,
    ObstacleSafetyAnalyzer,
)
from analysis.singularity_analysis import SingularityAnalyzer

from explainability.explanation_engine import ExplanationEngine

from models.data_models import (
    ExplanationResult,
    PathEvaluation,
    Trajectory,
    TrajectoryFeatures,
)

from prediction.failure_predictor import FailurePredictor

from robot.dynamics import DynamicAnalyzer
from robot.kinematics import KinematicAnalyzer
from robot.robot_model import Planar3RRobot

from planning.trajectory_generator import TrajectoryGenerator


class PathPlanner:
    """
    Orchestrates generation and complete evaluation of three candidate
    robot trajectories.

    The class deliberately reuses the existing modules instead of
    duplicating their calculations.

    One PathEvaluation is created for each candidate path.

    Attributes:
        robot:
            Existing Planar3RRobot instance.

        trajectory_generator:
            Existing TrajectoryGenerator used to generate the three
            candidates.

        kinematic_analyzer:
            Raw kinematic measurement module.

        dynamic_analyzer:
            Raw dynamic measurement module.

        kinematic_safety_analyzer:
            Kinematic limit interpretation module.

        dynamic_safety_analyzer:
            Dynamic limit interpretation module.

        obstacle_analyzer:
            Obstacle-clearance analysis module.

        singularity_analyzer:
            Singularity analysis module.

        failure_predictor:
            Failure-risk prediction module.

        explanation_engine:
            Explainable-AI module.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        robot: Planar3RRobot,
        n_samples: int = 100,
        duration: float = 4.0,
        goal_tolerance: float = 0.02,
    ) -> None:
        """
        Construct the complete three-path planning/evaluation pipeline.

        Args:
            robot:
                Existing Planar3RRobot instance.

            n_samples:
                Number of samples generated for each candidate trajectory.

            duration:
                Duration in seconds assigned to each candidate trajectory.

            goal_tolerance:
                Cartesian tolerance used by TrajectoryGenerator when
                checking candidate feasibility.

        Raises:
            ValueError:
                If the supplied robot is not a Planar3RRobot.
        """
        if not isinstance(robot, Planar3RRobot):
            raise ValueError(
                "robot must be an instance of Planar3RRobot."
            )

        self.robot = robot

        # --------------------------------------------------------------
        # Candidate generation
        # --------------------------------------------------------------
        #
        # This module does NOT generate paths itself. It delegates that
        # responsibility to the already-tested TrajectoryGenerator.
        #
        self.trajectory_generator = TrajectoryGenerator(
            robot=self.robot,
            n_samples=n_samples,
            duration=duration,
            goal_tolerance=goal_tolerance,
        )

        # --------------------------------------------------------------
        # Raw measurement modules
        # --------------------------------------------------------------

        self.kinematic_analyzer = KinematicAnalyzer(self.robot)
        self.dynamic_analyzer = DynamicAnalyzer(self.robot)

        # --------------------------------------------------------------
        # Safety-analysis modules
        # --------------------------------------------------------------

        self.kinematic_safety_analyzer = KinematicSafetyAnalyzer()
        self.dynamic_safety_analyzer = DynamicSafetyAnalyzer()

        # These two depend on the obstacle list / robot and are therefore
        # instantiated inside plan() when the environment is known.
        self.singularity_analyzer = SingularityAnalyzer(self.robot)

        # --------------------------------------------------------------
        # Prediction and explanation
        # --------------------------------------------------------------

        self.failure_predictor = FailurePredictor()
        self.explanation_engine = ExplanationEngine()

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start_position: Sequence[float],
        goal_position: Sequence[float],
        obstacles: Optional[Sequence[CircularObstacle]] = None,
    ) -> Tuple[List[PathEvaluation], PathEvaluation]:
        """
        Generate exactly three candidate paths, fully evaluate every
        candidate, and return the three PathEvaluation objects together
        with the lowest-risk candidate.

        Args:
            start_position:
                Cartesian [x, y] starting position in meters.

            goal_position:
                Cartesian [x, y] goal position in meters.

            obstacles:
                Sequence of virtual CircularObstacle objects.
                None means no obstacles.

        Returns:
            Tuple:
                evaluations:
                    List of exactly three PathEvaluation objects in
                    candidate order.

                best_path:
                    The candidate with the lowest estimated
                    failure-risk score.

        Important:
            "best_path" here means the candidate with the lowest
            estimated failure risk.

            This is NOT the final optimization/recovery stage. The
            project's future optimization module can later compare
            these PathEvaluation objects using additional criteria such
            as energy, smoothness, and execution time.
        """
        obstacle_list = self._validate_obstacles(obstacles)

        # ==============================================================
        # STEP 1
        # Generate exactly three candidate trajectories.
        # ==============================================================

        trajectories = self.trajectory_generator.generate_three_paths(
            start_position=start_position,
            goal_position=goal_position,
            obstacles=obstacle_list,
        )

        if len(trajectories) != 3:
            # Defensive check. TrajectoryGenerator itself guarantees
            # exactly three, but the orchestration layer should also
            # protect its own contract.
            raise RuntimeError(
                "TrajectoryGenerator returned "
                f"{len(trajectories)} candidates; exactly 3 are required."
            )

        # ==============================================================
        # STEP 2
        # Analyze every candidate independently.
        # ==============================================================

        evaluations: List[PathEvaluation] = []

        for trajectory in trajectories:
            evaluation = self._evaluate_single_path(
                trajectory=trajectory,
                obstacles=obstacle_list,
            )
            evaluations.append(evaluation)

        # ==============================================================
        # STEP 3
        # Select the lowest-risk candidate.
        # ==============================================================

        best_path = self._select_lowest_risk_path(evaluations)

        return evaluations, best_path

    # ------------------------------------------------------------------
    # Single-path pipeline
    # ------------------------------------------------------------------

    def _evaluate_single_path(
        self,
        trajectory: Trajectory,
        obstacles: Sequence[CircularObstacle],
    ) -> PathEvaluation:
        """
        Run the complete analysis -> prediction -> explanation pipeline
        for ONE trajectory.

        This method is deliberately kept separate from `plan()` so that
        the exact same pipeline is applied independently to Path 1,
        Path 2, and Path 3.

        Pipeline:

            Trajectory
                |
                +--> raw kinematics
                |
                +--> raw dynamics
                |
                +--> kinematic safety
                |
                +--> dynamic safety
                |
                +--> obstacle analysis
                |
                +--> singularity analysis
                |
                +--> feature assembly
                |
                +--> failure prediction
                |
                +--> explanation
                |
                v
            PathEvaluation
        """

        # ==============================================================
        # 1. RAW KINEMATIC ANALYSIS
        # ==============================================================

        kinematic_result = self.kinematic_analyzer.analyze(
            trajectory
        )

        # ==============================================================
        # 2. KINEMATIC SAFETY ANALYSIS
        # ==============================================================

        kinematic_analysis_result = (
            self.kinematic_safety_analyzer.analyze(
                kinematic_result
            )
        )

        # ==============================================================
        # 3. RAW DYNAMIC ANALYSIS
        # ==============================================================

        dynamic_result = self.dynamic_analyzer.analyze(
            trajectory
        )

        # ==============================================================
        # 4. DYNAMIC SAFETY ANALYSIS
        # ==============================================================

        dynamic_analysis_result = (
            self.dynamic_safety_analyzer.analyze(
                dynamic_result
            )
        )

        # ==============================================================
        # 5. OBSTACLE ANALYSIS
        # ==============================================================

        obstacle_analyzer = ObstacleSafetyAnalyzer(
            robot=self.robot,
            obstacles=obstacles,
        )

        obstacle_result = obstacle_analyzer.analyze(
            trajectory
        )

        # ==============================================================
        # 6. SINGULARITY ANALYSIS
        # ==============================================================

        singularity_result = self.singularity_analyzer.analyze(
            trajectory
        )

        # ==============================================================
        # 7. FEATURE ASSEMBLY
        # ==============================================================

        #
        # TrajectoryFeatures is a project-level data contract used by
        # downstream optimization/recovery components.
        #
        # The failure predictor currently consumes the structured
        # analysis results directly, so this object is assembled here
        # for the central PathEvaluation contract.
        #
        features = self._build_trajectory_features(
            trajectory=trajectory,
            kinematic_result=kinematic_result,
            dynamic_result=dynamic_result,
            obstacle_result=obstacle_result,
            singularity_result=singularity_result,
        )

        # ==============================================================
        # 8. FAILURE-RISK PREDICTION
        # ==============================================================

        risk_result = self.failure_predictor.predict(
            kinematic_analysis_result,
            dynamic_analysis_result,
            obstacle_result,
            singularity_result,
        )

        # ==============================================================
        # 9. EXPLAINABILITY
        # ==============================================================

        explanation = self.explanation_engine.explain(
            kinematic_analysis_result,
            dynamic_analysis_result,
            obstacle_result,
            singularity_result,
            risk_result,
        )

        # ==============================================================
        # 10. ASSEMBLE CENTRAL PathEvaluation OBJECT
        # ==============================================================

        evaluation = PathEvaluation(
            trajectory=trajectory,
            kinematic_result=kinematic_result,
            dynamic_result=dynamic_result,
            obstacle_result=obstacle_result,
            singularity_result=singularity_result,
            features=features,
            initial_risk=risk_result,
            initial_explanation=explanation,
        )

        # No recovery has happened yet.
        evaluation.recovery_evaluation = None

        # No post-recovery analysis has happened yet.
        evaluation.final_trajectory = None
        evaluation.final_kinematic_result = None
        evaluation.final_dynamic_result = None
        evaluation.final_obstacle_result = None
        evaluation.final_singularity_result = None
        evaluation.final_features = None
        evaluation.final_risk = None
        evaluation.final_explanation = None

        # At this stage the path has been evaluated but not recovered.
        #
        # If it is already safe, mark it SAFE.
        # Otherwise leave it as PENDING because the future recovery
        # module can still attempt to improve it.
        evaluation.status = (
            "SAFE"
            if risk_result.is_safe
            else "PENDING"
        )

        # The current stage uses failure risk itself as the simple
        # comparison score.
        evaluation.final_score = float(
            risk_result.overall_risk
        )

        return evaluation

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trajectory_features(
        trajectory: Trajectory,
        kinematic_result,
        dynamic_result,
        obstacle_result,
        singularity_result,
    ) -> TrajectoryFeatures:
        """
        Assemble the project's existing TrajectoryFeatures dataclass
        from already-computed analysis results.

        This method does NOT perform new physical calculations.

        It only maps existing measurements into the central feature
        contract used by PathEvaluation.

        Important:
            The actual failure-risk calculation remains inside
            FailurePredictor.
        """

        # --------------------------------------------------------------
        # Per-joint maximum torque
        # --------------------------------------------------------------

        max_torque_per_joint = np.asarray(
            dynamic_result.max_joint_torque,
            dtype=float,
        ).copy()

        # --------------------------------------------------------------
        # Per-joint maximum velocity
        # --------------------------------------------------------------

        max_velocity_per_joint = np.asarray(
            kinematic_result.max_joint_velocity,
            dtype=float,
        ).copy()

        # --------------------------------------------------------------
        # Per-joint maximum acceleration
        # --------------------------------------------------------------

        max_acceleration_per_joint = np.asarray(
            kinematic_result.max_joint_acceleration,
            dtype=float,
        ).copy()

        # --------------------------------------------------------------
        # Scalar feature values
        # --------------------------------------------------------------

        max_torque = float(
            np.max(max_torque_per_joint)
        )

        max_velocity = float(
            np.max(max_velocity_per_joint)
        )

        max_acceleration = float(
            np.max(max_acceleration_per_joint)
        )

        total_energy = float(
            dynamic_result.total_energy
        )

        execution_time = float(
            trajectory.duration
        )

        path_length = float(
            trajectory.path_length
        )

        minimum_clearance = float(
            obstacle_result.minimum_clearance
        )

        singularity_metric = float(
            singularity_result.worst_condition_metric
        )

        manipulability = float(
            singularity_result.minimum_manipulability
        )

        # --------------------------------------------------------------
        # Optional joint-space length
        # --------------------------------------------------------------

        joint_space_length = getattr(
            trajectory,
            "joint_space_length",
            None,
        )

        if joint_space_length is not None:
            joint_space_length = float(
                joint_space_length
            )

        return TrajectoryFeatures(
            max_torque=max_torque,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            total_energy=total_energy,
            execution_time=execution_time,
            path_length=path_length,
            min_obstacle_clearance=minimum_clearance,
            singularity_metric=singularity_metric,
            manipulability=manipulability,
            joint_space_length=joint_space_length,
            max_torque_per_joint=max_torque_per_joint,
            max_velocity_per_joint=max_velocity_per_joint,
            max_acceleration_per_joint=max_acceleration_per_joint,
        )

    # ------------------------------------------------------------------
    # Path selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_lowest_risk_path(
        evaluations: Sequence[PathEvaluation],
    ) -> PathEvaluation:
        """
        Select the candidate with the lowest estimated failure risk.

        This is deliberately a simple comparison at this stage.

        It does NOT perform the project's future multi-objective
        optimization. The optimization module can later use the full
        PathEvaluation objects and configured optimization weights.

        Tie-breaking:
            If two candidates have exactly the same risk, the earlier
            candidate in the generated list is selected. Python's
            min() provides this deterministic behavior.
        """

        if not evaluations:
            raise ValueError(
                "Cannot select a best path from an empty evaluation list."
            )

        if len(evaluations) != 3:
            raise ValueError(
                "PathPlanner requires exactly 3 candidate evaluations; "
                f"got {len(evaluations)}."
            )

        for evaluation in evaluations:
            if evaluation.initial_risk is None:
                raise ValueError(
                    f"Path '{evaluation.trajectory.path_id}' has no "
                    "initial risk result."
                )

            risk = float(
                evaluation.initial_risk.overall_risk
            )

            if not np.isfinite(risk):
                raise ValueError(
                    f"Path '{evaluation.trajectory.path_id}' has a "
                    f"non-finite risk value: {risk}."
                )

        return min(
            evaluations,
            key=lambda evaluation: float(
                evaluation.initial_risk.overall_risk
            ),
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_obstacles(
        obstacles: Optional[Sequence[CircularObstacle]],
    ) -> List[CircularObstacle]:
        """
        Validate and copy the obstacle sequence.

        The authoritative obstacle validation remains inside
        ObstacleSafetyAnalyzer / TrajectoryGenerator.

        This method provides an early, clear check at the planner
        boundary and prevents accidental mutation of the caller's list.
        """

        if obstacles is None:
            return []

        validated: List[CircularObstacle] = []

        for index, obstacle in enumerate(obstacles):

            if not isinstance(
                obstacle,
                CircularObstacle,
            ):
                raise ValueError(
                    "All obstacles must be CircularObstacle instances. "
                    f"Item {index} has type {type(obstacle).__name__}."
                )

            validated.append(
                CircularObstacle(
                    obstacle_id=str(
                        obstacle.obstacle_id
                    ),
                    center=np.array(
                        obstacle.center,
                        dtype=float,
                        copy=True,
                    ),
                    radius=float(
                        obstacle.radius
                    ),
                )
            )

        return validated


# =============================================================================
# DEMO / THREE-PATH END-TO-END SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    """Interactive three-path planning demo with user-friendly validation."""

    def read_float(prompt, minimum=None, strict=False):
        while True:
            raw = input(prompt).strip()
            try:
                value = float(raw)
            except ValueError:
                print("\nInvalid input. Please enter a numerical value.\n")
                continue
            if not np.isfinite(value):
                print("\nPlease enter a finite number.\n")
                continue
            if minimum is not None:
                if strict and value <= minimum:
                    print(f"\nValue must be greater than {minimum}.\n")
                    continue
                if not strict and value < minimum:
                    print(f"\nValue must be at least {minimum}.\n")
                    continue
            return value

    def read_int(prompt):
        while True:
            raw = input(prompt).strip()
            try:
                value = int(raw)
            except ValueError:
                print("\nPlease enter a whole number.\n")
                continue
            if value < 0:
                print("\nThe number cannot be negative.\n")
                continue
            return value

    def read_position(label):
        print(f"\n{label} position (metres)")
        x = read_float("X: ")
        y = read_float("Y: ")
        return np.array([x, y], dtype=float)

    def show_problem(title, lines):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for line in lines:
            print(line)
        print("=" * 72)

    def validate_environment(robot, start, goal, obstacles):
        """Reject impossible user environments before IK/path generation."""
        max_reach = float(robot.get_max_reach())
        start_distance = float(np.linalg.norm(start))
        goal_distance = float(np.linalg.norm(goal))
        tolerance = 1e-9

        if start_distance > max_reach + tolerance:
            show_problem(
                "START POSITION IS UNREACHABLE",
                [
                    f"Requested start: {start}",
                    f"Distance from robot base: {start_distance:.3f} m",
                    f"Robot maximum reach: {max_reach:.3f} m",
                    "",
                    "Why:",
                    "The requested start point lies outside the robot's "
                    "reachable workspace.",
                    "",
                    "What to do:",
                    f"Choose a start point whose distance from the base "
                    f"is at most {max_reach:.2f} m.",
                    "No trajectory was generated and no failure-risk "
                    "analysis was performed.",
                ],
            )
            return False

        if goal_distance > max_reach + tolerance:
            show_problem(
                "GOAL POSITION IS UNREACHABLE",
                [
                    f"Requested goal: {goal}",
                    f"Distance from robot base: {goal_distance:.3f} m",
                    f"Robot maximum reach: {max_reach:.3f} m",
                    "",
                    "Why:",
                    "The requested goal point lies outside the robot's "
                    "reachable workspace.",
                    "",
                    "What to do:",
                    f"Choose a goal point whose distance from the base "
                    f"is at most {max_reach:.2f} m.",
                    "No trajectory was generated and no failure-risk "
                    "analysis was performed.",
                ],
            )
            return False

        if np.linalg.norm(goal - start) <= tolerance:
            show_problem(
                "INVALID START AND GOAL",
                [
                    "The start and goal positions are identical.",
                    "",
                    "A motion trajectory requires a non-zero movement.",
                    "Please enter different start and goal positions.",
                ],
            )
            return False

        for obstacle in obstacles:
            start_distance_to_obstacle = float(
                np.linalg.norm(start - obstacle.center)
            )
            goal_distance_to_obstacle = float(
                np.linalg.norm(goal - obstacle.center)
            )

            if start_distance_to_obstacle <= obstacle.radius + tolerance:
                show_problem(
                    "START POSITION IS INSIDE AN OBSTACLE",
                    [
                        f"Start position: {start}",
                        f"Obstacle: {obstacle.obstacle_id}",
                        f"Obstacle center: {obstacle.center}",
                        f"Obstacle radius: {obstacle.radius:.3f} m",
                        f"Distance from start to obstacle center: "
                        f"{start_distance_to_obstacle:.3f} m",
                        "",
                        "Why:",
                        "The requested motion starts inside or exactly "
                        "on the obstacle boundary.",
                        "",
                        "A collision-free trajectory cannot begin from "
                        "inside an obstacle.",
                        "",
                        "What to do:",
                        "Move the start position outside the obstacle "
                        "or move/reduce the obstacle.",
                        "No trajectory was generated and no failure-risk "
                        "analysis was performed.",
                    ],
                )
                return False

            if goal_distance_to_obstacle <= obstacle.radius + tolerance:
                show_problem(
                    "GOAL POSITION IS INSIDE AN OBSTACLE",
                    [
                        f"Goal position: {goal}",
                        f"Obstacle: {obstacle.obstacle_id}",
                        f"Obstacle center: {obstacle.center}",
                        f"Obstacle radius: {obstacle.radius:.3f} m",
                        f"Distance from goal to obstacle center: "
                        f"{goal_distance_to_obstacle:.3f} m",
                        "",
                        "Why:",
                        "The requested motion ends inside or exactly "
                        "on the obstacle boundary.",
                        "",
                        "A collision-free trajectory cannot finish "
                        "inside an obstacle.",
                        "",
                        "What to do:",
                        "Move the goal position outside the obstacle "
                        "or move/reduce the obstacle.",
                        "No trajectory was generated and no failure-risk "
                        "analysis was performed.",
                    ],
                )
                return False

        return True

    robot = Planar3RRobot()

    print("\n" + "=" * 72)
    print("THREE-PATH ROBOT MOTION PLANNING AND FAILURE ANALYSIS")
    print("=" * 72)
    print("\nRobot: Planar3R")
    print(f"Maximum reach: {robot.get_max_reach():.2f} m")

    start_position = read_position("START")
    goal_position = read_position("GOAL")

    number_of_obstacles = read_int(
        "\nEnter number of circular obstacles: "
    )

    obstacles = []

    for i in range(number_of_obstacles):
        print(f"\n--- Obstacle {i + 1} ---")
        x = read_float("X: ")
        y = read_float("Y: ")
        radius = read_float(
            "Radius (metres): ",
            minimum=0.0,
            strict=True,
        )

        obstacles.append(
            CircularObstacle(
                obstacle_id=f"O{i + 1}",
                center=np.array([x, y], dtype=float),
                radius=radius,
            )
        )

    print("\n" + "=" * 72)
    print("INPUT SUMMARY")
    print("=" * 72)
    print(f"Start position: {start_position}")
    print(f"Goal position:  {goal_position}")
    print(f"Number of obstacles: {len(obstacles)}")

    for obstacle in obstacles:
        print(
            f"  {obstacle.obstacle_id}: "
            f"center={obstacle.center}, "
            f"radius={obstacle.radius:.3f} m"
        )

    if not validate_environment(
        robot,
        start_position,
        goal_position,
        obstacles,
    ):
        print("\nPlease run the program again with a valid environment.")
    else:
        planner = PathPlanner(
            robot=robot,
            n_samples=100,
            duration=4.0,
        )

        print("\nGenerating and evaluating three candidate paths...")

        try:
            evaluations, best_path = planner.plan(
                start_position=start_position,
                goal_position=goal_position,
                obstacles=obstacles,
            )

        except Exception as exc:
            message = str(exc)

            if (
                "failed to generate three feasible candidate trajectories"
                in message.lower()
            ):
                show_problem(
                    "UNABLE TO GENERATE THREE FEASIBLE PATHS",
                    [
                        "The start and goal positions are reachable, but "
                        "the current environment did not allow the planner "
                        "to produce three feasible candidate trajectories.",
                        "",
                        "This can happen when:",
                        "• obstacles block the available routes,",
                        "• an obstacle is too large or too close to the "
                        "requested motion,",
                        "• a candidate trajectory causes robot-link "
                        "collision, or",
                        "• inverse kinematics cannot find a valid "
                        "configuration for a candidate.",
                        "",
                        "What to do:",
                        "• Move the obstacles farther away.",
                        "• Reduce an obstacle radius.",
                        "• Try a different start/goal pair.",
                        "• Try a less constrained obstacle arrangement.",
                        "",
                        "The planner does not create duplicate or fake "
                        "paths just to reach a count of three.",
                    ],
                )
            else:
                show_problem(
                    "PLANNING COULD NOT BE COMPLETED",
                    [
                        "The requested environment could not be processed "
                        "successfully.",
                        "",
                        f"Reason: {message}",
                        "",
                        "Try a different start/goal position or obstacle "
                        "configuration.",
                        "No final path was selected.",
                    ],
                )

        else:
            print("\n" + "=" * 72)
            print("THREE-PATH ROBOT MOTION PLANNING AND FAILURE ANALYSIS")
            print("=" * 72)
            print(f"\nStart position: {start_position}")
            print(f"Goal position:  {goal_position}")
            print(f"Number of obstacles: {len(obstacles)}")
            print(f"Number of candidate paths evaluated: {len(evaluations)}")

            for index, evaluation in enumerate(evaluations, start=1):
                trajectory = evaluation.trajectory
                risk = evaluation.initial_risk
                explanation = evaluation.initial_explanation
                obstacle_result = evaluation.obstacle_result
                singularity_result = evaluation.singularity_result
                dynamic_result = evaluation.dynamic_result

                print("\n" + "-" * 72)
                print(f"PATH {index}: {trajectory.name}")
                print("-" * 72)
                print(f"ID: {trajectory.path_id}")
                print(f"Samples: {len(trajectory.time)}")
                print(f"Duration: {trajectory.duration:.4f} s")
                print(
                    f"Cartesian path length: "
                    f"{trajectory.path_length:.4f} m"
                )

                joint_space_length = getattr(
                    trajectory,
                    "joint_space_length",
                    None,
                )
                if joint_space_length is not None:
                    print(
                        f"Joint-space length: "
                        f"{float(joint_space_length):.4f} rad"
                    )

                print(f"Failure risk: {risk.risk_percentage:.1f}%")
                print(f"Risk level: {risk.risk_level}")
                print(f"Safe: {risk.is_safe}")
                print(
                    f"Obstacle collision: "
                    f"{obstacle_result.collision_detected}"
                )
                print(
                    f"Minimum obstacle clearance: "
                    f"{obstacle_result.minimum_clearance:.4f} m"
                )
                print(
                    f"Maximum |torque| per joint: "
                    f"{dynamic_result.max_joint_torque}"
                )
                print(
                    f"Singularity interpretation: "
                    f"{singularity_result.risk_level}"
                )
                print(
                    f"Minimum singularity metric: "
                    f"{singularity_result.worst_condition_metric:.6e}"
                )

                print("\nMain contributors:")
                if risk.main_contributors:
                    for rank, contributor in enumerate(
                        risk.main_contributors,
                        start=1,
                    ):
                        print(f"  {rank}. {contributor}")
                else:
                    print("  None")

                print("\nDetected problems:")
                if risk.detected_problems:
                    for problem in risk.detected_problems:
                        print(f"  - {problem}")
                else:
                    print("  None")

                print("\nExplanation summary:")
                print(f"  {explanation.summary}")

                print("\nMain reason:")
                print(f"  {explanation.main_reason}")

            print("\n" + "=" * 72)
            print("FINAL THREE-PATH COMPARISON")
            print("=" * 72)

            for evaluation in evaluations:
                trajectory = evaluation.trajectory
                risk = evaluation.initial_risk
                print(
                    f"{trajectory.path_id:<10} "
                    f"{trajectory.name:<30} "
                    f"{risk.risk_percentage:6.1f}% "
                    f"{risk.risk_level}"
                )

            print("\n" + "-" * 72)
            print("LOWEST-RISK CANDIDATE")
            print("-" * 72)
            print(f"Best path ID: {best_path.trajectory.path_id}")
            print(f"Best path name: {best_path.trajectory.name}")
            print(
                f"Estimated failure risk: "
                f"{best_path.initial_risk.risk_percentage:.1f}%"
            )
            print(f"Risk level: {best_path.initial_risk.risk_level}")
            print(
                "Selection reason: lowest estimated failure risk "
                "among the three generated candidates."
            )

            assert len(evaluations) == 3
            assert [
                e.trajectory.path_id for e in evaluations
            ] == ["path-1", "path-2", "path-3"]

            for evaluation in evaluations:
                assert isinstance(evaluation, PathEvaluation)
                assert isinstance(evaluation.trajectory, Trajectory)
                assert evaluation.kinematic_result is not None
                assert evaluation.dynamic_result is not None
                assert evaluation.obstacle_result is not None
                assert evaluation.singularity_result is not None
                assert np.isfinite(
                    evaluation.initial_risk.overall_risk
                )
                assert 0.0 <= evaluation.initial_risk.overall_risk <= 1.0
                assert isinstance(
                    evaluation.initial_explanation,
                    ExplanationResult,
                )

            assert best_path in evaluations

            minimum_risk = min(
                e.initial_risk.overall_risk
                for e in evaluations
            )

            assert np.isclose(
                best_path.initial_risk.overall_risk,
                minimum_risk,
            )

            print(
                "\nAll three paths were generated and fully evaluated."
            )
            print(
                "Failure prediction and explanation completed for all 3 paths."
            )
            print(
                "Best candidate selected using lowest estimated failure risk."
            )