"""
analysis/obstacle_analysis.py
================================

Obstacle Safety Analysis Module
--------------------------------
Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "Is this trajectory geometrically safe with respect to the
     defined virtual obstacles?"

It takes:
    - an already-generated Trajectory
    - the existing Planar3RRobot model
    - a list of circular virtual obstacles

and computes structured geometric evidence:

    - minimum obstacle clearance
    - whether a collision occurred
    - closest obstacle
    - closest point on the robot
    - minimum clearance for each obstacle
    - geometric obstacle proximity indicator

IMPORTANT:
    The obstacle proximity indicator is NOT a failure probability.
    Failure prediction belongs to the prediction module.

This module intentionally does NOT:
    - calculate failure probability
    - calculate an overall risk score
    - classify SAFE / UNSAFE
    - calculate torque, power, or energy
    - analyze joint limits
    - classify singularity severity
    - generate explanations
    - perform recovery
    - modify trajectories
    - generate new trajectories
    - compare multiple trajectories
    - optimize path selection


ARCHITECTURAL BOUNDARY
----------------------

    robot_model.py
        -> "What robot do we have?"

    kinematics.py
        -> "What is the robot doing kinematically?"

    dynamics.py
        -> "What forces/torques/energy are involved?"

    analysis/kinematic_analysis.py
        -> "Did kinematics violate configured limits?"

    analysis/dynamic_analysis.py
        -> "Did dynamics violate configured limits?"

    analysis/obstacle_analysis.py
        -> "Is the path too close to obstacles?"       <-- THIS FILE

    analysis/singularity_analysis.py
        -> "How severe is the singularity risk?"

    prediction/
        -> "What is the estimated failure risk?"

    explainability/
        -> "Why is it risky?"

    recovery/
        -> "Can we fix it?"

    optimization/
        -> "Which final path is best?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    COLLISION_DISTANCE,
    MIN_SAFE_CLEARANCE,
    OBSTACLE_WARNING_DISTANCE,
)
from models.data_models import ObstacleResult, Trajectory
from robot.robot_model import Planar3RRobot


# =============================================================================
# Numerical constants
# =============================================================================

# Small numerical tolerance used only to protect calculations against
# floating-point errors around exact tangency and degenerate segments.
# This is NOT a physical safety threshold.
_EPSILON: float = 1e-9


# Human-readable names for the three robot links.
_LINK_NAMES: Tuple[str, str, str] = (
    "Link 1",
    "Link 2",
    "Link 3",
)


# =============================================================================
# Circular obstacle representation
# =============================================================================


@dataclass(frozen=True)
class CircularObstacle:
    """
    Represents one circular virtual obstacle in the robot's XY workspace.

    Attributes:
        obstacle_id:
            Unique identifier such as "O1".

        center:
            2D obstacle-center position [x, y] in meters.

        radius:
            Radius of the obstacle in meters.
    """

    obstacle_id: str
    center: np.ndarray
    radius: float


# =============================================================================
# Obstacle Safety Analyzer
# =============================================================================


class ObstacleSafetyAnalyzer:
    """
    Performs geometric obstacle-clearance analysis for one trajectory.

    The analyzer checks every robot link against every circular obstacle
    at every trajectory sample.

    It does NOT decide whether the trajectory ultimately fails.

    Attributes:
        robot:
            Existing Planar3RRobot instance.

        obstacles:
            Validated immutable tuple of CircularObstacle objects.

        safety_margin:
            Additional distance used for safety-margin detection.
            Defaults to MIN_SAFE_CLEARANCE from config.py.
    """

    def __init__(
        self,
        robot: Planar3RRobot,
        obstacles: Sequence[CircularObstacle],
        safety_margin: Optional[float] = None,
    ) -> None:

        self.robot = robot

        # Obtain link lengths from the already-tested robot model.
        # No second robot model is created here.
        self._link_lengths = self._validate_link_lengths(
            robot.get_link_lengths()
        )

        self.obstacles: Tuple[CircularObstacle, ...] = (
            self._validate_obstacles(obstacles)
        )

        self.safety_margin: float = self._validate_safety_margin(
            MIN_SAFE_CLEARANCE
            if safety_margin is None
            else safety_margin
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def analyze(self, trajectory: Trajectory) -> ObstacleResult:
        """
        Analyze one trajectory against all configured obstacles.

        Args:
            trajectory:
                Trajectory containing q with shape (N, 3).

        Returns:
            ObstacleResult containing the project's existing obstacle
            analysis fields.

        Raises:
            ValueError:
                If the trajectory or configuration is invalid.
        """

        result, _, _ = self._analyze_internal(trajectory)

        return result

    # =========================================================================
    # Internal analysis API
    # =========================================================================

    def _analyze_internal(
        self,
        trajectory: Trajectory,
    ) -> Tuple[ObstacleResult, int, int]:
        """
        Perform the actual obstacle analysis.

        This private method returns:

            1. Official ObstacleResult
            2. Number of trajectory samples containing a collision
            3. Number of trajectory samples violating the safety margin

        The additional values are intentionally NOT added to
        ObstacleResult because the existing data contract does not
        contain those fields.

        Returns:
            Tuple[
                ObstacleResult,
                collision_sample_count,
                margin_violation_sample_count
            ]
        """

        q = self._validate_trajectory(trajectory)

        # ---------------------------------------------------------------------
        # Edge case: no obstacles
        # ---------------------------------------------------------------------

        if not self.obstacles:
            result = ObstacleResult(
                minimum_clearance=float("inf"),
                collision_detected=False,
                obstacle_risk_indicator=0.0,
                closest_point=None,
                closest_obstacle_id=None,
                obstacle_clearances={},
            )

            return result, 0, 0

        # ---------------------------------------------------------------------
        # Running global values
        # ---------------------------------------------------------------------

        global_min_clearance = float("inf")

        global_closest_point: Optional[np.ndarray] = None
        global_closest_obstacle_id: Optional[str] = None

        global_closest_sample: Optional[int] = None
        global_closest_link: Optional[str] = None

        # ---------------------------------------------------------------------
        # Per-obstacle minimum clearance
        # ---------------------------------------------------------------------

        per_obstacle_min_clearance: Dict[str, float] = {
            obstacle.obstacle_id: float("inf")
            for obstacle in self.obstacles
        }

        # ---------------------------------------------------------------------
        # Internal evidence
        # ---------------------------------------------------------------------

        n_samples = q.shape[0]

        collision_mask = np.zeros(
            n_samples,
            dtype=bool,
        )

        margin_violation_mask = np.zeros(
            n_samples,
            dtype=bool,
        )

        # =========================================================================
        # Main trajectory loop
        # =========================================================================

        for sample_index in range(n_samples):

            # -------------------------------------------------------------
            # Calculate robot link geometry
            # -------------------------------------------------------------

            link_points = self._link_points_from_q(
                q[sample_index]
            )

            # Four points:
            #
            # p0 = robot base
            # p1 = joint 1
            # p2 = joint 2
            # p3 = end effector
            #
            # Therefore:
            #
            # Link 1 = p0 -> p1
            # Link 2 = p1 -> p2
            # Link 3 = p2 -> p3

            segments = (
                (link_points[0], link_points[1]),
                (link_points[1], link_points[2]),
                (link_points[2], link_points[3]),
            )

            sample_collision = False
            sample_margin_violation = False

            # -------------------------------------------------------------
            # Check every obstacle against every robot link
            # -------------------------------------------------------------

            for obstacle in self.obstacles:

                for link_index, (seg_start, seg_end) in enumerate(
                    segments
                ):

                    distance, nearest_point = (
                        self._point_to_segment_distance(
                            obstacle.center,
                            seg_start,
                            seg_end,
                        )
                    )

                    # -----------------------------------------------------
                    # Clearance from obstacle boundary
                    # -----------------------------------------------------

                    clearance = (
                        distance - obstacle.radius
                    )

                    # -----------------------------------------------------
                    # Update per-obstacle minimum
                    # -----------------------------------------------------

                    if (
                        clearance
                        < per_obstacle_min_clearance[
                            obstacle.obstacle_id
                        ]
                    ):
                        per_obstacle_min_clearance[
                            obstacle.obstacle_id
                        ] = clearance

                    # -----------------------------------------------------
                    # Update global minimum
                    # -----------------------------------------------------

                    if clearance < global_min_clearance:

                        global_min_clearance = clearance

                        global_closest_point = np.array(
                            nearest_point,
                            copy=True,
                        )

                        global_closest_obstacle_id = (
                            obstacle.obstacle_id
                        )

                        global_closest_sample = sample_index

                        global_closest_link = (
                            _LINK_NAMES[link_index]
                        )

                    # -----------------------------------------------------
                    # Actual geometric collision
                    # -----------------------------------------------------
                    #
                    # Collision occurs when the robot link reaches or
                    # penetrates the obstacle boundary.
                    #
                    # distance <= radius
                    #
                    # Equivalently:
                    #
                    # clearance <= 0
                    #
                    # Tangency counts as collision.
                    # -----------------------------------------------------

                    if (
                        distance
                        <= obstacle.radius + _EPSILON
                    ):
                        sample_collision = True

                    # -----------------------------------------------------
                    # Safety-margin violation
                    # -----------------------------------------------------
                    #
                    # This is NOT collision.
                    #
                    # The robot is outside the obstacle but is closer
                    # than the configured safety margin.
                    # -----------------------------------------------------

                    elif (
                        distance
                        <= (
                            obstacle.radius
                            + self.safety_margin
                            + _EPSILON
                        )
                    ):
                        sample_margin_violation = True

            collision_mask[sample_index] = sample_collision
            margin_violation_mask[sample_index] = (
                sample_margin_violation
            )

        # =========================================================================
        # Final structured measurements
        # =========================================================================

        collision_detected = bool(
            np.any(collision_mask)
        )

        collision_sample_count = int(
            np.sum(collision_mask)
        )

        margin_violation_sample_count = int(
            np.sum(margin_violation_mask)
        )

        # Geometric proximity indicator.
        #
        # IMPORTANT:
        # This is NOT failure probability.
        # It is only a normalized measure of obstacle proximity.
        obstacle_risk_indicator = (
            self._compute_risk_indicator(
                global_min_clearance
            )
        )

        # =========================================================================
        # Build official project result
        # =========================================================================

        result = ObstacleResult(
            minimum_clearance=float(
                global_min_clearance
            ),
            collision_detected=collision_detected,
            obstacle_risk_indicator=(
                obstacle_risk_indicator
            ),
            closest_point=global_closest_point,
            closest_obstacle_id=(
                global_closest_obstacle_id
            ),
            obstacle_clearances={
                obstacle_id: float(value)
                for obstacle_id, value
                in per_obstacle_min_clearance.items()
            },
        )

        return (
            result,
            collision_sample_count,
            margin_violation_sample_count,
        )

    # =========================================================================
    # Configuration validation
    # =========================================================================

    @staticmethod
    def _validate_link_lengths(
        link_lengths,
    ) -> np.ndarray:
        """
        Validate the robot's configured link lengths.

        Exactly three finite positive values are required.
        """

        lengths = np.asarray(
            link_lengths,
            dtype=float,
        )

        if lengths.shape != (3,):
            raise ValueError(
                "robot.get_link_lengths() must have "
                f"shape (3,), got {lengths.shape}."
            )

        if not np.all(
            np.isfinite(lengths)
        ):
            raise ValueError(
                "robot.get_link_lengths() contains NaN "
                "or infinite values."
            )

        if np.any(lengths <= 0.0):
            raise ValueError(
                "All robot link lengths must be greater "
                f"than zero. Got {lengths}."
            )

        return np.array(
            lengths,
            copy=True,
        )

    @staticmethod
    def _validate_obstacles(
        obstacles: Sequence[CircularObstacle],
    ) -> Tuple[CircularObstacle, ...]:
        """
        Validate all supplied circular obstacles.

        Requirements:

            - obstacle ID must be non-empty
            - IDs must be unique
            - center must have shape (2,)
            - center must contain finite values
            - radius must be finite
            - radius must be > 0
        """

        validated: List[
            CircularObstacle
        ] = []

        seen_ids = set()

        for obstacle in obstacles:

            # -------------------------------------------------------------
            # ID
            # -------------------------------------------------------------

            if not isinstance(
                obstacle.obstacle_id,
                str,
            ):
                raise ValueError(
                    "obstacle_id must be a string."
                )

            if not obstacle.obstacle_id.strip():
                raise ValueError(
                    "Every obstacle must have a "
                    "non-empty obstacle_id."
                )

            if obstacle.obstacle_id in seen_ids:
                raise ValueError(
                    f"Duplicate obstacle_id "
                    f"'{obstacle.obstacle_id}'. "
                    "Obstacle IDs must be unique."
                )

            seen_ids.add(
                obstacle.obstacle_id
            )

            # -------------------------------------------------------------
            # Center
            # -------------------------------------------------------------

            center = np.asarray(
                obstacle.center,
                dtype=float,
            )

            if center.shape != (2,):
                raise ValueError(
                    f"Obstacle "
                    f"'{obstacle.obstacle_id}'.center "
                    f"must have shape (2,), "
                    f"got {center.shape}."
                )

            if not np.all(
                np.isfinite(center)
            ):
                raise ValueError(
                    f"Obstacle "
                    f"'{obstacle.obstacle_id}'.center "
                    "contains NaN or infinite values."
                )

            # -------------------------------------------------------------
            # Radius
            # -------------------------------------------------------------

            radius = float(
                obstacle.radius
            )

            if not np.isfinite(radius):
                raise ValueError(
                    f"Obstacle "
                    f"'{obstacle.obstacle_id}'.radius "
                    f"must be finite. Got {radius}."
                )

            if radius <= 0.0:
                raise ValueError(
                    f"Obstacle "
                    f"'{obstacle.obstacle_id}'.radius "
                    f"must be > 0. Got {radius}."
                )

            # -------------------------------------------------------------
            # Store defensive copies
            # -------------------------------------------------------------

            validated.append(
                CircularObstacle(
                    obstacle_id=obstacle.obstacle_id,
                    center=np.array(
                        center,
                        copy=True,
                    ),
                    radius=radius,
                )
            )

        return tuple(validated)

    @staticmethod
    def _validate_safety_margin(
        safety_margin: float,
    ) -> float:
        """
        Validate the obstacle safety margin.

        The margin must be finite and non-negative.
        """

        margin = float(
            safety_margin
        )

        if not np.isfinite(margin):
            raise ValueError(
                f"safety_margin must be finite. "
                f"Got {margin}."
            )

        if margin < 0.0:
            raise ValueError(
                f"safety_margin must be >= 0. "
                f"Got {margin}."
            )

        return margin

    # =========================================================================
    # Trajectory validation
    # =========================================================================

    @staticmethod
    def _validate_trajectory(
        trajectory: Trajectory,
    ) -> np.ndarray:
        """
        Validate trajectory.q.

        Requirements:

            shape = (N, 3)
            N > 0
            all values finite

        The original trajectory is never modified.
        """

        q = np.asarray(
            trajectory.q,
            dtype=float,
        )

        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(
                "trajectory.q must have "
                f"shape (N, 3), got {q.shape}."
            )

        if q.shape[0] == 0:
            raise ValueError(
                "trajectory.q contains zero samples. "
                "An empty trajectory cannot be analyzed."
            )

        if not np.all(
            np.isfinite(q)
        ):
            raise ValueError(
                "trajectory.q contains NaN or "
                "infinite values."
            )

        return np.array(
            q,
            copy=True,
        )

    # =========================================================================
    # Robot planar geometry
    # =========================================================================

    def _link_points_from_q(
        self,
        q_sample: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        Calculate the planar positions of:

            p0 = base
            p1 = joint 1
            p2 = joint 2
            p3 = end effector

        for one 3R joint configuration.

        Uses the configured robot link lengths.

        No second Robotics Toolbox robot is constructed.
        """

        q_sample = np.asarray(
            q_sample,
            dtype=float,
        )

        if q_sample.shape != (3,):
            raise ValueError(
                "q_sample must have shape (3,), "
                f"got {q_sample.shape}."
            )

        if not np.all(
            np.isfinite(q_sample)
        ):
            raise ValueError(
                "q_sample contains NaN or "
                "infinite values."
            )

        l1, l2, l3 = (
            self._link_lengths
        )

        q1, q2, q3 = q_sample

        # -----------------------------------------------------------------
        # Base
        # -----------------------------------------------------------------

        p0 = np.array(
            [0.0, 0.0],
            dtype=float,
        )

        # -----------------------------------------------------------------
        # Joint 1
        # -----------------------------------------------------------------

        x1 = (
            l1 * np.cos(q1)
        )

        y1 = (
            l1 * np.sin(q1)
        )

        p1 = np.array(
            [x1, y1],
            dtype=float,
        )

        # -----------------------------------------------------------------
        # Joint 2
        # -----------------------------------------------------------------

        q12 = q1 + q2

        x2 = (
            x1
            + l2 * np.cos(q12)
        )

        y2 = (
            y1
            + l2 * np.sin(q12)
        )

        p2 = np.array(
            [x2, y2],
            dtype=float,
        )

        # -----------------------------------------------------------------
        # End effector
        # -----------------------------------------------------------------

        q123 = q1 + q2 + q3

        x3 = (
            x2
            + l3 * np.cos(q123)
        )

        y3 = (
            y2
            + l3 * np.sin(q123)
        )

        p3 = np.array(
            [x3, y3],
            dtype=float,
        )

        return (
            p0,
            p1,
            p2,
            p3,
        )

    # =========================================================================
    # Point-to-segment distance
    # =========================================================================

    @staticmethod
    def _point_to_segment_distance(
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate the shortest Euclidean distance between:

            point

        and the finite line segment:

            seg_start -> seg_end

        Returns:

            distance:
                shortest distance in meters

            nearest_point:
                point on the robot link where the shortest
                distance occurs
        """

        point = np.asarray(
            point,
            dtype=float,
        )

        seg_start = np.asarray(
            seg_start,
            dtype=float,
        )

        seg_end = np.asarray(
            seg_end,
            dtype=float,
        )

        segment_vector = (
            seg_end - seg_start
        )

        segment_length_squared = float(
            np.dot(
                segment_vector,
                segment_vector,
            )
        )

        # -----------------------------------------------------------------
        # Degenerate segment
        # -----------------------------------------------------------------
        #
        # If both endpoints are effectively identical, treat the segment
        # as a single point.
        # -----------------------------------------------------------------

        if (
            segment_length_squared
            < _EPSILON
        ):

            nearest_point = np.array(
                seg_start,
                copy=True,
            )

            distance = float(
                np.linalg.norm(
                    point - seg_start
                )
            )

            return (
                distance,
                nearest_point,
            )

        # -----------------------------------------------------------------
        # Projection of point onto the infinite line
        # -----------------------------------------------------------------

        t = float(
            np.dot(
                point - seg_start,
                segment_vector,
            )
            / segment_length_squared
        )

        # -----------------------------------------------------------------
        # Clamp projection to the actual finite segment
        # -----------------------------------------------------------------

        t = min(
            1.0,
            max(0.0, t),
        )

        nearest_point = (
            seg_start
            + t * segment_vector
        )

        distance = float(
            np.linalg.norm(
                point - nearest_point
            )
        )

        return (
            distance,
            nearest_point,
        )

    # =========================================================================
    # Geometric proximity indicator
    # =========================================================================

    @staticmethod
    def _compute_risk_indicator(
        minimum_clearance: float,
    ) -> float:
        """
        Compute the normalized geometric obstacle-proximity indicator.

        IMPORTANT:
            This is NOT the project's failure probability.

        It only describes geometric proximity to the obstacle.

        Definition:

            clearance <= 0
                -> 1.0

            clearance >= OBSTACLE_WARNING_DISTANCE
                -> 0.0

            0 < clearance < OBSTACLE_WARNING_DISTANCE
                -> linear interpolation

        Therefore:

            collision / penetration
                -> maximum indicator

            obstacle boundary
                -> maximum indicator

            0.25 m or more from boundary
                -> zero indicator

        MIN_SAFE_CLEARANCE is intentionally NOT used here because it
        represents the configured safety-margin concept.

        COLLISION_DISTANCE is also NOT used here because actual
        collision is already determined geometrically using the
        obstacle radius.
        """

        # -------------------------------------------------------------
        # No finite obstacle clearance
        # -------------------------------------------------------------

        if not np.isfinite(
            minimum_clearance
        ):
            return 0.0

        # -------------------------------------------------------------
        # Collision / penetration
        # -------------------------------------------------------------

        if minimum_clearance <= 0.0:
            return 1.0

        # -------------------------------------------------------------
        # Comfortably clear
        # -------------------------------------------------------------

        if (
            minimum_clearance
            >= OBSTACLE_WARNING_DISTANCE
        ):
            return 0.0

        # -------------------------------------------------------------
        # Intermediate region
        # -------------------------------------------------------------
        #
        # Example:
        #
        # clearance = 0
        #     indicator = 1
        #
        # clearance = warning distance
        #     indicator = 0
        #
        # Linear interpolation.
        # -------------------------------------------------------------

        indicator = (
            OBSTACLE_WARNING_DISTANCE
            - minimum_clearance
        ) / OBSTACLE_WARNING_DISTANCE

        return float(
            np.clip(
                indicator,
                0.0,
                1.0,
            )
        )


# =============================================================================
# Demo / sanity check
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Create a synthetic trajectory
    # -------------------------------------------------------------------------

    n_samples = 100
    duration = 4.0

    time = np.linspace(
        0.0,
        duration,
        n_samples,
    )

    # Smooth joint motion.
    q1_target = 0.9
    q2_target = -0.6
    q3_target = 0.3

    fraction = (
        time / duration
    )

    q = np.stack(
        [
            q1_target * fraction,
            q2_target * fraction,
            q3_target * fraction,
        ],
        axis=1,
    )

    # Derivatives are included because Trajectory requires them.
    dt = time[1] - time[0]

    qd = np.gradient(
        q,
        dt,
        axis=0,
    )

    qdd = np.gradient(
        qd,
        dt,
        axis=0,
    )

    # -------------------------------------------------------------------------
    # Existing robot model
    # -------------------------------------------------------------------------

    robot = Planar3RRobot()

    # -------------------------------------------------------------------------
    # Create Trajectory
    # -------------------------------------------------------------------------

    demo_trajectory = Trajectory(
        path_id="demo-001",
        name="Obstacle safety analysis demo",
        description=(
            "Synthetic trajectory used only to test "
            "obstacle_analysis.py"
        ),
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=robot.end_effector_position(
            q[0]
        ),
        goal_position=robot.end_effector_position(
            q[-1]
        ),
        duration=duration,
        path_length=0.0,
    )

    # -------------------------------------------------------------------------
    # Define virtual circular obstacles
    # -------------------------------------------------------------------------

    demo_obstacles = [
        CircularObstacle(
            obstacle_id="O1",
            center=np.array(
                [1.3, 0.6],
                dtype=float,
            ),
            radius=0.20,
        ),
        CircularObstacle(
            obstacle_id="O2",
            center=np.array(
                [-1.0, 1.0],
                dtype=float,
            ),
            radius=0.15,
        ),
    ]

    # -------------------------------------------------------------------------
    # Create analyzer
    # -------------------------------------------------------------------------

    analyzer = ObstacleSafetyAnalyzer(
        robot=robot,
        obstacles=demo_obstacles,
    )

    # -------------------------------------------------------------------------
    # Run analysis
    # -------------------------------------------------------------------------

    (
        result,
        collision_sample_count,
        margin_violation_sample_count,
    ) = analyzer._analyze_internal(
        demo_trajectory
    )

    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------

    print(
        f"Number of samples: {n_samples}"
    )

    print(
        f"Number of obstacles: "
        f"{len(demo_obstacles)}"
    )

    print(
        f"\nCollision detected: "
        f"{result.collision_detected}"
    )

    print(
        f"Collision sample count: "
        f"{collision_sample_count}"
    )

    print(
        f"Safety-margin violation samples: "
        f"{margin_violation_sample_count}"
    )

    print(
        f"\nMinimum clearance (m): "
        f"{result.minimum_clearance:.6f}"
    )

    print(
        f"Closest obstacle: "
        f"{result.closest_obstacle_id}"
    )

    print(
        f"Closest point on robot (m): "
        f"{result.closest_point}"
    )

    print(
        "Obstacle proximity indicator (0-1, "
        "geometric only): "
        f"{result.obstacle_risk_indicator:.6f}"
    )

    print(
        "\nPer-obstacle minimum clearance (m):"
    )

    for (
        obstacle_id,
        clearance,
    ) in result.obstacle_clearances.items():

        print(
            f"  {obstacle_id}: "
            f"{clearance:.6f}"
        )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    assert np.isfinite(
        result.minimum_clearance
    )

    assert (
        0
        <= collision_sample_count
        <= n_samples
    )

    assert (
        0
        <= margin_violation_sample_count
        <= n_samples
    )

    if result.collision_detected:
        assert (
            collision_sample_count > 0
        )
    else:
        assert (
            collision_sample_count == 0
        )

    assert (
        0.0
        <= result.obstacle_risk_indicator
        <= 1.0
    )

    assert set(
        result.obstacle_clearances.keys()
    ) == {"O1", "O2"}

    # -------------------------------------------------------------------------
    # Verify geometric meaning of indicator
    # -------------------------------------------------------------------------

    # Collision / penetration -> maximum indicator.
    assert (
        ObstacleSafetyAnalyzer._compute_risk_indicator(
            0.0
        )
        == 1.0
    )

    assert (
        ObstacleSafetyAnalyzer._compute_risk_indicator(
            -0.1
        )
        == 1.0
    )

    # Warning-distance boundary -> zero.
    assert np.isclose(
        ObstacleSafetyAnalyzer._compute_risk_indicator(
            OBSTACLE_WARNING_DISTANCE
        ),
        0.0,
    )

    # Larger clearance -> zero.
    assert (
        ObstacleSafetyAnalyzer._compute_risk_indicator(
            OBSTACLE_WARNING_DISTANCE + 1.0
        )
        == 0.0
    )

    print(
        "\nAll sanity checks passed."
    )