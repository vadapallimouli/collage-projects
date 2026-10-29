"""
Clearance analysis for the Panda robot.

Clearance is measured between the moving robot links and
environment obstacles using PyBullet.

Important:
- CollisionChecker determines collision/failure.
- ClearanceAnalyzer provides geometric features.
- Clearance is NOT the ML failure label.
- The fixed Panda base link (-1) is excluded from clearance
  because it does not move with the robot joints and can
  otherwise dominate the minimum clearance.
"""

import numpy as np
import pybullet as p


class ClearanceAnalyzer:
    """
    Computes robot-obstacle clearance for configurations
    and trajectories.
    """

    def __init__(
        self,
        panda_robot,
        environment,
        client_id
    ):

        self.panda_robot = panda_robot
        self.environment = environment
        self.client_id = client_id

        # Maximum distance queried from PyBullet.
        self.query_distance = 10.0

        # Descriptive threshold only.
        # This is NOT a failure threshold.
        self.low_clearance_threshold = 0.02

    # ============================================================
    # OBSTACLE IDENTIFICATION
    # ============================================================

    def _get_obstacle_ids(self):
        """
        Find PyBullet body IDs corresponding to environment
        obstacles.
        """

        possible_attributes = [
            "obstacle_body_ids",
            "obstacle_ids",
            "pybullet_obstacle_ids",
            "pybullet_obstacles",
            "obstacle_body_id",
        ]

        for attribute in possible_attributes:

            if not hasattr(
                self.environment,
                attribute
            ):
                continue

            value = getattr(
                self.environment,
                attribute
            )

            if value is None:
                continue

            if isinstance(
                value,
                (int, np.integer)
            ):
                return [int(value)]

            try:

                ids = [
                    int(x)
                    for x in value
                ]

                if len(ids) > 0:
                    return ids

            except (
                TypeError,
                ValueError
            ):

                pass

        # --------------------------------------------------------
        # Fallback: inspect PyBullet bodies.
        # --------------------------------------------------------

        try:

            num_bodies = p.getNumBodies(
                physicsClientId=self.client_id
            )

        except TypeError:

            num_bodies = p.getNumBodies(
                self.client_id
            )

        robot_id = getattr(
            self.panda_robot,
            "robot_id",
            None
        )

        obstacle_ids = []

        for body_id in range(
            num_bodies
        ):

            # Exclude robot body.
            if (
                robot_id is not None
                and body_id == robot_id
            ):
                continue

            # Exclude plane.
            try:

                body_info = p.getBodyInfo(
                    body_id,
                    physicsClientId=self.client_id
                )

                names = []

                for item in body_info:

                    if isinstance(
                        item,
                        bytes
                    ):

                        names.append(
                            item.decode(
                                "utf-8",
                                errors="ignore"
                            ).lower()
                        )

                    else:

                        names.append(
                            str(item).lower()
                        )

                body_name = " ".join(
                    names
                )

                if "plane" in body_name:
                    continue

            except Exception:

                pass

            obstacle_ids.append(
                body_id
            )

        return obstacle_ids

    # ============================================================
    # CONFIGURATION CLEARANCE
    # ============================================================

    def calculate_configuration_clearance(
        self,
        configuration
    ):
        """
        Calculate minimum clearance between moving Panda links
        and environment obstacles.

        The fixed Panda base link (-1) is excluded.
        """

        q = np.asarray(
            configuration,
            dtype=float
        )

        # --------------------------------------------------------
        # Validate shape.
        # --------------------------------------------------------

        if q.shape != (
            self.panda_robot.DOF,
        ):

            raise ValueError(
                f"Configuration must have shape "
                f"({self.panda_robot.DOF},), "
                f"got {q.shape}"
            )

        # --------------------------------------------------------
        # Validate joint limits.
        # --------------------------------------------------------

        joint_limits = np.asarray(
            self.panda_robot.joint_limits,
            dtype=float
        )

        if (
            np.any(
                q < joint_limits[:, 0]
            )
            or
            np.any(
                q > joint_limits[:, 1]
            )
        ):

            return {
                "minimum_clearance": -np.inf,
                "closest_obstacle_index": None,
                "closest_robot_link": None,
                "valid": False,
            }

        # ========================================================
        # SET ROBOT CONFIGURATION
        # ========================================================

        for joint_index, joint_value in zip(
            self.panda_robot.ARM_JOINT_INDICES,
            q
        ):

            p.resetJointState(
                self.panda_robot.robot_id,
                joint_index,
                float(joint_value),
                targetVelocity=0.0,
                physicsClientId=self.client_id
            )

        # --------------------------------------------------------
        # Verify joint state.
        # --------------------------------------------------------

        actual_positions = np.array(
            [
                p.getJointState(
                    self.panda_robot.robot_id,
                    joint_index,
                    physicsClientId=self.client_id
                )[0]

                for joint_index
                in self.panda_robot.ARM_JOINT_INDICES
            ],
            dtype=float
        )

        if not np.allclose(
            actual_positions,
            q,
            atol=1e-6
        ):

            raise RuntimeError(
                "PyBullet joint state does not match "
                "requested configuration.\n"
                f"Requested: {q}\n"
                f"Actual:    {actual_positions}"
            )

        # --------------------------------------------------------
        # Update collision geometry.
        # --------------------------------------------------------

        try:

            p.performCollisionDetection(
                physicsClientId=self.client_id
            )

        except TypeError:

            p.performCollisionDetection(
                self.client_id
            )

        # ========================================================
        # GET OBSTACLES
        # ========================================================

        obstacle_ids = (
            self._get_obstacle_ids()
        )

        if len(obstacle_ids) == 0:

            return {
                "minimum_clearance":
                    self.query_distance,

                "closest_obstacle_index":
                    None,

                "closest_robot_link":
                    None,

                "valid":
                    True,
            }

        # ========================================================
        # CALCULATE CLEARANCE
        # ========================================================

        minimum_clearance = (
            self.query_distance
        )

        closest_obstacle_index = None
        closest_robot_link = None

        for obstacle_index, obstacle_id in enumerate(
            obstacle_ids
        ):

            try:

                points = p.getClosestPoints(
                    bodyA=self.panda_robot.robot_id,
                    bodyB=obstacle_id,
                    distance=self.query_distance,
                    physicsClientId=self.client_id
                )

            except TypeError:

                points = p.getClosestPoints(
                    self.panda_robot.robot_id,
                    obstacle_id,
                    self.query_distance,
                    physicsClientId=self.client_id
                )

            if not points:
                continue

            # ----------------------------------------------------
            # IMPORTANT:
            # Ignore linkIndexA == -1.
            #
            # This is the fixed Panda base link.
            # ----------------------------------------------------

            moving_link_points = [
                point
                for point in points
                if point[3] != -1
            ]

            if not moving_link_points:
                continue

            # ----------------------------------------------------
            # Find closest moving-link point.
            # ----------------------------------------------------

            closest_point = min(
                moving_link_points,
                key=lambda point: float(point[8])
            )

            clearance = float(
                closest_point[8]
            )

            if clearance < minimum_clearance:

                minimum_clearance = clearance

                closest_obstacle_index = (
                    obstacle_index
                )

                closest_robot_link = (
                    int(closest_point[3])
                )

        return {
            "minimum_clearance":
                float(minimum_clearance),

            "closest_obstacle_index":
                closest_obstacle_index,

            "closest_robot_link":
                closest_robot_link,

            "valid":
                True,
        }

    # ============================================================
    # TRAJECTORY CLEARANCE
    # ============================================================

    def calculate_trajectory_clearance(
        self,
        trajectory,
        include_endpoints=True
    ):
        """
        Calculate clearance statistics for a trajectory.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        # --------------------------------------------------------
        # Validate trajectory.
        # --------------------------------------------------------

        if trajectory.ndim != 2:

            raise ValueError(
                f"Trajectory must be 2D, "
                f"got shape {trajectory.shape}"
            )

        if trajectory.shape[1] != (
            self.panda_robot.DOF
        ):

            raise ValueError(
                f"Trajectory must have "
                f"{self.panda_robot.DOF} joints, "
                f"got {trajectory.shape[1]}"
            )

        num_configurations = len(
            trajectory
        )

        if num_configurations == 0:

            raise ValueError(
                "Trajectory cannot be empty."
            )

        # --------------------------------------------------------
        # Calculate clearance at every configuration.
        # --------------------------------------------------------

        clearances = []
        closest_obstacles = []
        closest_robot_links = []
        valid_flags = []

        for q in trajectory:

            result = (
                self.calculate_configuration_clearance(
                    q
                )
            )

            clearances.append(
                float(
                    result[
                        "minimum_clearance"
                    ]
                )
            )

            closest_obstacles.append(
                result[
                    "closest_obstacle_index"
                ]
            )

            closest_robot_links.append(
                result[
                    "closest_robot_link"
                ]
            )

            valid_flags.append(
                bool(
                    result["valid"]
                )
            )

        clearances = np.asarray(
            clearances,
            dtype=float
        )

        # ========================================================
        # FULL TRAJECTORY STATISTICS
        # ========================================================

        if include_endpoints:

            analysis_clearances = (
                clearances
            )

            analysis_offset = 0

        else:

            if num_configurations > 2:

                analysis_clearances = (
                    clearances[1:-1]
                )

                analysis_offset = 1

            else:

                analysis_clearances = (
                    clearances
                )

                analysis_offset = 0

        if len(
            analysis_clearances
        ) == 0:

            analysis_clearances = (
                clearances
            )

            analysis_offset = 0

        minimum_index_local = int(
            np.argmin(
                analysis_clearances
            )
        )

        minimum_clearance = float(
            np.min(
                analysis_clearances
            )
        )

        minimum_clearance_configuration = (
            minimum_index_local
            + analysis_offset
        )

        closest_obstacle_index = (
            closest_obstacles[
                minimum_clearance_configuration
            ]
        )

        closest_robot_link = (
            closest_robot_links[
                minimum_clearance_configuration
            ]
        )

        mean_clearance = float(
            np.mean(
                analysis_clearances
            )
        )

        clearance_std = float(
            np.std(
                analysis_clearances
            )
        )

        maximum_clearance = float(
            np.max(
                analysis_clearances
            )
        )

        low_clearance_count = int(
            np.sum(
                analysis_clearances
                < self.low_clearance_threshold
            )
        )

        low_clearance_fraction = float(
            low_clearance_count
            / len(analysis_clearances)
        )

        # ========================================================
        # INTERIOR STATISTICS
        # ========================================================

        if num_configurations > 2:

            interior_clearances = (
                clearances[1:-1]
            )

            interior_min_index = int(
                np.argmin(
                    interior_clearances
                )
            )

            interior_minimum_clearance = float(
                np.min(
                    interior_clearances
                )
            )

            interior_minimum_configuration = (
                interior_min_index + 1
            )

            interior_mean_clearance = float(
                np.mean(
                    interior_clearances
                )
            )

            interior_clearance_std = float(
                np.std(
                    interior_clearances
                )
            )

            interior_maximum_clearance = float(
                np.max(
                    interior_clearances
                )
            )

            interior_low_clearance_count = int(
                np.sum(
                    interior_clearances
                    < self.low_clearance_threshold
                )
            )

            interior_low_clearance_fraction = float(
                interior_low_clearance_count
                / len(interior_clearances)
            )

        else:

            interior_minimum_clearance = (
                minimum_clearance
            )

            interior_minimum_configuration = (
                minimum_clearance_configuration
            )

            interior_mean_clearance = (
                mean_clearance
            )

            interior_clearance_std = (
                clearance_std
            )

            interior_maximum_clearance = (
                maximum_clearance
            )

            interior_low_clearance_count = (
                low_clearance_count
            )

            interior_low_clearance_fraction = (
                low_clearance_fraction
            )

        # ========================================================
        # RETURN
        # ========================================================

        return {

            "minimum_clearance":
                minimum_clearance,

            "clearances":
                clearances,

            "minimum_clearance_configuration":
                minimum_clearance_configuration,

            "closest_obstacle_index":
                closest_obstacle_index,

            "closest_robot_link":
                closest_robot_link,

            "mean_clearance":
                mean_clearance,

            "clearance_std":
                clearance_std,

            "maximum_clearance":
                maximum_clearance,

            "low_clearance_count":
                low_clearance_count,

            "low_clearance_fraction":
                low_clearance_fraction,

            "interior_minimum_clearance":
                interior_minimum_clearance,

            "interior_minimum_configuration":
                interior_minimum_configuration,

            "interior_mean_clearance":
                interior_mean_clearance,

            "interior_clearance_std":
                interior_clearance_std,

            "interior_maximum_clearance":
                interior_maximum_clearance,

            "interior_low_clearance_count":
                interior_low_clearance_count,

            "interior_low_clearance_fraction":
                interior_low_clearance_fraction,

            "num_configurations":
                num_configurations,

            "num_interior_configurations":
                max(
                    0,
                    num_configurations - 2
                ),

            "valid":
                bool(
                    all(valid_flags)
                ),
        }