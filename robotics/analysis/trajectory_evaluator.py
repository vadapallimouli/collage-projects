import numpy as np


class TrajectoryEvaluator:
    """
    Evaluates candidate Panda trajectories using the existing
    collision and clearance analysis modules.
    """

    def __init__(
        self,
        panda_robot,
        environment,
        collision_checker,
        clearance_analyzer,
    ):
        self.panda_robot = panda_robot
        self.environment = environment
        self.collision_checker = collision_checker
        self.clearance_analyzer = clearance_analyzer

    def _validate_trajectory(self, trajectory):
        """
        Validate trajectory shape and values.

        Expected shape:
            (N, 7)
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "Trajectory must be a 2D array of shape (N, 7)."
            )

        if trajectory.shape[1] != self.panda_robot.get_dof():
            raise ValueError(
                f"Trajectory must have "
                f"{self.panda_robot.get_dof()} joints."
            )

        if trajectory.shape[0] < 2:
            raise ValueError(
                "Trajectory must contain at least 2 points."
            )

        if not np.all(np.isfinite(trajectory)):
            raise ValueError(
                "Trajectory contains non-finite values."
            )

        return trajectory

    def evaluate_trajectory(self, trajectory):
        """
        Evaluate one complete trajectory.

        Returns:
            dict containing collision, clearance, joint-limit,
            and basic trajectory features.
        """

        trajectory = self._validate_trajectory(
            trajectory
        )

        # -----------------------------------------------------
        # Collision analysis
        # -----------------------------------------------------

        collision_result = (
            self.collision_checker.check_trajectory(
                trajectory
            )
        )

        # -----------------------------------------------------
        # Clearance analysis
        # -----------------------------------------------------

        clearance_result = (
            self.clearance_analyzer
            .calculate_trajectory_clearance(
                trajectory
            )
        )

        # -----------------------------------------------------
        # Collision status
        # -----------------------------------------------------

        collision = not collision_result[
            "collision_free"
        ]

        self_collision = (
            collision_result[
                "self_collision_count"
            ] > 0
        )

        obstacle_collision = (
            collision_result[
                "obstacle_collision_count"
            ] > 0
        )

        joint_limit_violation = (
            collision_result[
                "joint_limit_violation_count"
            ] > 0
        )

        # -----------------------------------------------------
        # Additional joint-limit verification
        # -----------------------------------------------------

        for q in trajectory:

            if not self.panda_robot.is_within_joint_limits(q):

                joint_limit_violation = True

                break

        # -----------------------------------------------------
        # Basic trajectory measurements
        # -----------------------------------------------------

        joint_steps = np.diff(
            trajectory,
            axis=0
        )

        # Total distance travelled in joint space
        joint_space_length = np.sum(
            np.linalg.norm(
                joint_steps,
                axis=1
            )
        )

        # Largest change of any joint between
        # two consecutive trajectory points
        max_joint_step = np.max(
            np.abs(joint_steps)
        )

        # -----------------------------------------------------
        # Overall validity
        # -----------------------------------------------------

        valid = (
            not collision
            and not joint_limit_violation
        )

        # -----------------------------------------------------
        # Unified evaluation result
        # -----------------------------------------------------

        return {

            # -------------------------------------------------
            # Collision status
            # -------------------------------------------------

            "collision": collision,

            "self_collision": self_collision,

            "obstacle_collision": obstacle_collision,

            "joint_limit_violation": (
                joint_limit_violation
            ),

            # -------------------------------------------------
            # Collision counts
            # -------------------------------------------------

            "collision_count": (
                collision_result[
                    "collision_count"
                ]
            ),

            "self_collision_count": (
                collision_result[
                    "self_collision_count"
                ]
            ),

            "obstacle_collision_count": (
                collision_result[
                    "obstacle_collision_count"
                ]
            ),

            "joint_limit_violation_count": (
                collision_result[
                    "joint_limit_violation_count"
                ]
            ),

            # -------------------------------------------------
            # Clearance
            # -------------------------------------------------

            "minimum_clearance": (
                clearance_result[
                    "minimum_clearance"
                ]
            ),

            "closest_obstacle_index": (
                clearance_result[
                    "closest_obstacle_index"
                ]
            ),

            # -------------------------------------------------
            # Trajectory features
            # -------------------------------------------------

            "joint_space_length": (
                joint_space_length
            ),

            "max_joint_step": (
                max_joint_step
            ),

            "num_points": (
                trajectory.shape[0]
            ),

            # -------------------------------------------------
            # Final validity
            # -------------------------------------------------

            "valid": valid,
        }

    def evaluate_multiple(self, trajectories):
        """
        Evaluate multiple candidate trajectories.

        Parameters
        ----------
        trajectories : list
            List of trajectories, each with shape (N, 7).

        Returns
        -------
        list
            One evaluation dictionary per trajectory.
        """

        if len(trajectories) == 0:
            raise ValueError(
                "trajectories cannot be empty."
            )

        results = []

        for index, trajectory in enumerate(
            trajectories
        ):

            result = self.evaluate_trajectory(
                trajectory
            )

            result["trajectory_index"] = index

            results.append(result)

        return results