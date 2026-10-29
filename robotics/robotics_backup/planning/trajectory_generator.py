"""
Trajectory generation for the 7-DOF Franka Panda robot.

Provides:
    - Direct joint-space trajectories
    - Waypoint-based trajectories
    - Diverse candidate trajectories
    - Multiple goal trajectories
    - Basic trajectory metrics
"""

import numpy as np


class TrajectoryGenerator:
    """
    Generate joint-space trajectories for the Franka Panda robot.
    """

    def __init__(self, panda_robot):
        """
        Parameters
        ----------
        panda_robot : PandaRobot
            Panda robot model.
        """

        if panda_robot is None:
            raise ValueError("panda_robot cannot be None.")

        self.panda_robot = panda_robot

        # -----------------------------------------------------
        # Panda robot DOF
        # -----------------------------------------------------

        self.dof = int(panda_robot.DOF)

        if self.dof != 7:
            raise ValueError(
                f"Expected a 7-DOF Panda robot, got DOF={self.dof}"
            )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Your PandaRobot.joint_limits is a NumPy array.
        #
        # Do NOT use:
        #
        # panda_robot.joint_limits["lower"]
        #
        # Instead, the actual structure is:
        #
        #   joint_limits[:, 0] = lower limits
        #   joint_limits[:, 1] = upper limits
        # -----------------------------------------------------

        self.joint_limits = np.asarray(
            panda_robot.joint_limits,
            dtype=float
        )

        if self.joint_limits.shape != (self.dof, 2):
            raise ValueError(
                "Expected joint_limits to have shape "
                f"({self.dof}, 2), "
                f"got {self.joint_limits.shape}"
            )

        self.lower_limits = self.joint_limits[:, 0]

        self.upper_limits = self.joint_limits[:, 1]

        if np.any(
            self.lower_limits >= self.upper_limits
        ):
            raise ValueError(
                "Invalid Panda joint limits."
            )

    # =========================================================
    # CONFIGURATION VALIDATION
    # =========================================================

    def _validate_configuration(
        self,
        configuration,
        name="configuration"
    ):
        """
        Validate a 7-DOF joint configuration.
        """

        q = np.asarray(
            configuration,
            dtype=float
        )

        if q.shape != (self.dof,):
            raise ValueError(
                f"{name} must have shape "
                f"({self.dof},), got {q.shape}"
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        tolerance = 1e-9

        if np.any(
            q < self.lower_limits - tolerance
        ):
            raise ValueError(
                f"{name} violates lower joint limits."
            )

        if np.any(
            q > self.upper_limits + tolerance
        ):
            raise ValueError(
                f"{name} violates upper joint limits."
            )

        # Protect against tiny floating point errors
        q = np.clip(
            q,
            self.lower_limits,
            self.upper_limits
        )

        return q

    # =========================================================
    # DIRECT JOINT TRAJECTORY
    # =========================================================

    def generate_joint_trajectory(
        self,
        start_configuration,
        goal_configuration,
        num_points=100
    ):
        """
        Generate a smooth trajectory directly from start to goal.

        Uses cubic smoothstep:

            s(t) = 3t² - 2t³

        Parameters
        ----------
        start_configuration : array-like
            Starting joint configuration.

        goal_configuration : array-like
            Goal joint configuration.

        num_points : int
            Number of trajectory points.

        Returns
        -------
        numpy.ndarray
            Shape (num_points, 7)
        """

        if not isinstance(
            num_points,
            (int, np.integer)
        ):
            raise TypeError(
                "num_points must be an integer."
            )

        if num_points < 2:
            raise ValueError(
                "num_points must be at least 2."
            )

        q_start = self._validate_configuration(
            start_configuration,
            "start_configuration"
        )

        q_goal = self._validate_configuration(
            goal_configuration,
            "goal_configuration"
        )

        # Time values
        t = np.linspace(
            0.0,
            1.0,
            num_points
        )

        # Cubic smoothstep
        s = (
            3.0 * t**2
            - 2.0 * t**3
        )

        # Interpolate
        trajectory = (
            q_start[None, :]
            +
            s[:, None]
            *
            (
                q_goal - q_start
            )[None, :]
        )

        # Exact endpoints
        trajectory[0] = q_start
        trajectory[-1] = q_goal

        # Numerical safety
        trajectory = np.clip(
            trajectory,
            self.lower_limits,
            self.upper_limits
        )

        return trajectory

    # =========================================================
    # WAYPOINT TRAJECTORY
    # =========================================================

    def generate_waypoint_trajectory(
        self,
        start_configuration,
        waypoints,
        goal_configuration,
        num_points=100
    ):
        """
        Generate a trajectory through zero, one, or multiple
        intermediate waypoints.

        Examples
        --------

        No waypoints:

            START ----------------------> GOAL

        One waypoint:

            START ----------> W1 ----------> GOAL

        Multiple:

            START -> W1 -> W2 -> W3 -> GOAL
        """

        if not isinstance(
            num_points,
            (int, np.integer)
        ):
            raise TypeError(
                "num_points must be an integer."
            )

        if num_points < 2:
            raise ValueError(
                "num_points must be at least 2."
            )

        # -----------------------------------------------------
        # Validate start and goal
        # -----------------------------------------------------

        q_start = self._validate_configuration(
            start_configuration,
            "start_configuration"
        )

        q_goal = self._validate_configuration(
            goal_configuration,
            "goal_configuration"
        )

        # -----------------------------------------------------
        # Empty waypoint list is valid
        # -----------------------------------------------------

        if waypoints is None:
            waypoints = []

        waypoints = list(waypoints)

        # -----------------------------------------------------
        # Validate waypoints
        # -----------------------------------------------------

        validated_waypoints = []

        for i, waypoint in enumerate(waypoints):

            q_waypoint = self._validate_configuration(
                waypoint,
                f"waypoint[{i}]"
            )

            validated_waypoints.append(
                q_waypoint
            )

        # -----------------------------------------------------
        # CASE 1:
        #
        # No waypoints
        #
        # START ----------------------> GOAL
        # -----------------------------------------------------

        if len(validated_waypoints) == 0:

            return self.generate_joint_trajectory(
                q_start,
                q_goal,
                num_points=num_points
            )

        # -----------------------------------------------------
        # Build configuration sequence
        # -----------------------------------------------------

        configurations = [
            q_start,
            *validated_waypoints,
            q_goal
        ]

        num_segments = len(configurations) - 1

        # -----------------------------------------------------
        # Need at least one interval per segment
        # -----------------------------------------------------

        if num_points < num_segments + 1:

            raise ValueError(
                f"num_points={num_points} is too small "
                f"for {num_segments} segments. "
                f"Use at least {num_segments + 1}."
            )

        # -----------------------------------------------------
        # Allocate intervals across segments
        # -----------------------------------------------------

        total_intervals = num_points - 1

        base_intervals = (
            total_intervals // num_segments
        )

        remainder = (
            total_intervals % num_segments
        )

        segment_intervals = []

        for i in range(num_segments):

            intervals = base_intervals

            if i < remainder:
                intervals += 1

            segment_intervals.append(
                intervals
            )

        # -----------------------------------------------------
        # Generate individual segments
        # -----------------------------------------------------

        trajectory_segments = []

        for i in range(num_segments):

            q_segment_start = configurations[i]

            q_segment_goal = configurations[i + 1]

            segment_num_points = (
                segment_intervals[i] + 1
            )

            segment = (
                self.generate_joint_trajectory(
                    q_segment_start,
                    q_segment_goal,
                    num_points=segment_num_points
                )
            )

            # Remove duplicate boundary point
            if i > 0:
                segment = segment[1:]

            trajectory_segments.append(
                segment
            )

        # -----------------------------------------------------
        # Combine
        # -----------------------------------------------------

        trajectory = np.vstack(
            trajectory_segments
        )

        expected_shape = (
            num_points,
            self.dof
        )

        if trajectory.shape != expected_shape:

            raise RuntimeError(
                f"Generated trajectory has shape "
                f"{trajectory.shape}, "
                f"expected {expected_shape}."
            )

        # Exact endpoints
        trajectory[0] = q_start
        trajectory[-1] = q_goal

        # Final limit validation
        if np.any(
            trajectory
            < self.lower_limits - 1e-9
        ):
            raise RuntimeError(
                "Generated trajectory violates "
                "lower joint limits."
            )

        if np.any(
            trajectory
            > self.upper_limits + 1e-9
        ):
            raise RuntimeError(
                "Generated trajectory violates "
                "upper joint limits."
            )

        return trajectory

    # =========================================================
    # RANDOM WAYPOINT GENERATION
    # =========================================================

    def _generate_waypoints(
        self,
        start_configuration,
        goal_configuration,
        num_waypoints=1,
        seed=42,
        perturbation_scale=0.8
    ):
        """
        Generate random intermediate joint configurations.
        """

        q_start = self._validate_configuration(
            start_configuration,
            "start_configuration"
        )

        q_goal = self._validate_configuration(
            goal_configuration,
            "goal_configuration"
        )

        if not isinstance(
            num_waypoints,
            (int, np.integer)
        ):
            raise TypeError(
                "num_waypoints must be an integer."
            )

        if num_waypoints < 0:
            raise ValueError(
                "num_waypoints cannot be negative."
            )

        if perturbation_scale < 0:
            raise ValueError(
                "perturbation_scale cannot be negative."
            )

        if num_waypoints == 0:
            return []

        rng = np.random.default_rng(
            seed
        )

        joint_ranges = (
            self.upper_limits
            -
            self.lower_limits
        )

        waypoints = []

        for i in range(num_waypoints):

            # Position between start and goal
            alpha = (
                (i + 1)
                /
                (num_waypoints + 1)
            )

            base_point = (
                (1.0 - alpha)
                * q_start
                +
                alpha
                * q_goal
            )

            # Random perturbation
            perturbation = (
                rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=self.dof
                )
                *
                joint_ranges
                *
                0.25
                *
                perturbation_scale
            )

            waypoint = (
                base_point
                +
                perturbation
            )

            # Respect joint limits
            waypoint = np.clip(
                waypoint,
                self.lower_limits,
                self.upper_limits
            )

            waypoints.append(
                waypoint
            )

        return waypoints

    # =========================================================
    # DIVERSE TRAJECTORIES
    # =========================================================

    def generate_diverse_trajectories(
        self,
        start_configuration,
        goal_configuration,
        num_candidates=10,
        num_points=100,
        seed=42,
        perturbation_scale=0.8
    ):
        """
        Generate many diverse candidate trajectories.

        This generates a configurable set of candidate
        trajectories rather than hard-coding exactly three.
        """

        if not isinstance(
            num_candidates,
            (int, np.integer)
        ):
            raise TypeError(
                "num_candidates must be an integer."
            )

        if num_candidates <= 0:
            raise ValueError(
                "num_candidates must be greater than zero."
            )

        q_start = self._validate_configuration(
            start_configuration,
            "start_configuration"
        )

        q_goal = self._validate_configuration(
            goal_configuration,
            "goal_configuration"
        )

        trajectories = []

        # -----------------------------------------------------
        # Candidate 1: direct trajectory
        # -----------------------------------------------------

        direct_trajectory = (
            self.generate_joint_trajectory(
                q_start,
                q_goal,
                num_points=num_points
            )
        )

        trajectories.append(
            direct_trajectory
        )

        # -----------------------------------------------------
        # Remaining candidates
        # -----------------------------------------------------

        for candidate_index in range(
            1,
            num_candidates
        ):

            candidate_seed = (
                seed
                +
                candidate_index
            )

            # Vary waypoint count:
            #
            # 1, 2, 3, 1, 2, 3, ...
            num_waypoints = (
                1
                +
                (
                    candidate_index - 1
                )
                % 3
            )

            waypoints = (
                self._generate_waypoints(
                    q_start,
                    q_goal,
                    num_waypoints=num_waypoints,
                    seed=candidate_seed,
                    perturbation_scale=perturbation_scale
                )
            )

            trajectory = (
                self.generate_waypoint_trajectory(
                    q_start,
                    waypoints,
                    q_goal,
                    num_points=num_points
                )
            )

            trajectories.append(
                trajectory
            )

        return trajectories

    # =========================================================
    # MULTIPLE GOAL TRAJECTORIES
    # =========================================================

    def generate_multiple_trajectories(
        self,
        start_configuration,
        goal_configurations,
        num_points=100
    ):
        """
        Generate one direct trajectory for each goal
        configuration.
        """

        if goal_configurations is None:
            raise ValueError(
                "goal_configurations cannot be None."
            )

        goal_configurations = list(
            goal_configurations
        )

        if len(goal_configurations) == 0:
            raise ValueError(
                "goal_configurations cannot be empty."
            )

        q_start = self._validate_configuration(
            start_configuration,
            "start_configuration"
        )

        trajectories = []

        for i, goal in enumerate(
            goal_configurations
        ):

            q_goal = self._validate_configuration(
                goal,
                f"goal_configurations[{i}]"
            )

            trajectory = (
                self.generate_joint_trajectory(
                    q_start,
                    q_goal,
                    num_points=num_points
                )
            )

            trajectories.append(
                trajectory
            )

        return trajectories

    # =========================================================
    # CANDIDATE TRAJECTORIES
    # =========================================================

    def generate_candidate_trajectories(
        self,
        start_configuration,
        goal_configuration,
        num_candidates=10,
        num_points=100,
        seed=42
    ):
        """
        Convenience wrapper for generating candidate
        trajectories.
        """

        return (
            self.generate_diverse_trajectories(
                start_configuration=start_configuration,
                goal_configuration=goal_configuration,
                num_candidates=num_candidates,
                num_points=num_points,
                seed=seed
            )
        )

    # =========================================================
    # JOINT-SPACE LENGTH
    # =========================================================

    def calculate_joint_space_length(
        self,
        trajectory
    ):
        """
        Calculate total joint-space path length.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory must be a 2D array."
            )

        if trajectory.shape[1] != self.dof:
            raise ValueError(
                f"trajectory must have {self.dof} "
                "joint columns."
            )

        if len(trajectory) < 2:
            return 0.0

        differences = np.diff(
            trajectory,
            axis=0
        )

        step_lengths = np.linalg.norm(
            differences,
            axis=1
        )

        return float(
            np.sum(step_lengths)
        )

    # =========================================================
    # MAXIMUM JOINT STEP
    # =========================================================

    def calculate_max_joint_step(
        self,
        trajectory
    ):
        """
        Calculate the largest joint-space step.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory must be a 2D array."
            )

        if trajectory.shape[1] != self.dof:
            raise ValueError(
                f"trajectory must have {self.dof} "
                "joint columns."
            )

        if len(trajectory) < 2:
            return 0.0

        differences = np.diff(
            trajectory,
            axis=0
        )

        step_lengths = np.linalg.norm(
            differences,
            axis=1
        )

        return float(
            np.max(step_lengths)
        )

    # =========================================================
    # VELOCITIES
    # =========================================================

    def calculate_joint_velocities(
        self,
        trajectory,
        time_step=1.0
    ):
        """
        Estimate joint velocities using finite differences.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory must be a 2D array."
            )

        if trajectory.shape[1] != self.dof:
            raise ValueError(
                f"trajectory must have {self.dof} "
                "joint columns."
            )

        if time_step <= 0:
            raise ValueError(
                "time_step must be greater than zero."
            )

        velocities = np.gradient(
            trajectory,
            time_step,
            axis=0
        )

        return velocities

    # =========================================================
    # ACCELERATIONS
    # =========================================================

    def calculate_joint_accelerations(
        self,
        trajectory,
        time_step=1.0
    ):
        """
        Estimate joint accelerations.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory must be a 2D array."
            )

        if trajectory.shape[1] != self.dof:
            raise ValueError(
                f"trajectory must have {self.dof} "
                "joint columns."
            )

        if time_step <= 0:
            raise ValueError(
                "time_step must be greater than zero."
            )

        velocities = (
            self.calculate_joint_velocities(
                trajectory,
                time_step=time_step
            )
        )

        accelerations = np.gradient(
            velocities,
            time_step,
            axis=0
        )

        return accelerations

    # =========================================================
    # TRAJECTORY SUMMARY
    # =========================================================

    def summarize_trajectory(
        self,
        trajectory,
        time_step=1.0
    ):
        """
        Calculate basic trajectory metrics.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory must be a 2D array."
            )

        if trajectory.shape[1] != self.dof:
            raise ValueError(
                f"trajectory must have {self.dof} "
                "joint columns."
            )

        velocities = (
            self.calculate_joint_velocities(
                trajectory,
                time_step=time_step
            )
        )

        accelerations = (
            self.calculate_joint_accelerations(
                trajectory,
                time_step=time_step
            )
        )

        return {
            "num_points": int(
                len(trajectory)
            ),

            "joint_space_length": (
                self.calculate_joint_space_length(
                    trajectory
                )
            ),

            "max_joint_step": (
                self.calculate_max_joint_step(
                    trajectory
                )
            ),

            "max_joint_velocity": float(
                np.max(
                    np.abs(velocities)
                )
            ),

            "max_joint_acceleration": float(
                np.max(
                    np.abs(accelerations)
                )
            ),

            "start_configuration": (
                trajectory[0].copy()
            ),

            "goal_configuration": (
                trajectory[-1].copy()
            )
        }