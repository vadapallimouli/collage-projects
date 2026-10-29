import numpy as np
import pybullet as p


class GoalConfigurationGenerator:
    """
    Generate multiple valid Panda goal configurations.

    A goal is considered valid when the Panda end-effector
    enters the specified target volume.

    Exact end-effector orientation is not required.
    """

    def __init__(
        self,
        panda_robot,
        environment,
        target_position=None,
        target_half_extents=None,
        client_id=None,
        position_tolerance=0.02,
    ):

        self.robot = panda_robot
        self.environment = environment

        # -----------------------------------------------------
        # Target position
        # -----------------------------------------------------

        if target_position is None:
            target_position = environment.target_position

        self.target_position = np.asarray(
            target_position,
            dtype=float
        )

        # -----------------------------------------------------
        # Target volume
        # -----------------------------------------------------

        if target_half_extents is None:
            target_half_extents = environment.target_half_extents

        self.target_half_extents = np.asarray(
            target_half_extents,
            dtype=float
        )

        # -----------------------------------------------------
        # PyBullet client
        # -----------------------------------------------------

        if client_id is None:
            raise ValueError(
                "client_id must be provided."
            )

        self.client_id = client_id

        self.position_tolerance = float(
            position_tolerance
        )

        # -----------------------------------------------------
        # Panda joint limits
        # -----------------------------------------------------

        joint_limits = self.robot.get_joint_limits()

        self.lower_limits = np.array(
            [limit[0] for limit in joint_limits],
            dtype=float
        )

        self.upper_limits = np.array(
            [limit[1] for limit in joint_limits],
            dtype=float
        )

        self.joint_ranges = (
            self.upper_limits
            - self.lower_limits
        )

    # =========================================================
    # Target volume checking
    # =========================================================

    def is_inside_target_volume(self, position):
        """
        Check whether an end-effector position lies inside
        the target volume.

        A small tolerance is allowed for numerical IK error.
        """

        position = np.asarray(
            position,
            dtype=float
        )

        lower = (
            self.target_position
            - self.target_half_extents
            - self.position_tolerance
        )

        upper = (
            self.target_position
            + self.target_half_extents
            + self.position_tolerance
        )

        return bool(
            np.all(position >= lower)
            and np.all(position <= upper)
        )

    # =========================================================
    # Position error
    # =========================================================

    def position_error(self, position):
        """
        Euclidean distance from the achieved position to the
        center of the target volume.
        """

        position = np.asarray(
            position,
            dtype=float
        )

        return float(
            np.linalg.norm(
                position - self.target_position
            )
        )

    # =========================================================
    # Generate IK rest poses
    # =========================================================

    def _generate_rest_poses(
        self,
        num_poses,
        seed=42
    ):
        """
        Generate different starting configurations for IK.

        These are starting points for the IK solver, not
        final goal configurations.
        """

        rng = np.random.default_rng(seed)

        poses = []

        # -----------------------------------------------------
        # Standard Panda configuration
        # -----------------------------------------------------

        default_pose = np.array(
            [
                0.0,
                -0.4,
                0.0,
                -2.0,
                0.0,
                1.6,
                0.8
            ],
            dtype=float
        )

        default_pose = np.clip(
            default_pose,
            self.lower_limits + 1e-6,
            self.upper_limits - 1e-6
        )

        poses.append(default_pose)

        # -----------------------------------------------------
        # Random configurations
        # -----------------------------------------------------

        for _ in range(
            max(0, num_poses - 1)
        ):

            pose = rng.uniform(
                self.lower_limits + 0.05,
                self.upper_limits - 0.05
            )

            poses.append(pose)

        return poses

    # =========================================================
    # Solve one IK problem
    # =========================================================

    def _solve_from_rest_pose(
        self,
        rest_pose
    ):
        """
        Solve position-only IK using a supplied rest pose.
        """

        q = p.calculateInverseKinematics(
            self.robot.robot_id,
            self.robot.END_EFFECTOR_LINK_INDEX,
            targetPosition=self.target_position.tolist(),
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=self.joint_ranges.tolist(),
            restPoses=rest_pose.tolist(),
            physicsClientId=self.client_id
        )

        q = np.asarray(
            q[:self.robot.DOF],
            dtype=float
        )

        # -----------------------------------------------------
        # Numerical validation
        # -----------------------------------------------------

        if not np.all(np.isfinite(q)):
            return None

        # -----------------------------------------------------
        # Keep inside joint limits
        # -----------------------------------------------------

        q = np.clip(
            q,
            self.lower_limits + 1e-6,
            self.upper_limits - 1e-6
        )

        if not self.robot.is_within_joint_limits(q):
            return None

        # -----------------------------------------------------
        # Apply configuration
        # -----------------------------------------------------

        self.robot.set_joint_positions(q)

        achieved_position = np.asarray(
            self.robot.get_end_effector_position(),
            dtype=float
        )

        error = self.position_error(
            achieved_position
        )

        # -----------------------------------------------------
        # Check target volume
        # -----------------------------------------------------

        if not self.is_inside_target_volume(
            achieved_position
        ):
            return None

        return {
            "configuration": q.copy(),
            "position": achieved_position.copy(),
            "position_error": error,
        }

    # =========================================================
    # Remove duplicate configurations
    # =========================================================

    def _remove_duplicates(
        self,
        solutions,
        tolerance=1e-3
    ):
        """
        Remove nearly identical joint configurations.
        """

        unique = []

        for solution in solutions:

            q = solution["configuration"]

            duplicate = False

            for existing in unique:

                existing_q = (
                    existing["configuration"]
                )

                distance = np.linalg.norm(
                    q - existing_q
                )

                if distance < tolerance:
                    duplicate = True
                    break

            if not duplicate:
                unique.append(solution)

        return unique

    # =========================================================
    # Generate multiple goal configurations
    # =========================================================

    def solve_multiple(
        self,
        num_solutions=10,
        num_rest_poses=None,
        seed=42
    ):
        """
        Generate multiple valid goal configurations.

        Parameters
        ----------
        num_solutions:
            Maximum number of unique goals to return.

        num_rest_poses:
            Number of different IK starting configurations.

        seed:
            Random seed for reproducibility.
        """

        if num_rest_poses is None:
            num_rest_poses = max(
                30,
                num_solutions * 5
            )

        rest_poses = self._generate_rest_poses(
            num_rest_poses,
            seed=seed
        )

        solutions = []

        for rest_pose in rest_poses:

            result = self._solve_from_rest_pose(
                rest_pose
            )

            if result is not None:
                solutions.append(result)

        # -----------------------------------------------------
        # Remove duplicates
        # -----------------------------------------------------

        solutions = self._remove_duplicates(
            solutions
        )

        # -----------------------------------------------------
        # Sort by distance to target center
        # -----------------------------------------------------

        solutions.sort(
            key=lambda x: x["position_error"]
        )

        return solutions[:num_solutions]

    # =========================================================
    # Generate one goal
    # =========================================================

    def solve(self):
        """
        Generate one valid goal configuration.

        Returns
        -------
        dict or None
        """

        solutions = self.solve_multiple(
            num_solutions=1,
            num_rest_poses=20
        )

        if not solutions:
            return None

        return solutions[0]