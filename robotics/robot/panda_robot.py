from __future__ import annotations

import numpy as np
import pybullet as p


class PandaRobot:
    """
    Robot interface for the Franka Panda 7-DOF arm.

    This class contains robot-specific information and basic
    kinematic access. It does not perform path planning or
    failure prediction.
    """

    # The 7 revolute arm joints used for motion planning.
    ARM_JOINT_INDICES = tuple(range(7))

    DOF = 7

    # PyBullet Panda hand/link used as the end-effector.
    END_EFFECTOR_LINK_INDEX = 11

    def __init__(self, robot_id: int):
        self.robot_id = int(robot_id)

        if self.robot_id < 0:
            raise ValueError("Invalid PyBullet robot ID.")

        self.joint_limits = self._read_joint_limits()

    # ------------------------------------------------------------------
    # Robot information
    # ------------------------------------------------------------------

    def get_dof(self) -> int:
        """Return the number of controllable arm joints."""

        return self.DOF

    def get_joint_indices(self) -> tuple[int, ...]:
        """Return the PyBullet indices of the 7 arm joints."""

        return self.ARM_JOINT_INDICES

    # ------------------------------------------------------------------
    # Joint limits
    # ------------------------------------------------------------------

    def _read_joint_limits(self) -> np.ndarray:
        """
        Read lower and upper limits directly from the Panda URDF
        through PyBullet.

        Returns
        -------
        np.ndarray
            Shape (7, 2):

            [[lower_1, upper_1],
             ...
             [lower_7, upper_7]]
        """

        limits = []

        for joint_index in self.ARM_JOINT_INDICES:
            info = p.getJointInfo(
                self.robot_id,
                joint_index,
            )

            lower = float(info[8])
            upper = float(info[9])

            limits.append([lower, upper])

        return np.asarray(limits, dtype=float)

    def get_joint_limits(self) -> np.ndarray:
        """Return a copy of the 7 joint limits."""

        return self.joint_limits.copy()

    def is_within_joint_limits(self, q) -> bool:
        """Check whether a 7-joint configuration is valid."""

        q = self._validate_configuration(q)

        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]

        return bool(
            np.all(q >= lower)
            and np.all(q <= upper)
        )

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Return the current 7-joint configuration."""

        states = p.getJointStates(
            self.robot_id,
            list(self.ARM_JOINT_INDICES),
        )

        return np.asarray(
            [state[0] for state in states],
            dtype=float,
        )

    def set_joint_positions(self, q) -> None:
        """
        Set the Panda arm to a configuration.

        Parameters
        ----------
        q : array-like
            Seven joint angles in radians.
        """

        q = self._validate_configuration(q)

        if not self.is_within_joint_limits(q):
            raise ValueError(
                "Joint configuration is outside the Panda joint limits."
            )

        for joint_index, angle in zip(
            self.ARM_JOINT_INDICES,
            q,
        ):
            p.resetJointState(
                self.robot_id,
                joint_index,
                float(angle),
            )

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def get_end_effector_pose(self):
        """
        Return the end-effector position and orientation.

        Returns
        -------
        position : np.ndarray
            Shape (3,), XYZ position in metres.

        orientation : np.ndarray
            Shape (4,), quaternion [x, y, z, w].
        """

        state = p.getLinkState(
            self.robot_id,
            self.END_EFFECTOR_LINK_INDEX,
            computeForwardKinematics=True,
        )

        position = np.asarray(
            state[4],
            dtype=float,
        )

        orientation = np.asarray(
            state[5],
            dtype=float,
        )

        return position, orientation

    def get_end_effector_position(self) -> np.ndarray:
        """Return XYZ position of the Panda end-effector."""

        position, _ = self.get_end_effector_pose()

        return position

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_configuration(self, q) -> np.ndarray:
        """Validate and convert a joint configuration."""

        q = np.asarray(q, dtype=float)

        if q.shape != (self.DOF,):
            raise ValueError(
                f"Panda configuration must have shape "
                f"({self.DOF},), got {q.shape}."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "Panda configuration contains non-finite values."
            )

        return q