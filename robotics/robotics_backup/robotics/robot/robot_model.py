"""
Robot-Model Foundation
----------------------

Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "What robot are we simulating?"

It builds the virtual Planar3R robot, a 3-DOF planar revolute
manipulator, using Robotics Toolbox for Python (Peter Corke's
roboticstoolbox-python).

The module exposes basic robot-level capabilities that later modules
can build upon:

    - forward kinematics
    - end-effector position / pose
    - Jacobian
    - joint limit checks
    - joint velocity limit checks
    - basic reachability check
    - robot configuration access
    - dashboard/debug summary

This module intentionally does NOT implement:

    - path planning
    - trajectory generation
    - inverse kinematics
    - dynamics calculations
    - failure prediction
    - risk scoring
    - explainable AI
    - recoverability
    - optimization
    - visualization
    - Streamlit/UI code

All numerical robot parameters are imported from config.py.
config.py is the single source of truth for robot parameters.

Units:
    Joint angles        -> radians
    Joint velocities    -> radians/second
    Joint acceleration  -> radians/second^2
    Distance            -> meters
    Torque              -> N*m
    Energy              -> Joules
    Power               -> Watts
    Time                -> seconds


DH ASSUMPTIONS
--------------

The Planar3R robot is represented using standard Denavit-Hartenberg
parameters.

For all three revolute joints:

    alpha = 0
    d     = 0

and:

    a1 = L1
    a2 = L2
    a3 = L3

Therefore the robot remains in the X-Y plane.

The forward kinematics naturally reduce to:

    x = L1*cos(q1)
        + L2*cos(q1 + q2)
        + L3*cos(q1 + q2 + q3)

    y = L1*sin(q1)
        + L2*sin(q1 + q2)
        + L3*sin(q1 + q2 + q3)

    theta = q1 + q2 + q3


GRAVITY CONVENTION
------------------

The robot's kinematic plane is the X-Y plane.

For this project, the X-Y plane is treated as a vertical plane in
the physical world so that gravity produces meaningful joint
torques for the future dynamics module.

Therefore gravity acts along:

    [0, -GRAVITY, 0]

This is a modeling assumption for the simulation.

No dynamics calculations are performed in this file.


LINK INERTIA ASSUMPTION
-----------------------

Each link is approximated as a uniform slender rod whose length
lies along its local X-axis.

For this simplified model:

    Ixx ~= 0
    Iyy = Izz = m*L^2/12

The scalar values stored in config.py as LINK_INERTIAS represent:

    m*L^2/12

The Robotics Toolbox 6-element inertia representation is:

    [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]

Therefore each link is represented as:

    [0, I, I, 0, 0, 0]
"""

from __future__ import annotations

import numpy as np

from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3

from config import (
    ROBOT_NAME,
    ROBOT_DOF,
    LINK_LENGTHS,
    LINK_COM_RATIO,
    JOINT_LIMITS_RAD,
    MAX_JOINT_VELOCITIES,
    MAX_JOINT_TORQUES,
    LINK_MASSES,
    LINK_INERTIAS,
    GRAVITY,
)

from models.data_models import RobotConfig


class Planar3RRobot:
    """
    Virtual model of the project's 3-DOF planar revolute robot.

    The class wraps a Robotics Toolbox DHRobot constructed from
    the project's configuration.

    Attributes
    ----------
    robot_config : RobotConfig
        Shared robot configuration populated from config.py.

    toolbox_robot : DHRobot
        Underlying Robotics Toolbox robot model.
    """

    def __init__(self) -> None:
        """
        Construct the Planar3R robot.

        The robot configuration is loaded from config.py and then
        used to construct the Robotics Toolbox robot model.
        """

        self.robot_config: RobotConfig = self._build_robot_config()

        self.toolbox_robot: DHRobot = self._build_toolbox_robot()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_robot_config(self) -> RobotConfig:
        """
        Build the shared RobotConfig dataclass from config.py.

        Returns
        -------
        RobotConfig
            Robot configuration containing the project's robot
            parameters.
        """

        return RobotConfig(
            name=ROBOT_NAME,
            dof=ROBOT_DOF,
            link_lengths=list(LINK_LENGTHS),
            joint_limits_rad=list(JOINT_LIMITS_RAD),
            max_joint_velocities=np.asarray(
                MAX_JOINT_VELOCITIES,
                dtype=float,
            ),
            max_joint_torques=np.asarray(
                MAX_JOINT_TORQUES,
                dtype=float,
            ),
            link_masses=list(LINK_MASSES),
            link_inertias=list(LINK_INERTIAS),
            gravity=GRAVITY,
        )

    def _build_toolbox_robot(self) -> DHRobot:
        """
        Build the Robotics Toolbox DHRobot.

        Each link is represented as a RevoluteDH joint with:

            alpha = 0
            d = 0
            a = configured link length

        Joint limits, masses, center-of-mass locations, and inertias
        are taken directly from config.py.

        Returns
        -------
        DHRobot
            Configured Robotics Toolbox robot model.
        """

        links = []

        for i in range(ROBOT_DOF):

            link_length = float(LINK_LENGTHS[i])

            q_min, q_max = JOINT_LIMITS_RAD[i]

            link_mass = float(LINK_MASSES[i])

            link_inertia = float(LINK_INERTIAS[i])

            # Each link is approximated as a uniform slender rod
            # along its local X-axis:
            #
            #     Ixx ~= 0
            #     Iyy = Izz = mL^2/12
            #
            # Robotics Toolbox expects:
            #
            #     [Ixx, Iyy, Izz, Ixy, Iyz, Ixz]

            inertia_vector = [
                0.0,
                link_inertia,
                link_inertia,
                0.0,
                0.0,
                0.0,
            ]

            center_of_mass = [
                link_length * LINK_COM_RATIO,
                0.0,
                0.0,
            ]

            link = RevoluteDH(
                a=link_length,
                alpha=0.0,
                d=0.0,
                qlim=[q_min, q_max],
                m=link_mass,
                r=center_of_mass,
                I=inertia_vector,
            )

            links.append(link)

        toolbox_robot = DHRobot(
            links,
            name=ROBOT_NAME,
        )

        # The robot is modeled in the X-Y plane, which is treated
        # as a vertical plane for the dynamics simulation.
        #
        # Therefore gravity acts along the negative Y direction.

        toolbox_robot.gravity = np.array(
            [0.0, -GRAVITY, 0.0],
            dtype=float,
        )

        return toolbox_robot

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_joint_vector(
        q: np.ndarray,
        label: str = "q",
    ) -> np.ndarray:
        """
        Validate and convert a joint vector.

        Parameters
        ----------
        q : array-like
            Joint values.

        label : str
            Name used in the error message.

        Returns
        -------
        np.ndarray
            Validated floating-point array of shape (ROBOT_DOF,).

        Raises
        ------
        ValueError
            If the input does not contain exactly ROBOT_DOF values.
        """

        q_arr = np.asarray(q, dtype=float)

        if q_arr.shape != (ROBOT_DOF,):
            raise ValueError(
                f"Expected {label} to contain "
                f"{ROBOT_DOF} joint values, "
                f"received shape {q_arr.shape}."
            )

        return q_arr

    @staticmethod
    def _validate_position(
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and convert a 2D Cartesian position.

        Parameters
        ----------
        position : array-like
            Cartesian position [x, y].

        Returns
        -------
        np.ndarray
            Validated array of shape (2,).

        Raises
        ------
        ValueError
            If the input does not contain exactly two values.
        """

        pos_arr = np.asarray(position, dtype=float)

        if pos_arr.shape != (2,):
            raise ValueError(
                "Expected position to contain 2 values [x, y], "
                f"received shape {pos_arr.shape}."
            )

        return pos_arr

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def get_robot_config(self) -> RobotConfig:
        """
        Return the shared RobotConfig object.
        """

        return self.robot_config

    def get_toolbox_robot(self) -> DHRobot:
        """
        Return the underlying Robotics Toolbox robot.
        """

        return self.toolbox_robot

    def get_joint_limits(self):
        """
        Return configured joint limits in radians.

        The returned structure contains one (minimum, maximum)
        pair for each joint.
        """

        return JOINT_LIMITS_RAD

    def get_link_lengths(self) -> np.ndarray:
        """
        Return configured link lengths in meters.
        """

        return np.asarray(
            LINK_LENGTHS,
            dtype=float,
        )

    def get_max_reach(self) -> float:
        """
        Return the maximum radial reach of the robot.

        For this planar serial arm:

            maximum reach = L1 + L2 + L3

        The value is calculated from the configured link lengths.
        """

        return float(
            np.sum(
                np.asarray(
                    LINK_LENGTHS,
                    dtype=float,
                )
            )
        )

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        q: np.ndarray,
    ) -> SE3:
        """
        Compute the end-effector pose.

        Parameters
        ----------
        q : array-like
            Joint angles [q1, q2, q3] in radians.
            Shape must be (3,).

        Returns
        -------
        SE3
            Homogeneous transformation representing the
            end-effector pose.
        """

        q_arr = self._validate_joint_vector(q)

        return self.toolbox_robot.fkine(q_arr)

    def end_effector_position(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """
        Return the end-effector Cartesian position.

        Parameters
        ----------
        q : array-like
            Joint angles in radians.

        Returns
        -------
        np.ndarray
            [x, y] position in meters.
            Shape is (2,).
        """

        transform = self.forward_kinematics(q)

        return np.asarray(
            transform.t[:2],
            dtype=float,
        )

    def end_effector_pose(
        self,
        q: np.ndarray,
    ) -> dict:
        """
        Return the end-effector position, orientation, and transform.

        Parameters
        ----------
        q : array-like
            Joint angles in radians.

        Returns
        -------
        dict
            Dictionary containing:

                position:
                    np.ndarray of shape (2,), [x, y]

                orientation:
                    planar orientation about the Z-axis, in radians

                transform:
                    Robotics Toolbox SE3 transformation
        """

        q_arr = self._validate_joint_vector(q)

        transform = self.forward_kinematics(q_arr)

        position = np.asarray(
            transform.t[:2],
            dtype=float,
        )

        # For a planar 3R robot, the end-effector orientation
        # about the Z-axis is q1 + q2 + q3.
        #
        # We obtain it from the rotation matrix rather than
        # assuming a particular RPY ordering.

        orientation = float(
            np.arctan2(
                transform.R[1, 0],
                transform.R[0, 0],
            )
        )

        return {
            "position": position,
            "orientation": orientation,
            "transform": transform,
        }

    def jacobian(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """
        Return the base-frame geometric Jacobian.

        Parameters
        ----------
        q : array-like
            Joint angles in radians.

        Returns
        -------
        np.ndarray
            Geometric Jacobian of shape (6, 3).

            Rows correspond to:

                [vx, vy, vz, wx, wy, wz]

            Columns correspond to:

                [joint1, joint2, joint3]

        Notes
        -----
        Later modules such as singularity analysis are responsible
        for selecting and interpreting the planar-relevant
        components.
        """

        q_arr = self._validate_joint_vector(q)

        return np.asarray(
            self.toolbox_robot.jacob0(q_arr),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Joint limit checks
    # ------------------------------------------------------------------

    def check_joint_limits(
        self,
        q: np.ndarray,
    ) -> bool:
        """
        Check whether every joint is within its configured limits.

        Boundary values are considered valid.

        Parameters
        ----------
        q : array-like
            Joint angles in radians.

        Returns
        -------
        bool
            True if all joints are within their limits.
            False otherwise.
        """

        q_arr = self._validate_joint_vector(q)

        for value, (q_min, q_max) in zip(
            q_arr,
            JOINT_LIMITS_RAD,
        ):
            if value < q_min or value > q_max:
                return False

        return True

    def check_joint_velocity_limits(
        self,
        qd: np.ndarray,
    ) -> bool:
        """
        Check whether every joint velocity is within its limit.

        Absolute velocity is compared against the configured
        maximum velocity for each joint.

        Parameters
        ----------
        qd : array-like
            Joint velocities in radians/second.

        Returns
        -------
        bool
            True if all joint velocities are within their limits.
            False otherwise.
        """

        qd_arr = self._validate_joint_vector(
            qd,
            label="qd",
        )

        limits = np.asarray(
            MAX_JOINT_VELOCITIES,
            dtype=float,
        )

        return bool(
            np.all(
                np.abs(qd_arr) <= limits
            )
        )

    # ------------------------------------------------------------------
    # Basic workspace check
    # ------------------------------------------------------------------

    def is_position_reachable(
        self,
        position: np.ndarray,
    ) -> bool:
        """
        Perform a basic Cartesian workspace reachability check.

        The method checks whether the requested point lies within
        the maximum radial reach:

            distance_from_origin <= L1 + L2 + L3

        IMPORTANT
        ---------
        This is only a necessary workspace check.

        It does NOT guarantee that:

            - a valid inverse-kinematics solution exists
            - the joint limits can be satisfied
            - the desired orientation can be achieved

        Full inverse kinematics belongs to:

            planning/inverse_kinematics.py

        Parameters
        ----------
        position : array-like
            Cartesian position [x, y] in meters.

        Returns
        -------
        bool
            True if the point lies within the basic workspace
            boundary, otherwise False.
        """

        pos_arr = self._validate_position(position)

        distance_from_origin = float(
            np.linalg.norm(pos_arr)
        )

        return (
            distance_from_origin
            <= self.get_max_reach()
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a simple summary of the robot configuration.

        This is useful for debugging and later dashboard display.
        """

        return {
            "robot_name": self.robot_config.name,
            "dof": self.robot_config.dof,
            "link_lengths": self.get_link_lengths().tolist(),
            "joint_limits_rad": [
                list(limit)
                for limit in self.get_joint_limits()
            ],
            "max_joint_velocities": np.asarray(
                MAX_JOINT_VELOCITIES,
                dtype=float,
            ).tolist(),
            "max_joint_torques": np.asarray(
                MAX_JOINT_TORQUES,
                dtype=float,
            ).tolist(),
            "max_reach_m": self.get_max_reach(),
        }


# ----------------------------------------------------------------------
# Minimal standalone sanity test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal robot-model sanity test.

    This section intentionally does NOT perform:

        - path planning
        - trajectory generation
        - dynamics
        - risk prediction
        - recovery
        - optimization
        - visualization

    It only verifies the basic robot model.
    """

    robot = Planar3RRobot()

    print("Robot summary:")

    for key, value in robot.summary().items():
        print(f"  {key}: {value}")

    # Fully extended configuration.
    q_home = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    # --------------------------------------------------------------
    # Forward kinematics sanity check
    # --------------------------------------------------------------

    ee_position = robot.end_effector_position(q_home)

    print(
        "\nEnd-effector position at "
        "q=[0, 0, 0]:"
    )

    print(ee_position)

    print(
        "Expected approximately: "
        "[2.4, 0.0]"
    )

    # --------------------------------------------------------------
    # End-effector pose
    # --------------------------------------------------------------

    pose = robot.end_effector_pose(q_home)

    print(
        "\nEnd-effector orientation "
        "at q=[0, 0, 0]:"
    )

    print(
        f"{pose['orientation']:.6f} rad"
    )

    # --------------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------------

    J = robot.jacobian(q_home)

    print(
        "\nJacobian at q=[0, 0, 0]:"
    )

    print(J)

    # --------------------------------------------------------------
    # Joint-limit check
    # --------------------------------------------------------------

    limits_ok = robot.check_joint_limits(
        q_home
    )

    print(
        "\nJoint limits satisfied for "
        "q=[0, 0, 0]:"
    )

    print(limits_ok)

    # --------------------------------------------------------------
    # Basic reachability check
    # --------------------------------------------------------------

    test_position = np.array(
        [2.4, 0.0],
        dtype=float,
    )

    reachable = robot.is_position_reachable(
        test_position
    )

    print(
        "\nIs [2.4, 0.0] within the basic "
        "workspace check?"
    )

    print(reachable)