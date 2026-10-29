"""
Kinematic Analysis Module
-------------------------

Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "What is the robot doing kinematically along this trajectory?"

It takes an already-generated Trajectory (time, q, qd, qdd) and,
using the existing Planar3RRobot model, computes per-sample
kinematic measurements:

    - end-effector position
    - end-effector velocity
    - Jacobian-derived manipulability
    - singularity-related numerical metric
    - per-joint maximum velocity
    - per-joint maximum acceleration

This module intentionally does NOT:

    - decide whether any measurement is dangerous or a "failure"
    - classify singularity risk as LOW / MEDIUM / HIGH
    - compute torque, power, energy, or any dynamics quantity
    - check for obstacles or collisions
    - generate or modify trajectories
    - solve inverse kinematics
    - calculate failure probability
    - perform recovery
    - perform optimization
    - perform visualization

Those responsibilities belong to their own modules elsewhere
in the project.


ROBOT MODEL USAGE
-----------------

This module does not rebuild the robot.

It relies entirely on the already-tested Planar3RRobot class
from robot/robot_model.py:

    self.robot.end_effector_position(q)
    self.robot.jacobian(q)


DATA CONTRACT NOTE
------------------

models/data_models.py defines:

    KinematicResult.manipulability
    KinematicResult.singularity_metric

as single float values rather than arrays.

However, this module calculates manipulability and the
singularity-related metric at EVERY trajectory sample.

Therefore:

    - full per-sample values are calculated internally
    - the minimum value across the complete trajectory is stored
      in KinematicResult

The minimum is the worst-case value because:

    lower manipulability
        -> closer to singular behavior

    lower smallest singular value
        -> closer to singular behavior

If KinematicResult is extended in the future to store complete
per-sample arrays, this module can be updated accordingly.


ARCHITECTURAL BOUNDARY
----------------------

    robot_model.py
        -> "What robot do we have?"

    kinematics.py
        -> "What is the robot doing kinematically?"

    dynamics.py
        -> "What forces/torques/energy are involved?"

    analysis/
        -> "Is what happened dangerous?"

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

import numpy as np

from robot.robot_model import Planar3RRobot
from models.data_models import Trajectory, KinematicResult


class KinematicAnalyzer:
    """
    Computes kinematic measurements for a single robot trajectory.

    This class performs measurement only.

    It does NOT:

        - classify risk
        - predict failure
        - calculate torque
        - calculate energy
        - analyze obstacles
        - perform recovery
        - optimize trajectories

    Parameters
    ----------
    robot : Planar3RRobot
        Already-created robot model used for all kinematic
        calculations.
    """

    def __init__(self, robot: Planar3RRobot) -> None:
        """
        Initialize the kinematic analyzer.

        Parameters
        ----------
        robot : Planar3RRobot
            Existing robot model.

        Notes
        -----
        The robot is NOT rebuilt inside this class.
        """

        self.robot = robot

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def analyze(
        self,
        trajectory: Trajectory,
    ) -> KinematicResult:
        """
        Run the complete kinematic analysis for one trajectory.

        Parameters
        ----------
        trajectory : Trajectory
            Existing trajectory containing:

                time : (N,)
                q    : (N,3)
                qd   : (N,3)
                qdd  : (N,3)

            The trajectory is treated as read-only.

        Returns
        -------
        KinematicResult
            Complete kinematic measurements for the trajectory.

        Output fields
        -------------
        joint_positions:
            Shape (N,3), radians.

        joint_velocities:
            Shape (N,3), radians/second.

        joint_accelerations:
            Shape (N,3), radians/second².

        end_effector_positions:
            Shape (N,2), meters.

        end_effector_velocities:
            Shape (N,2), meters/second.

        max_joint_velocity:
            Shape (3,), radians/second.

        max_joint_acceleration:
            Shape (3,), radians/second².

        manipulability:
            Scalar worst-case/minimum manipulability.

        singularity_metric:
            Scalar worst-case/minimum smallest singular value.

        Raises
        ------
        ValueError
            If trajectory data has invalid dimensions,
            inconsistent sample counts, no samples, or
            non-finite numerical values.
        """

        # --------------------------------------------------------------
        # 1. Validate the trajectory before performing calculations.
        # --------------------------------------------------------------

        self._validate_trajectory(trajectory)

        # --------------------------------------------------------------
        # 2. Copy the input arrays.
        #
        # The original trajectory must never be modified.
        # --------------------------------------------------------------

        q = np.array(
            trajectory.q,
            dtype=float,
            copy=True,
        )

        qd = np.array(
            trajectory.qd,
            dtype=float,
            copy=True,
        )

        qdd = np.array(
            trajectory.qdd,
            dtype=float,
            copy=True,
        )

        # --------------------------------------------------------------
        # 3. Joint position / velocity / acceleration data.
        #
        # These are already provided by the trajectory.
        # We do not recalculate them unnecessarily.
        # --------------------------------------------------------------

        joint_positions = q.copy()

        joint_velocities = qd.copy()

        joint_accelerations = qdd.copy()

        # --------------------------------------------------------------
        # 4. End-effector positions.
        # --------------------------------------------------------------

        end_effector_positions = (
            self._compute_end_effector_positions(q)
        )

        # --------------------------------------------------------------
        # 5. End-effector velocities and planar Jacobians.
        #
        # The Jacobians are calculated once and reused for:
        #
        #   - end-effector velocity
        #   - manipulability
        #   - singularity metric
        # --------------------------------------------------------------

        (
            end_effector_velocities,
            planar_jacobians,
        ) = self._compute_end_effector_velocities(
            q,
            qd,
        )

        # --------------------------------------------------------------
        # 6. Manipulability at every trajectory sample.
        # --------------------------------------------------------------

        manipulability_series = (
            self._compute_manipulability(
                planar_jacobians
            )
        )

        # --------------------------------------------------------------
        # 7. Smallest singular value at every trajectory sample.
        # --------------------------------------------------------------

        singularity_metric_series = (
            self._compute_singularity_metric(
                planar_jacobians
            )
        )

        # --------------------------------------------------------------
        # 8. Maximum velocity and acceleration for EACH joint.
        # --------------------------------------------------------------

        (
            max_joint_velocity,
            max_joint_acceleration,
        ) = self._compute_max_joint_values(
            qd,
            qdd,
        )

        # --------------------------------------------------------------
        # 9. KinematicResult currently stores manipulability and
        #    singularity_metric as scalar values.
        #
        #    Therefore store the minimum/worst-case value across
        #    the complete trajectory.
        # --------------------------------------------------------------

        manipulability = float(
            np.min(manipulability_series)
        )

        singularity_metric = float(
            np.min(singularity_metric_series)
        )

        # --------------------------------------------------------------
        # 10. Construct the project's shared result object.
        # --------------------------------------------------------------

        return KinematicResult(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_accelerations=joint_accelerations,
            end_effector_positions=end_effector_positions,
            end_effector_velocities=end_effector_velocities,
            max_joint_velocity=max_joint_velocity,
            max_joint_acceleration=max_joint_acceleration,
            manipulability=manipulability,
            singularity_metric=singularity_metric,
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_trajectory(
        trajectory: Trajectory,
    ) -> None:
        """
        Validate the trajectory before kinematic calculations.

        Required shapes:

            time -> (N,)

            q    -> (N,3)
            qd   -> (N,3)
            qdd  -> (N,3)

        The method also verifies:

            - at least one trajectory sample exists
            - all arrays have the same number of samples
            - all numerical values are finite

        The method never:

            - resizes
            - truncates
            - interpolates
            - modifies

        the trajectory.
        """

        # --------------------------------------------------------------
        # Convert to NumPy arrays only for validation.
        # These are not written back to the trajectory.
        # --------------------------------------------------------------

        time = np.asarray(
            trajectory.time,
            dtype=float,
        )

        q = np.asarray(
            trajectory.q,
            dtype=float,
        )

        qd = np.asarray(
            trajectory.qd,
            dtype=float,
        )

        qdd = np.asarray(
            trajectory.qdd,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Validate time dimensionality.
        # --------------------------------------------------------------

        if time.ndim != 1:
            raise ValueError(
                "trajectory.time must be 1-D with shape (N,), "
                f"got shape {time.shape}."
            )

        # --------------------------------------------------------------
        # Reject empty trajectories.
        # --------------------------------------------------------------

        n_samples = time.shape[0]

        if n_samples == 0:
            raise ValueError(
                "Trajectory must contain at least one sample."
            )

        # --------------------------------------------------------------
        # Validate joint arrays.
        # --------------------------------------------------------------

        for name, array in (
            ("q", q),
            ("qd", qd),
            ("qdd", qdd),
        ):

            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(
                    f"trajectory.{name} must have shape (N, 3), "
                    f"got shape {array.shape}."
                )

            if array.shape[0] != n_samples:
                raise ValueError(
                    f"trajectory.{name} has "
                    f"{array.shape[0]} samples but "
                    f"trajectory.time has {n_samples} samples. "
                    "All trajectory arrays must have the same "
                    "sample count."
                )

        # --------------------------------------------------------------
        # Reject NaN and infinite values.
        #
        # Invalid numerical values could otherwise propagate through
        # Jacobian, SVD, and manipulability calculations.
        # --------------------------------------------------------------

        if not np.all(np.isfinite(time)):
            raise ValueError(
                "trajectory.time contains NaN or infinite values."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "trajectory.q contains NaN or infinite values."
            )

        if not np.all(np.isfinite(qd)):
            raise ValueError(
                "trajectory.qd contains NaN or infinite values."
            )

        if not np.all(np.isfinite(qdd)):
            raise ValueError(
                "trajectory.qdd contains NaN or infinite values."
            )

    # ==================================================================
    # END-EFFECTOR POSITION
    # ==================================================================

    def _compute_end_effector_positions(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """
        Compute end-effector [x, y] position for every trajectory sample.

        Parameters
        ----------
        q : np.ndarray
            Joint positions with shape (N,3).

        Returns
        -------
        np.ndarray
            End-effector positions with shape (N,2).

        Units
        -----
        Position: meters.
        """

        n_samples = q.shape[0]

        positions = np.zeros(
            (n_samples, 2),
            dtype=float,
        )

        for i in range(n_samples):

            positions[i] = (
                self.robot.end_effector_position(
                    q[i]
                )
            )

        return positions

    # ==================================================================
    # END-EFFECTOR VELOCITY + PLANAR JACOBIANS
    # ==================================================================

    def _compute_end_effector_velocities(
        self,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector translational velocity and planar Jacobian.

        Parameters
        ----------
        q : np.ndarray
            Joint positions, shape (N,3), radians.

        qd : np.ndarray
            Joint velocities, shape (N,3), radians/second.

        Returns
        -------
        tuple
            Two arrays:

            end_effector_velocities:
                Shape (N,2), containing [vx, vy] in m/s.

            planar_jacobians:
                Shape (N,3,3).

        Jacobian convention
        -------------------
        Planar3RRobot.jacobian() returns the full geometric
        Jacobian with shape (6,3):

            row 0 -> vx
            row 1 -> vy
            row 2 -> vz
            row 3 -> wx
            row 4 -> wy
            row 5 -> wz

        For the planar robot, we use:

            J_planar = J[[0,1,5], :]

        resulting in:

            [vx]
            [vy]
            [wz]

        The angular velocity wz is used internally but is not
        stored in KinematicResult.end_effector_velocities,
        which is defined as the translational [vx, vy] velocity.
        """

        n_samples = q.shape[0]

        end_effector_velocities = np.zeros(
            (n_samples, 2),
            dtype=float,
        )

        planar_jacobians = np.zeros(
            (n_samples, 3, 3),
            dtype=float,
        )

        for i in range(n_samples):

            # ----------------------------------------------------------
            # Get the full geometric Jacobian from the existing
            # robot model.
            # ----------------------------------------------------------

            full_jacobian = self.robot.jacobian(
                q[i]
            )

            # ----------------------------------------------------------
            # Extract the planar task-space components:
            #
            #   vx
            #   vy
            #   wz
            #
            # Result shape: (3,3)
            # ----------------------------------------------------------

            j_planar = full_jacobian[
                [0, 1, 5],
                :,
            ]

            planar_jacobians[i] = j_planar

            # ----------------------------------------------------------
            # Cartesian task-space velocity:
            #
            #     v = J * qd
            #
            # Gives:
            #
            #     [vx, vy, wz]
            # ----------------------------------------------------------

            task_velocity = (
                j_planar @ qd[i]
            )

            # Only translational velocity is stored.
            end_effector_velocities[i] = (
                task_velocity[:2]
            )

        return (
            end_effector_velocities,
            planar_jacobians,
        )

    # ==================================================================
    # MANIPULABILITY
    # ==================================================================

    @staticmethod
    def _compute_manipulability(
        planar_jacobians: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Yoshikawa-style manipulability at every sample.

        Formula:

            manipulability =
                sqrt(det(J_planar @ J_planar.T))

        Parameters
        ----------
        planar_jacobians : np.ndarray
            Shape (N,3,3).

        Returns
        -------
        np.ndarray
            Manipulability series with shape (N,).

        Interpretation
        --------------
        Higher manipulability generally indicates better local
        motion capability.

        Values approaching zero indicate proximity to a
        singular configuration.

        This method does NOT classify singularity risk.
        """

        n_samples = (
            planar_jacobians.shape[0]
        )

        manipulability = np.zeros(
            n_samples,
            dtype=float,
        )

        for i in range(n_samples):

            j_planar = planar_jacobians[i]

            # J * J^T
            jj_t = (
                j_planar
                @ j_planar.T
            )

            determinant = np.linalg.det(
                jj_t
            )

            # Numerical round-off can produce a tiny negative
            # determinant near a singular configuration.
            safe_determinant = max(
                float(determinant),
                0.0,
            )

            manipulability[i] = np.sqrt(
                safe_determinant
            )

        return manipulability

    # ==================================================================
    # SINGULARITY-RELATED METRIC
    # ==================================================================

    @staticmethod
    def _compute_singularity_metric(
        planar_jacobians: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the smallest singular value of the planar Jacobian.

        For each sample:

            singular_values = SVD(J_planar)

            singularity_metric =
                smallest singular value

        Parameters
        ----------
        planar_jacobians : np.ndarray
            Shape (N,3,3).

        Returns
        -------
        np.ndarray
            Smallest singular value for every sample,
            shape (N,).

        Interpretation
        --------------
        Larger value:
            Better local velocity capability.

        Value approaching zero:
            Increasing proximity to singular behavior.

        IMPORTANT
        ---------
        This method only calculates the numerical metric.

        It does NOT:

            - compare against configuration thresholds
            - classify risk
            - declare a failure
        """

        n_samples = (
            planar_jacobians.shape[0]
        )

        singularity_metric = np.zeros(
            n_samples,
            dtype=float,
        )

        for i in range(n_samples):

            singular_values = np.linalg.svd(
                planar_jacobians[i],
                compute_uv=False,
            )

            # SVD returns singular values in descending order,
            # so the final value is the smallest singular value.
            singularity_metric[i] = (
                singular_values[-1]
            )

        return singularity_metric

    # ==================================================================
    # PER-JOINT MAXIMUM VALUES
    # ==================================================================

    @staticmethod
    def _compute_max_joint_values(
        qd: np.ndarray,
        qdd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute maximum absolute velocity and acceleration for
        each joint separately.

        Parameters
        ----------
        qd : np.ndarray
            Joint velocities, shape (N,3), rad/s.

        qdd : np.ndarray
            Joint accelerations, shape (N,3), rad/s².

        Returns
        -------
        tuple
            max_joint_velocity:
                Shape (3,), rad/s.

            max_joint_acceleration:
                Shape (3,), rad/s².

        Important
        ---------
        This returns one maximum value PER JOINT.

        It does not return one global maximum for the entire robot.
        """

        max_joint_velocity = np.max(
            np.abs(qd),
            axis=0,
        )

        max_joint_acceleration = np.max(
            np.abs(qdd),
            axis=0,
        )

        return (
            max_joint_velocity,
            max_joint_acceleration,
        )


# ======================================================================
# DEMO / KINEMATICS SANITY CHECK
# ======================================================================

if __name__ == "__main__":

    """
    Minimal synthetic trajectory used only to verify the
    kinematics module.

    This demo intentionally does NOT perform:

        - path planning
        - inverse kinematics
        - dynamics
        - torque calculation
        - energy calculation
        - obstacle analysis
        - failure prediction
        - Explainable AI
        - recovery
        - optimization
        - visualization
    """

    # --------------------------------------------------------------
    # Create a simple trajectory.
    # --------------------------------------------------------------

    n_samples = 100

    duration = 4.0

    time = np.linspace(
        0.0,
        duration,
        n_samples,
    )

    # --------------------------------------------------------------
    # Small smooth target joint angles.
    # --------------------------------------------------------------

    q1_target = 0.4
    q2_target = 0.3
    q3_target = 0.2

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

    # --------------------------------------------------------------
    # Numerically estimate velocity and acceleration ONLY for
    # this synthetic demo trajectory.
    #
    # The actual project trajectory generator will provide qd/qdd.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Create the existing robot model.
    # --------------------------------------------------------------

    robot = Planar3RRobot()

    # --------------------------------------------------------------
    # Create the project's Trajectory dataclass.
    # --------------------------------------------------------------

    demo_trajectory = Trajectory(
        path_id="demo-001",
        name="Kinematics sanity-check trajectory",
        description=(
            "Synthetic trajectory used only to test "
            "kinematics.py"
        ),
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=(
            robot.end_effector_position(
                q[0]
            )
        ),
        goal_position=(
            robot.end_effector_position(
                q[-1]
            )
        ),
        duration=duration,
        path_length=0.0,
    )

    # --------------------------------------------------------------
    # Analyze the trajectory.
    # --------------------------------------------------------------

    analyzer = KinematicAnalyzer(
        robot
    )

    result = analyzer.analyze(
        demo_trajectory
    )

    # --------------------------------------------------------------
    # Display results.
    # --------------------------------------------------------------

    print(
        f"Number of samples: {n_samples}"
    )

    print(
        "Maximum velocity per joint "
        f"(rad/s): {result.max_joint_velocity}"
    )

    print(
        "Maximum acceleration per joint "
        f"(rad/s^2): "
        f"{result.max_joint_acceleration}"
    )

    print(
        "Final end-effector position "
        f"(m): "
        f"{result.end_effector_positions[-1]}"
    )

    print(
        "Worst-case (minimum) "
        f"manipulability: "
        f"{result.manipulability}"
    )

    print(
        "Worst-case (minimum) "
        f"singularity metric: "
        f"{result.singularity_metric}"
    )

    # --------------------------------------------------------------
    # Independent sanity check at q=[0,0,0].
    #
    # The robot should be fully extended along +X.
    # --------------------------------------------------------------

    q_home = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    ee_home = (
        robot.end_effector_position(
            q_home
        )
    )

    print(
        "\nSanity check -- "
        "end-effector position at "
        "q=[0,0,0]:"
    )

    print(ee_home)

    print(
        "Expected approximately: "
        "[2.4, 0.0]"
    )

    # --------------------------------------------------------------
    # At q=[0,0,0], the planar Jacobian is singular.
    # Therefore manipulability and the smallest singular value
    # should be very close to zero.
    #
    # This is expected behavior, NOT an error.
    # --------------------------------------------------------------

    home_jacobian = robot.jacobian(
        q_home
    )

    planar_home_jacobian = (
        home_jacobian[
            [0, 1, 5],
            :
        ]
    )

    home_manipulability = np.sqrt(
        max(
            float(
                np.linalg.det(
                    planar_home_jacobian
                    @ planar_home_jacobian.T
                )
            ),
            0.0,
        )
    )

    home_singular_values = (
        np.linalg.svd(
            planar_home_jacobian,
            compute_uv=False,
        )
    )

    home_singularity_metric = (
        home_singular_values[-1]
    )

    print(
        "\nHome configuration "
        "manipulability:"
    )

    print(home_manipulability)

    print(
        "Home configuration "
        "smallest singular value:"
    )

    print(home_singularity_metric)

    print(
        "\nThe values above should be "
        "very close to zero because "
        "q=[0,0,0] is a singular "
        "configuration for this planar "
        "3R robot."
    )