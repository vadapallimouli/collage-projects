"""
analysis/kinematic_analysis.py
================================

Kinematic Safety Analysis Module
------------------------------------
Project: Intelligent Robot Motion Planning with Failure Prediction,
         Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "Did the kinematic measurements of this trajectory violate or
     approach the robot's configured kinematic limits?"

It takes the already-computed `KinematicResult` (produced by
`robot/kinematics.py`) and compares the relevant measurements against
the robot's configured limits from `config.py`
(`MAX_JOINT_VELOCITIES`, `JOINT_LIMITS_RAD`). It produces structured,
numerical/Boolean kinematic-safety evidence -- not a final failure
verdict.

This module intentionally does NOT:
    - recompute any kinematic quantity (forward kinematics, Jacobian,
      manipulability, singular values, velocities, accelerations)
    - calculate failure probability, risk score, risk percentage,
      overall risk, or a SAFE/UNSAFE classification
    - classify singularity severity (LOW/MEDIUM/HIGH)
    - analyze torque, power, or energy
    - analyze obstacles or clearance
    - generate natural-language explanations
    - suggest or apply recovery actions
    - compare or select between multiple trajectories

Those responsibilities belong to their own modules elsewhere in the
project (see the architecture note at the bottom of this docstring).

ROBOT MODEL / CONFIG USAGE
------------------------------
This module does not need a `Planar3RRobot` instance -- all of the
kinematic measurements it needs are already present on the input
`KinematicResult`. The only external source of truth it relies on is
`config.py`, specifically:

    MAX_JOINT_VELOCITIES   (rad/s, shape (3,))
    JOINT_LIMITS_RAD        (radians, shape (3,2): [(lower, upper), ...])

Neither is duplicated or re-hard-coded here.

DATA CONTRACT
-----------------
`models/data_models.py` defines `KinematicAnalysisResult` with exactly
these fields, which this module populates as-is (no renamed, added, or
omitted fields):

    velocity_limit_violated: bool
    velocity_violation_count: np.ndarray        # (3,) int
    max_velocity_utilization: np.ndarray        # (3,) float
    max_velocity_excess: np.ndarray             # (3,) float, rad/s
    worst_velocity_joint: int                   # 1-based
    worst_velocity_sample: int                  # 0-based
    joint_limit_violated: np.ndarray            # (3,) bool
    joint_limit_violation_count: np.ndarray      # (3,) int
    min_joint_limit_margin: np.ndarray           # (3,) float, radians
    minimum_manipulability: float
    minimum_singularity_metric: float
    max_joint_acceleration: np.ndarray           # (3,) float, rad/s^2

ARCHITECTURAL BOUNDARY
--------------------------
    robot_model.py            -> "What robot do we have?"
    kinematics.py              -> "What is the robot doing kinematically?"
    dynamics.py                 -> "What forces/torques/energy are involved?"
    kinematic_analysis.py      -> "Did kinematics violate configured limits?"  (THIS FILE)
    dynamic_analysis.py         -> "Did dynamics violate configured limits?"
    obstacle_analysis.py       -> "Is the path too close to obstacles?"
    singularity_analysis.py     -> "How severe is the singularity risk?"
    prediction/                -> "What is the estimated failure risk?"
    explainability/             -> "Why is it risky?"
    recovery/                   -> "Can we fix it?"
    optimization/                -> "Which final path is best?"
"""

from __future__ import annotations

import numpy as np

from config import JOINT_LIMITS_RAD, MAX_JOINT_VELOCITIES
from models.data_models import KinematicAnalysisResult, KinematicResult


class KinematicSafetyAnalyzer:
    """
    Interprets an existing `KinematicResult` against the robot's
    configured kinematic limits (`config.py`) and produces structured
    kinematic-safety evidence.

    This class performs interpretation only. It does not recompute any
    kinematic quantity, and it does not decide whether the trajectory
    will ultimately "fail" -- that responsibility belongs to
    `prediction/failure_predictor.py`, which combines this evidence
    with evidence from other analysis modules.
    """

    def __init__(self) -> None:
        self._velocity_limits = self._validate_velocity_limits(
            MAX_JOINT_VELOCITIES
        )
        self._joint_limits = self._validate_joint_limits(JOINT_LIMITS_RAD)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, kinematic_result: KinematicResult) -> KinematicAnalysisResult:
        """
        Run the complete kinematic safety analysis for one
        `KinematicResult` and return the project's existing
        `KinematicAnalysisResult` dataclass, unmodified in shape/field
        names.

        Args:
            kinematic_result: A `KinematicResult` instance with
                `joint_positions`, `joint_velocities`, and
                `joint_accelerations` each shape (N, 3). Treated as
                read-only; never modified.

        Returns:
            A `KinematicAnalysisResult` populated with velocity-limit
            evidence, joint-limit evidence, and pass-through
            manipulability/singularity/acceleration measurements.

        Raises:
            ValueError: if the input arrays are inconsistent, wrongly
                shaped, or contain non-finite values (see
                `_validate_input`).
        """
        q, qd, qdd = self._validate_input(kinematic_result)

        (
            velocity_limit_violated,
            velocity_violation_count,
            max_velocity_utilization,
            max_velocity_excess,
            worst_velocity_joint,
            worst_velocity_sample,
        ) = self._analyze_velocity_limits(qd)

        (
            joint_limit_violated,
            joint_limit_violation_count,
            min_joint_limit_margin,
        ) = self._analyze_joint_limits(q)

        minimum_manipulability = float(kinematic_result.manipulability)
        minimum_singularity_metric = float(kinematic_result.singularity_metric)
        max_joint_acceleration = np.array(
            kinematic_result.max_joint_acceleration, dtype=float, copy=True
        )

        return KinematicAnalysisResult(
            velocity_limit_violated=velocity_limit_violated,
            velocity_violation_count=velocity_violation_count,
            max_velocity_utilization=max_velocity_utilization,
            max_velocity_excess=max_velocity_excess,
            worst_velocity_joint=worst_velocity_joint,
            worst_velocity_sample=worst_velocity_sample,
            joint_limit_violated=joint_limit_violated,
            joint_limit_violation_count=joint_limit_violation_count,
            min_joint_limit_margin=min_joint_limit_margin,
            minimum_manipulability=minimum_manipulability,
            minimum_singularity_metric=minimum_singularity_metric,
            max_joint_acceleration=max_joint_acceleration,
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_velocity_limits(max_joint_velocities) -> np.ndarray:
        """
        Validate that `MAX_JOINT_VELOCITIES` (from config.py) contains
        exactly 3 finite, strictly positive values. Raises ValueError
        otherwise.
        """
        limits = np.asarray(max_joint_velocities, dtype=float)
        if limits.shape != (3,):
            raise ValueError(
                f"config.MAX_JOINT_VELOCITIES must have shape (3,), got "
                f"shape {limits.shape}."
            )
        if not np.all(np.isfinite(limits)):
            raise ValueError(
                "config.MAX_JOINT_VELOCITIES contains NaN or infinite "
                "values, which are not allowed."
            )
        if np.any(limits <= 0.0):
            raise ValueError(
                f"config.MAX_JOINT_VELOCITIES must be strictly positive, "
                f"got {limits}."
            )
        return limits

    @staticmethod
    def _validate_joint_limits(joint_limits_rad) -> np.ndarray:
        """
        Validate that `JOINT_LIMITS_RAD` (from config.py) contains
        exactly 3 (lower, upper) pairs with lower < upper. Raises
        ValueError otherwise.

        Returns:
            np.ndarray, shape (3, 2): [[lower_1, upper_1], ...].
        """
        limits = np.asarray(joint_limits_rad, dtype=float)
        if limits.shape != (3, 2):
            raise ValueError(
                f"config.JOINT_LIMITS_RAD must have shape (3, 2), got "
                f"shape {limits.shape}."
            )
        if not np.all(np.isfinite(limits)):
            raise ValueError(
                "config.JOINT_LIMITS_RAD contains NaN or infinite values, "
                "which are not allowed."
            )
        lower = limits[:, 0]
        upper = limits[:, 1]
        if np.any(lower >= upper):
            raise ValueError(
                f"Each config.JOINT_LIMITS_RAD pair must satisfy "
                f"lower < upper, got lower={lower}, upper={upper}."
            )
        return limits

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(
        kinematic_result: KinematicResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate `kinematic_result.joint_positions`,
        `joint_velocities`, and `joint_accelerations`: each must be
        shape (N, 3) with N > 0 and all-finite values. Returns
        read-only copies so the input is never mutated.

        Raises:
            ValueError: on any shape mismatch, empty trajectory, or
                non-finite value.
        """
        q = np.asarray(kinematic_result.joint_positions, dtype=float)
        qd = np.asarray(kinematic_result.joint_velocities, dtype=float)
        qdd = np.asarray(kinematic_result.joint_accelerations, dtype=float)

        for name, array in (
            ("joint_positions", q),
            ("joint_velocities", qd),
            ("joint_accelerations", qdd),
        ):
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(
                    f"kinematic_result.{name} must have shape (N, 3), got "
                    f"shape {array.shape}."
                )
            if array.shape[0] == 0:
                raise ValueError(
                    f"kinematic_result.{name} has 0 samples; an empty "
                    f"trajectory cannot be analyzed."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"kinematic_result.{name} contains NaN or infinite "
                    f"values, which are not allowed."
                )

        n_samples = q.shape[0]
        if qd.shape[0] != n_samples or qdd.shape[0] != n_samples:
            raise ValueError(
                "kinematic_result.joint_positions, joint_velocities, and "
                "joint_accelerations must all have the same number of "
                f"samples; got {q.shape[0]}, {qd.shape[0]}, {qdd.shape[0]}."
            )

        # Return read-only copies -- never mutate the caller's arrays.
        return (
            np.array(q, copy=True),
            np.array(qd, copy=True),
            np.array(qdd, copy=True),
        )

    # ------------------------------------------------------------------
    # Velocity-limit analysis
    # ------------------------------------------------------------------

    def _analyze_velocity_limits(
        self, qd: np.ndarray
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Compare per-sample joint velocities against
        `config.MAX_JOINT_VELOCITIES` and derive velocity-limit
        evidence.

        Args:
            qd: shape (N, 3), radians/second.

        Returns:
            Tuple of:
                velocity_limit_violated: bool, True if ANY joint
                    exceeds its limit at ANY sample.
                velocity_violation_count: np.ndarray, shape (3,), int,
                    per-joint count of violating samples.
                max_velocity_utilization: np.ndarray, shape (3,),
                    float, per-joint max(|qd| / limit).
                max_velocity_excess: np.ndarray, shape (3,), float,
                    rad/s, per-joint max(0, max(|qd| - limit)).
                worst_velocity_joint: int, 1-based joint index with the
                    single highest utilization across all
                    samples/joints.
                worst_velocity_sample: int, 0-based sample index of
                    that worst utilization.
        """
        abs_qd = np.abs(qd)  # (N, 3)

        # Broadcasting (N,3) / (3,) -> (N,3)
        velocity_utilization = abs_qd / self._velocity_limits
        velocity_excess = abs_qd - self._velocity_limits

        # Strict greater-than: exactly at the limit is NOT a violation.
        velocity_violation_mask = abs_qd > self._velocity_limits  # (N,3) bool

        velocity_violation_count = np.sum(
            velocity_violation_mask, axis=0
        ).astype(int)  # (3,)
        velocity_limit_violated = bool(np.any(velocity_violation_mask))

        max_velocity_utilization = np.max(velocity_utilization, axis=0)  # (3,)
        max_velocity_excess = np.maximum(
            np.max(velocity_excess, axis=0), 0.0
        )  # (3,)

        worst_flat_index = int(np.argmax(velocity_utilization))
        worst_sample, worst_joint_zero_based = np.unravel_index(
            worst_flat_index, velocity_utilization.shape
        )
        worst_velocity_joint = int(worst_joint_zero_based) + 1  # 1-based
        worst_velocity_sample = int(worst_sample)  # 0-based

        return (
            velocity_limit_violated,
            velocity_violation_count,
            max_velocity_utilization,
            max_velocity_excess,
            worst_velocity_joint,
            worst_velocity_sample,
        )

    # ------------------------------------------------------------------
    # Joint position-limit analysis
    # ------------------------------------------------------------------

    def _analyze_joint_limits(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compare per-sample joint positions against
        `config.JOINT_LIMITS_RAD` and derive joint-position-limit
        evidence.

        Args:
            q: shape (N, 3), radians.

        Returns:
            Tuple of:
                joint_limit_violated: np.ndarray, shape (3,), bool.
                    True for a joint if at least one sample lies
                    outside its configured [lower, upper] range.
                    Exactly at a boundary is allowed (not a violation).
                joint_limit_violation_count: np.ndarray, shape (3,),
                    int, per-joint count of violating samples.
                min_joint_limit_margin: np.ndarray, shape (3,), float,
                    radians. Minimum signed distance to the nearer
                    boundary across the trajectory. May be negative if
                    the joint went outside its configured range (a
                    negative margin indicates how far outside the
                    boundary the trajectory went); it is intentionally
                    not clamped to zero.
        """
        lower = self._joint_limits[:, 0]  # (3,)
        upper = self._joint_limits[:, 1]  # (3,)

        joint_limit_violation_mask = (q < lower) | (q > upper)  # (N,3) bool

        joint_limit_violation_count = np.sum(
            joint_limit_violation_mask, axis=0
        ).astype(int)  # (3,)
        joint_limit_violated = np.any(joint_limit_violation_mask, axis=0)  # (3,)

        margin_to_lower = q - lower  # (N,3)
        margin_to_upper = upper - q  # (N,3)
        nearest_margin = np.minimum(margin_to_lower, margin_to_upper)  # (N,3)

        min_joint_limit_margin = np.min(nearest_margin, axis=0)  # (3,)

        return (
            joint_limit_violated,
            joint_limit_violation_count,
            min_joint_limit_margin,
        )


# =============================================================================
# Demo / kinematic-safety-analysis sanity check
# =============================================================================

if __name__ == "__main__":
    from robot.robot_model import Planar3RRobot
    from robot.kinematics import KinematicAnalyzer
    from models.data_models import Trajectory

    # Kinematic analysis demo -- synthetic trajectory used ONLY to
    # sanity-check this module's calculations. No risk, torque,
    # obstacle, or recovery logic is exercised here. This is not a
    # real robot failure; some samples are intentionally pushed past
    # the configured velocity limit to exercise the violation logic.
    n_samples = 100
    duration = 4.0
    time = np.linspace(0.0, duration, n_samples)

    # Smooth joint motion from 0 to target angles. Joint 2's target is
    # intentionally large enough to push its velocity above the
    # configured limit (2.0 rad/s) for part of the trajectory, purely
    # to exercise the violation-detection logic in this demo.
    q1_target, q2_target, q3_target = 0.4, 3.5, 0.2
    fraction = time / duration  # 0 -> 1

    q = np.stack(
        [
            q1_target * fraction,
            q2_target * fraction,
            q3_target * fraction,
        ],
        axis=1,
    )

    dt = time[1] - time[0]
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)

    robot = Planar3RRobot()

    demo_trajectory = Trajectory(
        path_id="demo-001",
        name="Kinematic safety analysis demo",
        description="Synthetic trajectory used only to test kinematic_analysis.py",
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=robot.end_effector_position(q[0]),
        goal_position=robot.end_effector_position(q[-1]),
        duration=duration,
        path_length=0.0,
    )

    kinematic_analyzer = KinematicAnalyzer(robot)
    kinematic_result = kinematic_analyzer.analyze(demo_trajectory)

    safety_analyzer = KinematicSafetyAnalyzer()
    analysis_result = safety_analyzer.analyze(kinematic_result)

    print(f"Number of samples: {n_samples}")
    print(f"\nVelocity limit violated: {analysis_result.velocity_limit_violated}")
    print(f"Velocity violation count (per joint): "
          f"{analysis_result.velocity_violation_count}")
    print(f"Maximum velocity utilization (per joint): "
          f"{analysis_result.max_velocity_utilization}")
    print(f"Maximum velocity excess (rad/s, per joint): "
          f"{analysis_result.max_velocity_excess}")
    print(f"Worst velocity joint (1-based): "
          f"Joint {analysis_result.worst_velocity_joint}")
    print(f"Worst velocity sample (0-based): "
          f"{analysis_result.worst_velocity_sample}")

    print(f"\nJoint limit violated (per joint): "
          f"{analysis_result.joint_limit_violated}")
    print(f"Joint limit violation count (per joint): "
          f"{analysis_result.joint_limit_violation_count}")
    print(f"Minimum joint limit margin (rad, per joint): "
          f"{analysis_result.min_joint_limit_margin}")

    print(f"\nMinimum manipulability: {analysis_result.minimum_manipulability}")
    print(f"Minimum singularity metric: "
          f"{analysis_result.minimum_singularity_metric}")
    print(f"Maximum acceleration per joint (rad/s^2): "
          f"{analysis_result.max_joint_acceleration}")

    # Sanity checks (assertions only, no interpretation/classification).
    assert analysis_result.velocity_violation_count.shape == (3,)
    assert analysis_result.max_velocity_utilization.shape == (3,)
    assert analysis_result.max_velocity_excess.shape == (3,)
    assert analysis_result.joint_limit_violated.shape == (3,)
    assert analysis_result.joint_limit_violation_count.shape == (3,)
    assert analysis_result.min_joint_limit_margin.shape == (3,)
    assert analysis_result.max_joint_acceleration.shape == (3,)

    assert np.isfinite(analysis_result.max_velocity_utilization).all()
    assert np.isfinite(analysis_result.max_velocity_excess).all()
    assert np.isfinite(analysis_result.min_joint_limit_margin).all()
    assert np.isfinite(analysis_result.max_joint_acceleration).all()

    assert 1 <= analysis_result.worst_velocity_joint <= 3
    assert 0 <= analysis_result.worst_velocity_sample < n_samples

    print("\nAll sanity checks passed.")