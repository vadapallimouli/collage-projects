"""
robot/dynamics.py
====================

Dynamic Analysis Module
----------------------------
Project: Intelligent Robot Motion Planning with Failure Prediction,
         Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "What forces, torques, power, and energy are required to execute
    this robot trajectory?"

Given an already-generated `Trajectory` (time, q, qd, qdd) and the
existing, tested `Planar3RRobot` model, this module computes:

    - joint torque at every trajectory sample                (N, 3)
    - instantaneous mechanical power at every sample          (N,)
    - cumulative mechanical energy along the trajectory       (N,)
    - maximum ABSOLUTE torque reached by each joint            (3,)
    - peak ABSOLUTE instantaneous power                        scalar
    - total (final cumulative) mechanical energy               scalar
    - average absolute torque across the trajectory            scalar
    - per-joint mechanical energy breakdown                    (3,)

This module intentionally does NOT:
    - interpret whether any measurement is dangerous
    - compare torque/power/energy against config.py limits
    - classify overload, risk, or failure probability
    - generate explanations
    - analyze obstacles or singularities
    - attempt recovery
    - choose or optimize between trajectories
    - generate or modify trajectories

Those responsibilities belong to their own modules elsewhere in the
project (see the architecture note at the bottom of this docstring).

ROBOT MODEL USAGE
--------------------
This module does not rebuild the robot, and does not create a second
`DHRobot`. It relies entirely on the existing, tested `Planar3RRobot`:

    toolbox_robot = self.robot.get_toolbox_robot()
    tau = toolbox_robot.rne(q[i], qd[i], qdd[i])

Gravity is never re-specified here. `Planar3RRobot._build_toolbox_robot`
already sets `toolbox_robot.gravity = [0, -GRAVITY, 0]` (see the
"GRAVITY CONVENTION" note in `robot/robot_model.py`), so `rne()` already
accounts for gravity torque. This module must not add a second gravity
contribution, or gravity would be double-counted.

DATA CONTRACT NOTE (IMPORTANT -- READ BEFORE USING THIS FILE)
-----------------------------------------------------------------
`models/data_models.py` already defines `DynamicResult` as a fixed data
contract:

    joint_torques: np.ndarray          # (N, 3)
    max_joint_torque: np.ndarray       # (3,) per-joint, not a single float
    power: np.ndarray                  # (N,)
    total_energy: float
    average_torque: float
    energy_per_joint: Optional[np.ndarray] = None   # (3,)

This module uses that dataclass exactly as-is: field names are not
renamed, and no competing/duplicate result class is created.

The task specification that motivated this module additionally asked
for two further outputs: a per-sample `cumulative_energy` series
(shape (N,)) and a scalar `peak_power`. **Neither field exists on the
current `DynamicResult` dataclass.** Per the instruction to flag any
mismatch rather than silently invent conflicting field names, this
module:

    - computes the full cumulative-energy series and peak power
      internally (needed anyway to compute `total_energy`, and useful
      for the demo/sanity checks below),
    - exposes them to callers via two small internal helper methods
      (`analyze_with_series`) that returns them ALONGSIDE the official
      `DynamicResult`, WITHOUT altering `DynamicResult` itself,
    - populates every field that DOES exist on `DynamicResult`
      (`joint_torques`, `max_joint_torque`, `power`, `total_energy`,
      `average_torque`, and the optional `energy_per_joint`).

If a future revision of `DynamicResult` adds `cumulative_energy` and/or
`peak_power` fields, this module can be updated to also store them
directly on the dataclass.

`total_energy` is the FINAL cumulative signed mechanical work, i.e.
`total_energy == cumulative_energy[-1]`. It represents integrated
signed mechanical work, NOT electrical/battery energy consumption --
this project has no motor-efficiency model. It may be positive,
negative, or near zero, depending on how much of the motion is
accelerating (consuming) vs. decelerating (regenerative) with respect
to the configured gravity field.

ARCHITECTURAL BOUNDARY
--------------------------
    robot_model.py -> "What robot do we have?"
    kinematics.py  -> "What is the robot doing kinematically?"
    dynamics.py    -> "What forces/torques/energy are involved?"  (THIS FILE)
    analysis/      -> "Is what happened dangerous?"
    prediction/    -> "What is the estimated failure risk?"
    explainability/-> "Why is it risky?"
    recovery/      -> "Can we fix it?"
    optimization/  -> "Which final path is best?"
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from robot.robot_model import Planar3RRobot
from models.data_models import Trajectory, DynamicResult


class DynamicAnalyzer:
    """
    Computes per-sample dynamic measurements (torque, power, energy)
    for a single trajectory, using an existing `Planar3RRobot` instance
    and its underlying Robotics Toolbox inverse-dynamics solver.

    This class performs measurement only. It does not interpret,
    classify, or judge any of the values it calculates -- that
    responsibility belongs to later modules (e.g.
    `analysis/dynamic_analysis.py`, `prediction/risk_calculator.py`).

    Attributes:
        robot: The `Planar3RRobot` instance used for all inverse-
            dynamics calculations. Never rebuilt or duplicated by this
            class.
    """

    def __init__(self, robot: Planar3RRobot) -> None:
        self.robot = robot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, trajectory: Trajectory) -> DynamicResult:
        """
        Run the complete dynamic analysis for one trajectory and return
        the project's existing `DynamicResult` dataclass, unmodified in
        shape/field names.

        Args:
            trajectory: A `Trajectory` instance with `time` shape (N,)
                and `q`, `qd`, `qdd` each shape (N, 3). Treated as
                read-only; never modified.

        Returns:
            A `DynamicResult` populated with:
                joint_torques       (N, 3)  N*m, signed
                max_joint_torque    (3,)    N*m, per-joint max |torque|
                power               (N,)    Watts, signed mechanical
                                             power
                total_energy        float   Joules, final cumulative
                                             signed mechanical work
                average_torque      float   N*m, mean |torque| across
                                             all joints and samples
                energy_per_joint    (3,)    Joules, per-joint signed
                                             mechanical work

        Raises:
            ValueError: if the trajectory's arrays are inconsistent,
                contain non-finite values, or `time` is not strictly
                increasing (see `_validate_trajectory`).
        """
        result, _cumulative_energy, _peak_power = self.analyze_with_series(
            trajectory
        )
        return result

    def analyze_with_series(
        self, trajectory: Trajectory
    ) -> Tuple[DynamicResult, np.ndarray, float]:
        """
        Same as `analyze`, but also returns the full per-sample
        cumulative-energy series and the scalar peak (absolute) power.

        These two values are NOT part of the current `DynamicResult`
        data contract (see the "DATA CONTRACT NOTE" in the module
        docstring), so they are returned separately instead of being
        attached to the dataclass.

        Returns:
            Tuple of:
                result: DynamicResult (see `analyze`)
                cumulative_energy: np.ndarray, shape (N,), Joules
                peak_power: float, Watts (max absolute instantaneous
                    power)
        """
        self._validate_trajectory(trajectory)

        # Read-only copies so we never mutate the input trajectory.
        time = np.array(trajectory.time, dtype=float, copy=True)
        q = np.array(trajectory.q, dtype=float, copy=True)
        qd = np.array(trajectory.qd, dtype=float, copy=True)
        qdd = np.array(trajectory.qdd, dtype=float, copy=True)

        joint_torques = self._compute_torques(q, qd, qdd)
        power, per_joint_power = self._compute_power(joint_torques, qd)
        cumulative_energy, energy_per_joint = self._compute_energy(
            time, power, per_joint_power
        )

        max_joint_torque = self._compute_max_joint_torque(joint_torques)
        peak_power = float(np.max(np.abs(power)))
        total_energy = float(cumulative_energy[-1])
        average_torque = float(np.mean(np.abs(joint_torques)))

        result = DynamicResult(
            joint_torques=joint_torques,
            max_joint_torque=max_joint_torque,
            power=power,
            total_energy=total_energy,
            average_torque=average_torque,
            energy_per_joint=energy_per_joint,
        )
        return result, cumulative_energy, peak_power

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_trajectory(trajectory: Trajectory) -> None:
        """
        Validate that the trajectory's arrays are shaped, sized, and
        numerically well-formed. Raises ValueError on any inconsistency.
        Never resizes, truncates, reorders, or interpolates the
        trajectory.
        """
        time = np.asarray(trajectory.time, dtype=float)
        q = np.asarray(trajectory.q, dtype=float)
        qd = np.asarray(trajectory.qd, dtype=float)
        qdd = np.asarray(trajectory.qdd, dtype=float)

        if time.ndim != 1:
            raise ValueError(
                f"trajectory.time must be 1-D with shape (N,), got shape "
                f"{time.shape}."
            )

        n_samples = time.shape[0]

        if n_samples == 0:
            raise ValueError(
                "trajectory.time has 0 samples; an empty trajectory "
                "cannot be analyzed."
            )

        for name, array in (("q", q), ("qd", qd), ("qdd", qdd)):
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(
                    f"trajectory.{name} must have shape (N, 3), got shape "
                    f"{array.shape}."
                )
            if array.shape[0] != n_samples:
                raise ValueError(
                    f"trajectory.{name} has {array.shape[0]} samples but "
                    f"trajectory.time has {n_samples} samples. All "
                    f"trajectory arrays must have the same sample count."
                )

        for name, array in (
            ("time", time),
            ("q", q),
            ("qd", qd),
            ("qdd", qdd),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"trajectory.{name} contains NaN or infinite values, "
                    f"which are not allowed."
                )

        if n_samples > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("trajectory.time must be strictly increasing.")

    # ------------------------------------------------------------------
    # Torque
    # ------------------------------------------------------------------

    def _compute_torques(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray
    ) -> np.ndarray:
        """
        Compute signed joint torque at every trajectory sample using the
        robot model's underlying Robotics Toolbox inverse-dynamics
        solver (`rne`). Gravity is taken from the robot model's own
        configuration (see module docstring "ROBOT MODEL USAGE") and is
        never re-specified or double-applied here.

        Args:
            q: shape (N, 3), radians.
            qd: shape (N, 3), radians/second.
            qdd: shape (N, 3), radians/second^2.

        Returns:
            np.ndarray, shape (N, 3), N*m, signed.
        """
        toolbox_robot = self.robot.get_toolbox_robot()
        n_samples = q.shape[0]
        joint_torques = np.zeros((n_samples, 3), dtype=float)

        for i in range(n_samples):
            tau = toolbox_robot.rne(q[i], qd[i], qdd[i])
            joint_torques[i] = np.asarray(tau, dtype=float).reshape(3)

        return joint_torques

    @staticmethod
    def _compute_max_joint_torque(joint_torques: np.ndarray) -> np.ndarray:
        """
        Compute the maximum ABSOLUTE torque reached by each joint
        individually (not a single global maximum), so downstream
        explainability can name the specific joint responsible.

        Args:
            joint_torques: shape (N, 3), N*m, signed.

        Returns:
            np.ndarray, shape (3,), N*m.
        """
        return np.max(np.abs(joint_torques), axis=0)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_power(
        joint_torques: np.ndarray, qd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total instantaneous mechanical power (P = tau . qd) at
        every sample, plus the per-joint power contributions (needed to
        later compute a per-joint energy breakdown).

        Signed power is preserved: negative power (regenerative /
        decelerating motion) is NOT clamped to zero.

        Args:
            joint_torques: shape (N, 3), N*m, signed.
            qd: shape (N, 3), radians/second.

        Returns:
            Tuple of:
                power: shape (N,), Watts, signed total power
                per_joint_power: shape (N, 3), Watts, signed per-joint
                    power contribution (tau_j * qd_j)
        """
        per_joint_power = joint_torques * qd  # elementwise, shape (N, 3)
        power = np.sum(per_joint_power, axis=1)  # shape (N,)
        return power, per_joint_power

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_energy(
        time: np.ndarray, power: np.ndarray, per_joint_power: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate signed mechanical power over the trajectory's actual
        (non-uniform) time samples using trapezoidal integration, to
        obtain a cumulative signed mechanical-energy series, plus a
        per-joint energy breakdown.

        For each interval:
            energy_increment[i] = 0.5 * (P[i] + P[i+1]) * (t[i+1] - t[i])

        cumulative_energy[i] is the integrated signed mechanical work
        from the start of the trajectory through sample i.
        cumulative_energy[0] == 0.0 by definition.

        Args:
            time: shape (N,), seconds, strictly increasing.
            power: shape (N,), Watts, signed total power.
            per_joint_power: shape (N, 3), Watts, signed per-joint power.

        Returns:
            Tuple of:
                cumulative_energy: shape (N,), Joules, signed
                energy_per_joint: shape (3,), Joules, signed total work
                    done by each joint (final cumulative value per
                    joint)
        """
        n_samples = time.shape[0]

        cumulative_energy = np.zeros(n_samples, dtype=float)
        if n_samples > 1:
            dt = np.diff(time)
            increments = 0.5 * (power[:-1] + power[1:]) * dt
            cumulative_energy[1:] = np.cumsum(increments)

        energy_per_joint = np.zeros(3, dtype=float)
        if n_samples > 1:
            dt = np.diff(time)
            for j in range(3):
                joint_power_j = per_joint_power[:, j]
                joint_increments = 0.5 * (joint_power_j[:-1] + joint_power_j[1:]) * dt
                energy_per_joint[j] = float(np.sum(joint_increments))

        return cumulative_energy, energy_per_joint


# =============================================================================
# Demo / dynamics sanity check
# =============================================================================

if __name__ == "__main__":
    # Minimal synthetic trajectory used ONLY to sanity-check this
    # module's calculations. No risk, obstacle, explanation, recovery,
    # or optimization logic is exercised here.
    n_samples = 100
    duration = 4.0
    time = np.linspace(0.0, duration, n_samples)

    # Smooth joint motion from 0 to small target angles for each joint,
    # matching the style of the existing kinematics.py demo.
    q1_target, q2_target, q3_target = 0.4, 0.3, 0.2
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
        name="Dynamics sanity-check trajectory",
        description="Synthetic trajectory used only to test dynamics.py",
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=robot.end_effector_position(q[0]),
        goal_position=robot.end_effector_position(q[-1]),
        duration=duration,
        path_length=0.0,
    )

    analyzer = DynamicAnalyzer(robot)
    result, cumulative_energy, peak_power = analyzer.analyze_with_series(
        demo_trajectory
    )

    print(f"Number of samples: {n_samples}")
    print(f"Torque history shape: {result.joint_torques.shape}")
    print(f"Power history shape: {result.power.shape}")
    print(f"Cumulative energy series shape: {cumulative_energy.shape}")
    print(f"\nMaximum |torque| per joint (N*m): {result.max_joint_torque}")
    print(f"Average |torque| across all joints/samples (N*m): "
          f"{result.average_torque:.4f}")
    print(f"Peak |power| (W): {peak_power:.4f}")
    print(f"Total (final cumulative) mechanical energy (J): "
          f"{result.total_energy:.4f}")
    print(f"Per-joint mechanical energy (J): {result.energy_per_joint}")

    # Sanity checks (assertions only, no interpretation/classification).
    assert result.joint_torques.shape == (n_samples, 3)
    assert result.power.shape == (n_samples,)
    assert cumulative_energy.shape == (n_samples,)
    assert result.max_joint_torque.shape == (3,)
    assert np.all(np.isfinite(result.joint_torques))
    assert np.all(np.isfinite(result.power))
    assert np.all(np.isfinite(cumulative_energy))
    assert np.isclose(result.total_energy, cumulative_energy[-1])
    print("\nAll sanity checks passed.")