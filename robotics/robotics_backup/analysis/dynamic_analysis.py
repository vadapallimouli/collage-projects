"""
analysis/dynamic_analysis.py
================================

Dynamic Safety Analysis Module
-----------------------------------
Project: Intelligent Robot Motion Planning with Failure Prediction,
         Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "Did the dynamic measurements of this trajectory violate or
     approach the robot's configured dynamic limits?"

It takes the already-computed `DynamicResult` (produced by
`robot/dynamics.py`) and compares the relevant measurements against
the robot's configured torque limits from `config.py`
(`MAX_JOINT_TORQUES`). It produces structured, numerical/Boolean
dynamic-safety evidence -- not a final failure verdict.

This module intentionally does NOT:
    - recompute any dynamic quantity (torque, power, energy)
    - calculate failure probability, risk score, risk percentage,
      overall risk, or a SAFE/UNSAFE classification
    - create an "energy violation" (no configured energy limit exists)
    - analyze joint position/velocity limits, manipulability, or
      singularity severity
    - analyze obstacles or clearance
    - generate natural-language explanations
    - suggest or apply recovery actions
    - compare or select between multiple trajectories
    - modify the input trajectory or DynamicResult in any way

Those responsibilities belong to their own modules elsewhere in the
project (see the architecture note at the bottom of this docstring).

ROBOT MODEL / CONFIG USAGE
------------------------------
This module does not need a `Planar3RRobot` instance -- all of the
dynamic measurements it needs are already present on the input
`DynamicResult`. The only external source of truth it relies on is
`config.py`, specifically:

    MAX_JOINT_TORQUES   (N*m, shape (3,))

This value is never duplicated or re-hard-coded here.

DATA CONTRACT NOTE (IMPORTANT -- READ BEFORE USING THIS FILE)
-----------------------------------------------------------------
`models/data_models.py` defines `DynamicAnalysisResult` as the
project's existing analysis-result data contract for this module,
following exactly the same pattern already established by
`KinematicAnalysisResult` (used by `analysis/kinematic_analysis.py`).
This module uses `DynamicAnalysisResult` exactly as-is: field names
are not renamed, and no competing/duplicate result class is created
here. Its fields are:

    torque_limit_violated: bool
    torque_violation_count: np.ndarray        # (3,) int
    max_torque_utilization: np.ndarray        # (3,) float
    max_torque_excess: np.ndarray             # (3,) float, N*m
    worst_torque_joint: int                   # 1-based
    worst_torque_sample: int                  # 0-based
    peak_power: float                         # W
    average_torque: float                     # N*m
    total_energy: float                       # J
    energy_per_joint: np.ndarray              # (3,) J

If a future revision of the project's data-model layer is found to be
missing this dataclass, that mismatch should be reported rather than
silently inventing a second, competing `DynamicAnalysisResult`.

ARCHITECTURAL BOUNDARY
--------------------------
    robot_model.py             -> "What robot do we have?"
    kinematics.py               -> "What is the robot doing kinematically?"
    dynamics.py                  -> "What forces/torques/power/energy are involved?"
    kinematic_analysis.py       -> "Did kinematics violate configured limits?"
    dynamic_analysis.py         -> "Did dynamics violate configured limits?"  (THIS FILE)
    obstacle_analysis.py        -> "Is the path too close to obstacles?"
    singularity_analysis.py      -> "How severe is the singularity risk?"
    prediction/                  -> "What is the estimated failure risk?"
    explainability/               -> "Why is it risky?"
    recovery/                     -> "Can we fix it?"
    optimization/                  -> "Which final path is best?"
"""

from __future__ import annotations

import numpy as np

from config import MAX_JOINT_TORQUES
from models.data_models import DynamicAnalysisResult, DynamicResult


class DynamicSafetyAnalyzer:
    """
    Interprets an existing `DynamicResult` against the robot's
    configured torque limits (`config.py`) and produces structured
    dynamic-safety evidence.

    This class performs interpretation only. It does not recompute any
    dynamic quantity, and it does not decide whether the trajectory
    will ultimately "fail" -- that responsibility belongs to
    `prediction/failure_predictor.py`, which combines this evidence
    with evidence from other analysis modules.
    """

    def __init__(self) -> None:
        self._torque_limits = self._validate_configuration(MAX_JOINT_TORQUES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, dynamic_result: DynamicResult) -> DynamicAnalysisResult:
        """
        Run the complete dynamic safety analysis for one
        `DynamicResult` and return the project's existing
        `DynamicAnalysisResult` dataclass, unmodified in shape/field
        names.

        Args:
            dynamic_result: A `DynamicResult` instance with
                `joint_torques` shape (N, 3), `power` shape (N,),
                `max_joint_torque` shape (3,), `total_energy` and
                `average_torque` scalars, and optional
                `energy_per_joint` shape (3,). Treated as read-only;
                never modified.

        Returns:
            A `DynamicAnalysisResult` populated with torque-limit
            evidence and pass-through power/energy measurements.

        Raises:
            ValueError: if the input arrays are inconsistent, wrongly
                shaped, or contain non-finite values (see
                `_validate_input`).
        """
        joint_torques, power, energy_per_joint = self._validate_input(
            dynamic_result
        )

        (
            torque_limit_violated,
            torque_violation_count,
            max_torque_utilization,
            max_torque_excess,
            worst_torque_joint,
            worst_torque_sample,
        ) = self._analyze_torque_limits(joint_torques)

        peak_power = float(np.max(np.abs(power)))
        average_torque = float(dynamic_result.average_torque)
        total_energy = float(dynamic_result.total_energy)

        return DynamicAnalysisResult(
            torque_limit_violated=torque_limit_violated,
            torque_violation_count=torque_violation_count,
            max_torque_utilization=max_torque_utilization,
            max_torque_excess=max_torque_excess,
            worst_torque_joint=worst_torque_joint,
            worst_torque_sample=worst_torque_sample,
            peak_power=peak_power,
            average_torque=average_torque,
            total_energy=total_energy,
            energy_per_joint=energy_per_joint,
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_configuration(max_joint_torques) -> np.ndarray:
        """
        Validate that `MAX_JOINT_TORQUES` (from config.py) contains
        exactly 3 finite, strictly positive values. Raises ValueError
        otherwise.
        """
        limits = np.asarray(max_joint_torques, dtype=float)
        if limits.shape != (3,):
            raise ValueError(
                f"config.MAX_JOINT_TORQUES must have shape (3,), got "
                f"shape {limits.shape}."
            )
        if not np.all(np.isfinite(limits)):
            raise ValueError(
                "config.MAX_JOINT_TORQUES contains NaN or infinite "
                "values, which are not allowed."
            )
        if np.any(limits <= 0.0):
            raise ValueError(
                f"config.MAX_JOINT_TORQUES must be strictly positive, "
                f"got {limits}."
            )
        return limits

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(
        dynamic_result: DynamicResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate `dynamic_result.joint_torques`, `power`,
        `max_joint_torque`, `total_energy`, `average_torque`, and
        `energy_per_joint`. Returns read-only copies of
        `joint_torques`, `power`, and `energy_per_joint` so the input
        is never mutated.

        Raises:
            ValueError: on any shape mismatch, non-finite value, or
                missing required `energy_per_joint` data.
        """
        joint_torques = np.asarray(dynamic_result.joint_torques, dtype=float)
        power = np.asarray(dynamic_result.power, dtype=float)
        max_joint_torque = np.asarray(
            dynamic_result.max_joint_torque, dtype=float
        )

        if joint_torques.ndim != 2 or joint_torques.shape[1] != 3:
            raise ValueError(
                f"dynamic_result.joint_torques must have shape (N, 3), "
                f"got shape {joint_torques.shape}."
            )
        n_samples = joint_torques.shape[0]
        if n_samples == 0:
            raise ValueError(
                "dynamic_result.joint_torques has 0 samples; an empty "
                "trajectory cannot be analyzed."
            )
        if not np.all(np.isfinite(joint_torques)):
            raise ValueError(
                "dynamic_result.joint_torques contains NaN or infinite "
                "values, which are not allowed."
            )

        if power.ndim != 1 or power.shape[0] != n_samples:
            raise ValueError(
                f"dynamic_result.power must have shape ({n_samples},) to "
                f"match joint_torques, got shape {power.shape}."
            )
        if not np.all(np.isfinite(power)):
            raise ValueError(
                "dynamic_result.power contains NaN or infinite values, "
                "which are not allowed."
            )

        if max_joint_torque.shape != (3,):
            raise ValueError(
                f"dynamic_result.max_joint_torque must have shape (3,), "
                f"got shape {max_joint_torque.shape}."
            )
        if not np.all(np.isfinite(max_joint_torque)):
            raise ValueError(
                "dynamic_result.max_joint_torque contains NaN or "
                "infinite values, which are not allowed."
            )

        total_energy = dynamic_result.total_energy
        if not np.isfinite(total_energy):
            raise ValueError(
                f"dynamic_result.total_energy must be finite, got "
                f"{total_energy}."
            )

        average_torque = dynamic_result.average_torque
        if not np.isfinite(average_torque):
            raise ValueError(
                f"dynamic_result.average_torque must be finite, got "
                f"{average_torque}."
            )

        if dynamic_result.energy_per_joint is None:
            raise ValueError(
                "dynamic_result.energy_per_joint is required (shape "
                "(3,)) but is None. This analyzer does not silently "
                "invent per-joint energy values."
            )
        energy_per_joint = np.asarray(
            dynamic_result.energy_per_joint, dtype=float
        )
        if energy_per_joint.shape != (3,):
            raise ValueError(
                f"dynamic_result.energy_per_joint must have shape (3,), "
                f"got shape {energy_per_joint.shape}."
            )
        if not np.all(np.isfinite(energy_per_joint)):
            raise ValueError(
                "dynamic_result.energy_per_joint contains NaN or "
                "infinite values, which are not allowed."
            )

        # Return read-only copies -- never mutate the caller's arrays.
        return (
            np.array(joint_torques, copy=True),
            np.array(power, copy=True),
            np.array(energy_per_joint, copy=True),
        )

    # ------------------------------------------------------------------
    # Torque-limit analysis
    # ------------------------------------------------------------------

    def _analyze_torque_limits(
        self, joint_torques: np.ndarray
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Compare per-sample joint torques against
        `config.MAX_JOINT_TORQUES` and derive torque-limit evidence.

        Args:
            joint_torques: shape (N, 3), N*m, signed.

        Returns:
            Tuple of:
                torque_limit_violated: bool, True if ANY joint exceeds
                    its torque limit at ANY sample.
                torque_violation_count: np.ndarray, shape (3,), int,
                    per-joint count of violating samples.
                max_torque_utilization: np.ndarray, shape (3,), float,
                    per-joint max(|torque| / limit).
                max_torque_excess: np.ndarray, shape (3,), float, N*m,
                    per-joint max(0, max(|torque| - limit)).
                worst_torque_joint: int, 1-based joint index with the
                    single highest torque utilization across all
                    samples/joints.
                worst_torque_sample: int, 0-based sample index of that
                    worst utilization.
        """
        abs_torque = np.abs(joint_torques)  # (N, 3)

        # Broadcasting (N,3) / (3,) -> (N,3)
        torque_utilization = abs_torque / self._torque_limits
        torque_excess = abs_torque - self._torque_limits

        # Strict greater-than: exactly at the limit is NOT a violation.
        torque_violation_mask = abs_torque > self._torque_limits  # (N,3)

        torque_violation_count = np.sum(
            torque_violation_mask, axis=0
        ).astype(int)  # (3,)
        torque_limit_violated = bool(np.any(torque_violation_mask))

        max_torque_utilization = np.max(torque_utilization, axis=0)  # (3,)
        max_torque_excess = np.maximum(
            np.max(torque_excess, axis=0), 0.0
        )  # (3,)

        worst_flat_index = int(np.argmax(torque_utilization))
        worst_sample, worst_joint_zero_based = np.unravel_index(
            worst_flat_index, torque_utilization.shape
        )
        worst_torque_joint = int(worst_joint_zero_based) + 1  # 1-based
        worst_torque_sample = int(worst_sample)  # 0-based

        return (
            torque_limit_violated,
            torque_violation_count,
            max_torque_utilization,
            max_torque_excess,
            worst_torque_joint,
            worst_torque_sample,
        )


# =============================================================================
# Demo / dynamic-safety-analysis sanity check
# =============================================================================

if __name__ == "__main__":
    from robot.robot_model import Planar3RRobot
    from robot.dynamics import DynamicAnalyzer
    from models.data_models import Trajectory

    # Dynamic safety analysis demo -- synthetic trajectory used ONLY to
    # sanity-check this module's calculations. No risk, obstacle,
    # explanation, or recovery logic is exercised here. This is not a
    # real robot failure; the trajectory is intentionally aggressive
    # enough to push Joint 1's torque past its configured limit, purely
    # to exercise the violation-detection logic in this demo.
    n_samples = 100
    duration = 4.0
    time = np.linspace(0.0, duration, n_samples)

    # Smooth joint motion from 0 to target angles. Joint 1's target is
    # intentionally large enough to require torque above the configured
    # limit (50 N*m) for part of the trajectory, purely to exercise the
    # violation-detection logic in this demo.
    q1_target, q2_target, q3_target = 2.6, 0.8, 0.3
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
        name="Dynamic safety analysis demo",
        description="Synthetic test trajectory used only to test dynamic_analysis.py",
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=robot.end_effector_position(q[0]),
        goal_position=robot.end_effector_position(q[-1]),
        duration=duration,
        path_length=0.0,
    )

    dynamic_analyzer = DynamicAnalyzer(robot)
    dynamic_result = dynamic_analyzer.analyze(demo_trajectory)

    safety_analyzer = DynamicSafetyAnalyzer()
    analysis_result = safety_analyzer.analyze(dynamic_result)

    print(f"Number of samples: {n_samples}")

    print(f"\nTorque limit violated: {analysis_result.torque_limit_violated}")
    print(f"Torque violation count (per joint): "
          f"{analysis_result.torque_violation_count}")
    print(f"Maximum torque utilization (per joint): "
          f"{analysis_result.max_torque_utilization}")
    print(f"Maximum torque excess (N*m, per joint): "
          f"{analysis_result.max_torque_excess}")
    print(f"Worst torque joint (1-based): "
          f"Joint {analysis_result.worst_torque_joint}")
    print(f"Worst torque sample (0-based): "
          f"{analysis_result.worst_torque_sample}")

    print(f"\nPeak absolute mechanical power (W): "
          f"{analysis_result.peak_power:.4f}")
    print(f"Average absolute torque (N*m): "
          f"{analysis_result.average_torque:.4f}")
    print(f"Total signed mechanical work (J): "
          f"{analysis_result.total_energy:.4f}")
    print(f"Mechanical work per joint (J): "
          f"{analysis_result.energy_per_joint}")

    # Sanity checks (assertions only, no interpretation/classification).
    assert analysis_result.torque_violation_count.shape == (3,)
    assert analysis_result.max_torque_utilization.shape == (3,)
    assert analysis_result.max_torque_excess.shape == (3,)
    assert analysis_result.energy_per_joint.shape == (3,)

    assert np.isfinite(analysis_result.max_torque_utilization).all()
    assert np.isfinite(analysis_result.max_torque_excess).all()
    assert np.isfinite(analysis_result.peak_power)
    assert np.isfinite(analysis_result.average_torque)
    assert np.isfinite(analysis_result.total_energy)
    assert np.isfinite(analysis_result.energy_per_joint).all()

    assert 1 <= analysis_result.worst_torque_joint <= 3
    assert 0 <= analysis_result.worst_torque_sample < n_samples

    print("\nAll sanity checks passed.")