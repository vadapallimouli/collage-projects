"""
analysis/singularity_analysis.py
=================================

Singularity Proximity Analysis Module
--------------------------------------

Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

Purpose
-------
This module determines how close a robot trajectory comes to a
kinematic singular configuration.

For every trajectory sample, it calculates:

    1. Planar Jacobian
    2. Singular values of the Jacobian
    3. Manipulability
    4. Smallest singular value (singularity metric)

Then it summarizes the trajectory using:

    - minimum manipulability
    - minimum singularity metric
    - worst trajectory sample
    - configured singularity interpretation
    - near-singularity flag

IMPORTANT ARCHITECTURAL BOUNDARY
---------------------------------

This module provides singularity evidence.

It does NOT calculate the final project failure probability.

The larger project pipeline is:

    Trajectory
        |
        +-----------------------------+
        |                             |
        v                             v
    Kinematic Analysis          Dynamic Analysis
        |                             |
        v                             v
    Kinematic Evidence          Dynamic Evidence
        |                             |
        +-------------+---------------+
                      |
                      v
               Obstacle Analysis
                      |
                      v
             Singularity Analysis
                      |
                      v
              Feature Extraction
                      |
                      v
             Failure Prediction
                      |
                      v
              Explainable AI
                      |
                      v
               Recoverability
                      |
                      v
                Optimization

The final failure probability belongs to the prediction layer,
not this module.

DATA CONTRACT
-------------

The existing models/data_models.py contains:

    SingularityResult

with fields:

    minimum_manipulability: float
    worst_condition_metric: float
    risk_level: str
    near_singularity: bool
    worst_sample_index: Optional[int]

This module uses that existing dataclass exactly.

IMPORTANT:
    `worst_condition_metric` is used to store the minimum singular
    value because that is the project's existing field name.

It is NOT the classical matrix condition number.

Classical condition number would be:

    sigma_max / sigma_min

This module intentionally does not calculate that value because
the current data contract does not require it.

SINGULARITY METRIC
------------------

For this project:

    singularity_metric = minimum singular value

Therefore:

    large minimum singular value
        -> better conditioned

    small minimum singular value
        -> closer to singularity

    approximately zero
        -> singular configuration

MANIPULABILITY
--------------

For singular values:

    sigma_1
    sigma_2
    sigma_3

the manipulability measure used here is:

    manipulability = sigma_1 * sigma_2 * sigma_3

This is equivalent to the absolute determinant of the square
planar Jacobian, while SVD provides better numerical stability
near singular configurations.

PLANAR ROBOT
------------

The robot is a planar 3R robot.

The full Robotics Toolbox Jacobian is 6x3.

Only the physically relevant planar task-space rows are used:

    row 0 -> X linear velocity
    row 1 -> Y linear velocity
    row 5 -> Z angular velocity

Therefore:

    J_planar = J_full[[0, 1, 5], :]

which produces a 3x3 Jacobian.

THRESHOLDS
----------

The existing config.py contains:

    SINGULARITY_THRESHOLDS

with:

    HIGH_RISK_BELOW
    LOW_RISK_AT_OR_ABOVE

These configured values are used because the existing
SingularityResult requires:

    risk_level
    near_singularity

No new thresholds are invented in this module.

The `risk_level` here is only the configured singularity
interpretation required by the existing data contract.

It is NOT the final failure probability.

ARCHITECTURAL RESPONSIBILITIES
-------------------------------

robot/robot_model.py
    What robot do we have?

robot/kinematics.py
    What is the robot doing kinematically?

robot/dynamics.py
    What forces, torque, power, and energy are involved?

analysis/kinematic_analysis.py
    Did the trajectory violate kinematic limits?

analysis/dynamic_analysis.py
    Did the trajectory violate dynamic limits?

analysis/obstacle_analysis.py
    Is the trajectory too close to obstacles?

analysis/singularity_analysis.py
    How close is the trajectory to singularity?

prediction/
    What is the estimated probability of failure?

explainability/
    Why is the trajectory risky?

recovery/
    Can the trajectory be improved?

optimization/
    Which candidate path is best?
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config import SINGULARITY_THRESHOLDS
from models.data_models import SingularityResult, Trajectory
from robot.robot_model import Planar3RRobot


class SingularityAnalyzer:
    """
    Analyze singularity proximity along one robot trajectory.

    The analyzer uses the existing Planar3RRobot instance and its
    Jacobian calculation.

    It does not create another robot model.

    Attributes:
        robot:
            Existing Planar3RRobot instance.

        _high_risk_below:
            Configured lower singularity threshold.

        _low_risk_at_or_above:
            Configured threshold above which the singularity
            condition is interpreted as LOW.
    """

    def __init__(self, robot: Planar3RRobot) -> None:
        """
        Initialize the singularity analyzer.

        Args:
            robot:
                Existing Planar3RRobot instance.

        Raises:
            ValueError:
                If the configured singularity thresholds are invalid.
        """

        self.robot = robot

        (
            self._high_risk_below,
            self._low_risk_at_or_above,
        ) = self._validate_singularity_thresholds(
            SINGULARITY_THRESHOLDS
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def analyze(
        self,
        trajectory: Trajectory,
    ) -> SingularityResult:
        """
        Analyze one trajectory.

        This method returns only the official project
        SingularityResult.

        Args:
            trajectory:
                Trajectory containing q with shape (N, 3).

        Returns:
            SingularityResult containing:

                minimum_manipulability
                worst_condition_metric
                risk_level
                near_singularity
                worst_sample_index

        The trajectory is never modified.
        """

        (
            result,
            _manipulability_history,
            _singularity_metric_history,
            _worst_sample,
        ) = self.analyze_with_series(
            trajectory
        )

        return result

    def analyze_with_series(
        self,
        trajectory: Trajectory,
    ) -> Tuple[
        SingularityResult,
        np.ndarray,
        np.ndarray,
        int,
    ]:
        """
        Analyze one trajectory and additionally return the full
        per-sample singularity histories.

        Returns:
            result:
                Official SingularityResult.

            manipulability_history:
                Shape (N,).
                Manipulability at every trajectory sample.

            singularity_metric_history:
                Shape (N,).
                Minimum singular value at every trajectory sample.

            worst_sample:
                Zero-based trajectory index where the minimum
                singularity metric occurs.
        """

        # ---------------------------------------------------------------------
        # Validate trajectory
        # ---------------------------------------------------------------------

        q = self._validate_trajectory(
            trajectory
        )

        n_samples = q.shape[0]

        # ---------------------------------------------------------------------
        # Allocate histories
        # ---------------------------------------------------------------------

        manipulability_history = np.zeros(
            n_samples,
            dtype=float,
        )

        singularity_metric_history = np.zeros(
            n_samples,
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # Analyze every trajectory sample
        # ---------------------------------------------------------------------

        for sample_index in range(
            n_samples
        ):

            q_sample = q[
                sample_index
            ]

            # -------------------------------------------------------------
            # Compute planar Jacobian
            # -------------------------------------------------------------

            j_planar = (
                self._compute_planar_jacobian(
                    q_sample
                )
            )

            # -------------------------------------------------------------
            # Compute singular values
            # -------------------------------------------------------------

            singular_values = (
                self._compute_singular_values(
                    j_planar
                )
            )

            # -------------------------------------------------------------
            # Manipulability
            # -------------------------------------------------------------

            manipulability = float(
                np.prod(
                    singular_values
                )
            )

            # Numerical noise can theoretically produce a tiny negative
            # value in derived calculations. Singular values themselves
            # are non-negative, so explicitly protect the stored value.
            manipulability = max(
                0.0,
                manipulability,
            )

            manipulability_history[
                sample_index
            ] = manipulability

            # -------------------------------------------------------------
            # Singularity metric
            # -------------------------------------------------------------

            minimum_singular_value = float(
                np.min(
                    singular_values
                )
            )

            minimum_singular_value = max(
                0.0,
                minimum_singular_value,
            )

            singularity_metric_history[
                sample_index
            ] = minimum_singular_value

        # ---------------------------------------------------------------------
        # Trajectory-level summary
        # ---------------------------------------------------------------------

        minimum_manipulability = float(
            np.min(
                manipulability_history
            )
        )

        minimum_singularity_metric = float(
            np.min(
                singularity_metric_history
            )
        )

        # The worst point is defined by the smallest singular value,
        # because that directly represents the closest approach to
        # singularity.
        worst_sample = int(
            np.argmin(
                singularity_metric_history
            )
        )

        # ---------------------------------------------------------------------
        # Existing data-contract interpretation
        # ---------------------------------------------------------------------

        risk_level = (
            self._classify_risk_level(
                minimum_singularity_metric
            )
        )

        near_singularity = bool(
            minimum_singularity_metric
            < self._high_risk_below
        )

        # ---------------------------------------------------------------------
        # Construct official project result
        # ---------------------------------------------------------------------

        result = SingularityResult(
            minimum_manipulability=(
                minimum_manipulability
            ),
            worst_condition_metric=(
                minimum_singularity_metric
            ),
            risk_level=risk_level,
            near_singularity=near_singularity,
            worst_sample_index=(
                worst_sample
            ),
        )

        return (
            result,
            manipulability_history,
            singularity_metric_history,
            worst_sample,
        )

    # =========================================================================
    # CONFIGURATION VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_singularity_thresholds(
        thresholds,
    ) -> Tuple[float, float]:
        """
        Validate config.SINGULARITY_THRESHOLDS.

        Required keys:

            HIGH_RISK_BELOW
            LOW_RISK_AT_OR_ABOVE

        Required relationship:

            HIGH_RISK_BELOW
                <
            LOW_RISK_AT_OR_ABOVE
        """

        if not isinstance(
            thresholds,
            dict,
        ):
            raise ValueError(
                "SINGULARITY_THRESHOLDS must be "
                "a dictionary."
            )

        try:
            high_risk_below = float(
                thresholds[
                    "HIGH_RISK_BELOW"
                ]
            )

            low_risk_at_or_above = float(
                thresholds[
                    "LOW_RISK_AT_OR_ABOVE"
                ]
            )

        except KeyError as exc:

            raise ValueError(
                "SINGULARITY_THRESHOLDS must contain "
                "'HIGH_RISK_BELOW' and "
                "'LOW_RISK_AT_OR_ABOVE'."
            ) from exc

        # ---------------------------------------------------------------------
        # Finite values
        # ---------------------------------------------------------------------

        if not np.isfinite(
            high_risk_below
        ):

            raise ValueError(
                "HIGH_RISK_BELOW must be finite."
            )

        if not np.isfinite(
            low_risk_at_or_above
        ):

            raise ValueError(
                "LOW_RISK_AT_OR_ABOVE must be finite."
            )

        # ---------------------------------------------------------------------
        # Non-negative
        # ---------------------------------------------------------------------

        if high_risk_below < 0.0:

            raise ValueError(
                "HIGH_RISK_BELOW must be >= 0."
            )

        if low_risk_at_or_above < 0.0:

            raise ValueError(
                "LOW_RISK_AT_OR_ABOVE must be >= 0."
            )

        # ---------------------------------------------------------------------
        # Correct threshold ordering
        # ---------------------------------------------------------------------

        if (
            high_risk_below
            >= low_risk_at_or_above
        ):

            raise ValueError(
                "SINGULARITY_THRESHOLDS must satisfy "
                "HIGH_RISK_BELOW < LOW_RISK_AT_OR_ABOVE. "
                f"Got HIGH_RISK_BELOW={high_risk_below}, "
                f"LOW_RISK_AT_OR_ABOVE={low_risk_at_or_above}."
            )

        return (
            high_risk_below,
            low_risk_at_or_above,
        )

    # =========================================================================
    # TRAJECTORY VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_trajectory(
        trajectory: Trajectory,
    ) -> np.ndarray:
        """
        Validate trajectory.q.

        Requirements:

            shape = (N, 3)
            N > 0
            all values finite

        Returns:
            Defensive copy of q.

        Raises:
            ValueError:
                If q is invalid.
        """

        q = np.asarray(
            trajectory.q,
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # Shape
        # ---------------------------------------------------------------------

        if (
            q.ndim != 2
            or q.shape[1] != 3
        ):

            raise ValueError(
                "trajectory.q must have "
                f"shape (N, 3), got {q.shape}."
            )

        # ---------------------------------------------------------------------
        # Empty trajectory
        # ---------------------------------------------------------------------

        if q.shape[0] == 0:

            raise ValueError(
                "trajectory.q has 0 samples; "
                "an empty trajectory cannot be analyzed."
            )

        # ---------------------------------------------------------------------
        # Finite values
        # ---------------------------------------------------------------------

        if not np.all(
            np.isfinite(q)
        ):

            raise ValueError(
                "trajectory.q contains NaN or "
                "infinite values."
            )

        return np.array(
            q,
            copy=True,
        )

    # =========================================================================
    # PLANAR JACOBIAN
    # =========================================================================

    def _compute_planar_jacobian(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """
        Extract the physically relevant planar 3x3 Jacobian.

        The Robotics Toolbox returns a 6x3 geometric Jacobian:

            [vx]
            [vy]
            [vz]
            [wx]
            [wy]
            [wz]

        For this planar 3R robot, the relevant task variables are:

            vx
            vy
            wz

        Therefore we select:

            rows [0, 1, 5]

        resulting in:

            J_planar = J_full[[0, 1, 5], :]

        Args:
            q:
                Shape (3,).

        Returns:
            Shape (3, 3).

        Raises:
            ValueError:
                If the Jacobian is not valid.
        """

        q = np.asarray(
            q,
            dtype=float,
        )

        if q.shape != (3,):

            raise ValueError(
                "q must have shape (3,), "
                f"got {q.shape}."
            )

        if not np.all(
            np.isfinite(q)
        ):

            raise ValueError(
                "q contains NaN or infinite values."
            )

        # ---------------------------------------------------------------------
        # Use the existing robot model.
        # ---------------------------------------------------------------------

        j_full = self.robot.jacobian(
            q
        )

        j_full = np.asarray(
            j_full,
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # Validate full Jacobian
        # ---------------------------------------------------------------------

        if j_full.shape != (6, 3):

            raise ValueError(
                "Planar3RRobot.jacobian(q) is expected "
                "to return shape (6, 3), got "
                f"{j_full.shape}."
            )

        if not np.all(
            np.isfinite(j_full)
        ):

            raise ValueError(
                "Robot Jacobian contains NaN or "
                "infinite values."
            )

        # ---------------------------------------------------------------------
        # Extract planar task-space Jacobian
        # ---------------------------------------------------------------------

        j_planar = j_full[
            [0, 1, 5],
            :,
        ]

        if j_planar.shape != (3, 3):

            raise ValueError(
                "Planar Jacobian must have shape "
                f"(3, 3), got {j_planar.shape}."
            )

        if not np.all(
            np.isfinite(j_planar)
        ):

            raise ValueError(
                "Planar Jacobian contains NaN or "
                "infinite values."
            )

        return j_planar

    # =========================================================================
    # SINGULAR VALUE CALCULATION
    # =========================================================================

    @staticmethod
    def _compute_singular_values(
        j_planar: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the singular values of the planar Jacobian.

        SVD is used because it is numerically stable near singular
        configurations.

        For:

            J = U Sigma V^T

        the singular values are the diagonal elements of Sigma.

        Returns:
            Three non-negative singular values, sorted in descending
            order by NumPy.
        """

        j_planar = np.asarray(
            j_planar,
            dtype=float,
        )

        if j_planar.shape != (3, 3):

            raise ValueError(
                "j_planar must have shape "
                f"(3, 3), got {j_planar.shape}."
            )

        if not np.all(
            np.isfinite(j_planar)
        ):

            raise ValueError(
                "j_planar contains NaN or "
                "infinite values."
            )

        singular_values = np.linalg.svd(
            j_planar,
            compute_uv=False,
        )

        singular_values = np.asarray(
            singular_values,
            dtype=float,
        )

        if singular_values.shape != (3,):

            raise ValueError(
                "Expected three singular values, "
                f"got shape {singular_values.shape}."
            )

        if not np.all(
            np.isfinite(singular_values)
        ):

            raise ValueError(
                "Singular value calculation produced "
                "NaN or infinite values."
            )

        # Singular values are mathematically non-negative.
        # Clip extremely tiny negative numerical artifacts, if any.
        singular_values = np.maximum(
            singular_values,
            0.0,
        )

        return singular_values

    # =========================================================================
    # RISK-LEVEL INTERPRETATION REQUIRED BY EXISTING DATA CONTRACT
    # =========================================================================

    def _classify_risk_level(
        self,
        minimum_singularity_metric: float,
    ) -> str:
        """
        Convert the minimum singularity metric into the configured
        LOW/MEDIUM/HIGH interpretation required by SingularityResult.

        Rules:

            metric < HIGH_RISK_BELOW
                -> HIGH

            HIGH_RISK_BELOW <= metric
                < LOW_RISK_AT_OR_ABOVE
                -> MEDIUM

            metric >= LOW_RISK_AT_OR_ABOVE
                -> LOW

        IMPORTANT:

            This is NOT the final project failure probability.

            It is only the singularity interpretation required by
            the existing SingularityResult data contract.
        """

        metric = float(
            minimum_singularity_metric
        )

        if not np.isfinite(
            metric
        ):

            raise ValueError(
                "minimum_singularity_metric must be finite."
            )

        if metric < self._high_risk_below:

            return "HIGH"

        if (
            metric
            < self._low_risk_at_or_above
        ):

            return "MEDIUM"

        return "LOW"


# =============================================================================
# DEMO / SANITY CHECK
# =============================================================================

if __name__ == "__main__":

    """
    Standalone sanity test.

    The trajectory intentionally begins at:

        q = [0, 0, 0]

    This is the known fully extended configuration of the current
    planar 3R robot and is expected to be singular.

    Therefore the first trajectory sample should have:

        manipulability approximately 0
        minimum singular value approximately 0

    The trajectory then moves away from the singular configuration,
    allowing us to verify that the singularity metric increases.
    """

    # -------------------------------------------------------------------------
    # Synthetic trajectory
    # -------------------------------------------------------------------------

    n_samples = 100
    duration = 4.0

    time = np.linspace(
        0.0,
        duration,
        n_samples,
    )

    # Target joint configuration.
    q1_target = 0.5
    q2_target = 0.6
    q3_target = 0.3

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

    # -------------------------------------------------------------------------
    # Derivatives
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Existing robot
    # -------------------------------------------------------------------------

    robot = Planar3RRobot()

    # -------------------------------------------------------------------------
    # Create trajectory
    # -------------------------------------------------------------------------

    demo_trajectory = Trajectory(
        path_id="demo-singularity-001",
        name="Singularity analysis demo",
        description=(
            "Synthetic trajectory starting at the known "
            "singular configuration q=[0,0,0]."
        ),
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=robot.end_effector_position(
            q[0]
        ),
        goal_position=robot.end_effector_position(
            q[-1]
        ),
        duration=duration,
        path_length=0.0,
    )

    # -------------------------------------------------------------------------
    # Analyzer
    # -------------------------------------------------------------------------

    analyzer = SingularityAnalyzer(
        robot
    )

    # -------------------------------------------------------------------------
    # Run analysis
    # -------------------------------------------------------------------------

    (
        result,
        manipulability_history,
        singularity_metric_history,
        worst_sample,
    ) = analyzer.analyze_with_series(
        demo_trajectory
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    print(
        f"Number of samples: "
        f"{n_samples}"
    )

    print(
        "Manipulability history shape: "
        f"{manipulability_history.shape}"
    )

    print(
        "Singularity metric history shape: "
        f"{singularity_metric_history.shape}"
    )

    print(
        "\nMinimum manipulability: "
        f"{result.minimum_manipulability:.6e}"
    )

    print(
        "Minimum singularity metric "
        "(smallest singular value): "
        f"{result.worst_condition_metric:.6e}"
    )

    print(
        "Singularity interpretation: "
        f"{result.risk_level}"
    )

    print(
        "Near singularity: "
        f"{result.near_singularity}"
    )

    print(
        "Worst singularity sample "
        f"(0-based): {result.worst_sample_index}"
    )

    # -------------------------------------------------------------------------
    # Explicit home-configuration sanity check
    # -------------------------------------------------------------------------

    home_q = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    home_jacobian = (
        analyzer._compute_planar_jacobian(
            home_q
        )
    )

    home_singular_values = (
        analyzer._compute_singular_values(
            home_jacobian
        )
    )

    home_manipulability = float(
        np.prod(
            home_singular_values
        )
    )

    home_singularity_metric = float(
        np.min(
            home_singular_values
        )
    )

    print(
        "\nPlanar Jacobian at q=[0, 0, 0]:"
    )

    print(
        home_jacobian
    )

    print(
        "\nSingular values at q=[0, 0, 0]:"
    )

    print(
        home_singular_values
    )

    print(
        "\nManipulability at q=[0, 0, 0]: "
        f"{home_manipulability:.6e}"
    )

    print(
        "Smallest singular value at "
        "q=[0, 0, 0]: "
        f"{home_singularity_metric:.6e}"
    )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    assert (
        manipulability_history.shape
        == (n_samples,)
    )

    assert (
        singularity_metric_history.shape
        == (n_samples,)
    )

    assert np.all(
        np.isfinite(
            manipulability_history
        )
    )

    assert np.all(
        np.isfinite(
            singularity_metric_history
        )
    )

    assert np.all(
        manipulability_history
        >= 0.0
    )

    assert np.all(
        singularity_metric_history
        >= 0.0
    )

    assert np.isclose(
        result.minimum_manipulability,
        np.min(
            manipulability_history
        ),
    )

    assert np.isclose(
        result.worst_condition_metric,
        np.min(
            singularity_metric_history
        ),
    )

    assert (
        0
        <= worst_sample
        < n_samples
    )

    assert (
        result.worst_sample_index
        == worst_sample
    )

    # Home configuration should be singular.
    assert np.isclose(
        home_manipulability,
        0.0,
        atol=1e-6,
    )

    assert np.isclose(
        home_singularity_metric,
        0.0,
        atol=1e-6,
    )

    # The trajectory moves away from q=[0,0,0], so the final sample
    # should have a larger singularity metric than the singular home
    # configuration.
    assert (
        singularity_metric_history[-1]
        > home_singularity_metric
    )

    print(
        "\nAll sanity checks passed."
    )