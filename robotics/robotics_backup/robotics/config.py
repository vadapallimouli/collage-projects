"""
config.py
=========

Single Source of Truth — Project Configuration
------------------------------------------------
Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

This module contains ONLY project-wide constants, configuration values,
and lightweight configuration validation.

It intentionally contains:
    - NO data models
    - NO robotics algorithms
    - NO trajectory planning algorithms
    - NO failure prediction logic
    - NO Explainable AI logic
    - NO recovery logic
    - NO optimization logic
    - NO visualization / UI code

Future modules MUST import values from this file instead of hard-coding
or duplicating them elsewhere.

Example:

    from config import MAX_JOINT_TORQUES

UNIT CONVENTION
---------------

    Angle           -> radians
    Angular speed   -> radians / second
    Angular accel.  -> radians / second^2
    Cartesian dist. -> meters (m)
    Torque          -> Newton-meters (N*m)
    Energy          -> Joules (J)
    Power           -> Watts (W)
    Time            -> seconds (s)

Degrees are used ONLY at the configuration/presentation boundary.
All internal robot calculations use radians.

RISK SCORE DISCLAIMER
---------------------

Values referred to as "failure risk" throughout this project are
SIMULATION-BASED, RULE-BASED estimated risk scores derived from
measurable robot conditions.

They are:

    - NOT statistically validated probabilities
    - NOT learned probabilities from a trained ML model
    - NOT guarantees of real-world robot failure
    - NOT suitable for controlling an industrial robot

Internally, estimated failure risk is represented on a continuous
scale:

    0.0 -> minimal estimated risk
    1.0 -> very high estimated risk

The UI may display:

    0.68 -> 68%

but this is only a presentation conversion.

ENERGY NOTE
-----------

The project computes mechanical energy because energy is useful for:

    - trajectory comparison
    - explainability
    - optimization
    - performance analysis

However, the current project configuration does NOT define a maximum
allowable energy limit.

Therefore energy is NOT treated as an independent primary failure-risk
category.

The failure-risk engine instead uses:

    1. torque overload
    2. excessive velocity
    3. joint-limit violation
    4. insufficient obstacle clearance
    5. singularity proximity

This keeps the risk model tied to conditions that have explicit
configured limits or thresholds.
"""

from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# A1. ROBOT CONFIGURATION
# =============================================================================

# 3-DOF planar revolute robot arm operating primarily in the X-Y plane.

ROBOT_NAME: str = "Planar3R"

ROBOT_DOF: int = 3


# Link lengths in meters.
#
# Ordered as:
#
#     [Link 1, Link 2, Link 3]
#
# These values are the single source of truth for robot geometry.
# Do not duplicate these values in other modules.

LINK_LENGTHS: np.ndarray = np.array(
    [1.0, 0.8, 0.6],
    dtype=float,
)


# =============================================================================
# A2. JOINT LIMITS
# =============================================================================

# Joint limits are specified here in degrees only for readability.
#
# They are immediately converted to radians so that every other module
# can work internally using radians.

_JOINT_LIMITS_DEG: List[Tuple[float, float]] = [
    (-170.0, 170.0),   # Joint 1
    (-150.0, 150.0),   # Joint 2
    (-160.0, 160.0),   # Joint 3
]


# Internal representation used throughout the project.
#
# Format:
#
#     [
#         (joint_1_min, joint_1_max),
#         (joint_2_min, joint_2_max),
#         (joint_3_min, joint_3_max),
#     ]
#
# Units: radians.

JOINT_LIMITS_RAD: List[Tuple[float, float]] = [
    (np.deg2rad(lower), np.deg2rad(upper))
    for lower, upper in _JOINT_LIMITS_DEG
]


# =============================================================================
# A3. JOINT VELOCITY LIMITS
# =============================================================================

# Maximum allowed angular velocity for each joint.
#
# Units: rad/s
#
# Ordered:
#
#     [Joint 1, Joint 2, Joint 3]
#
# These are per-joint limits rather than one global limit so that the
# analysis system can identify the exact joint causing excessive velocity.

MAX_JOINT_VELOCITIES: np.ndarray = np.array(
    [2.5, 2.0, 3.0],
    dtype=float,
)


# =============================================================================
# A4. JOINT TORQUE LIMITS
# =============================================================================

# Maximum allowed absolute torque for each joint.
#
# Units: N*m
#
# IMPORTANT:
# These are educational simulation assumptions.
# They are NOT manufacturer specifications for a real robot.

MAX_JOINT_TORQUES: np.ndarray = np.array(
    [50.0, 35.0, 15.0],
    dtype=float,
)


# =============================================================================
# A5. ROBOT PHYSICAL PARAMETERS
# =============================================================================

# Link masses in kilograms.
#
# Ordered:
#
#     [Link 1, Link 2, Link 3]

LINK_MASSES: np.ndarray = np.array(
    [3.5, 2.5, 1.5],
    dtype=float,
)


# Fraction of each link length at which the center of mass is located.
#
# 0.5 means the center of mass is assumed to be at the midpoint of
# each uniform/slender link.

LINK_COM_RATIO: float = 0.5


# Link inertias.
#
# Slender-rod approximation:
#
#     I = m * L^2 / 12
#
# Calculated directly from LINK_MASSES and LINK_LENGTHS so the
# numerical values do not need to be manually duplicated.

LINK_INERTIAS: np.ndarray = (
    LINK_MASSES * LINK_LENGTHS**2
) / 12.0


# Gravitational acceleration.
#
# Units: m/s^2

GRAVITY: float = 9.81


# =============================================================================
# A6. TRAJECTORY CONFIGURATION
# =============================================================================

# Number of candidate trajectories generated by the planner.

DEFAULT_NUM_PATHS: int = 4


# Default duration of each generated trajectory.
#
# Units: seconds

DEFAULT_TRAJECTORY_DURATION: float = 4.0


# Number of samples along each trajectory.

DEFAULT_NUM_SAMPLES: int = 100


# =============================================================================
# A7. FAILURE-RISK CATEGORIES AND WEIGHTS
# =============================================================================

"""
Primary failure-risk categories
--------------------------------

The failure-risk engine uses exactly five primary categories:

    torque_overload
        Excessive joint torque relative to configured torque limits.

    excessive_velocity
        Joint velocity exceeds configured velocity limits.

    joint_limit_violation
        Robot joint position exceeds its configured allowable range.

    insufficient_clearance
        Robot comes too close to or collides with a virtual obstacle.

    singularity_proximity
        Robot approaches a kinematic singular configuration.

These weights represent the relative contribution of each category
to the overall rule-based estimated failure risk.

The values MUST sum to exactly 1.0 within the configured validation
tolerance.

IMPORTANT ENERGY DESIGN DECISION
--------------------------------

Energy is intentionally NOT included as a primary failure-risk
category.

The project calculates mechanical energy, but config.py does not
define a maximum allowable energy or energy-failure threshold.

Therefore:

    energy -> supporting information

rather than:

    energy -> independent failure-risk category

Energy can still be used later by:

    - Explainable AI
    - optimization
    - trajectory comparison
    - performance analysis
"""

RISK_WEIGHTS: Dict[str, float] = {
    "torque_overload": 0.30,
    "excessive_velocity": 0.20,
    "joint_limit_violation": 0.15,
    "insufficient_clearance": 0.20,
    "singularity_proximity": 0.15,
}


# =============================================================================
# A8. RISK LEVEL THRESHOLDS
# =============================================================================

"""
Overall risk interpretation.

The prediction module converts the final weighted risk score into
a human-readable risk level using these thresholds.

    LOW:
        risk < 0.35

    MEDIUM:
        0.35 <= risk < 0.65

    HIGH:
        risk >= 0.65
"""

RISK_LEVEL_THRESHOLDS: Dict[str, float] = {
    "LOW_MAX": 0.35,
    "MEDIUM_MAX": 0.65,
}


# =============================================================================
# A9. OBSTACLE CONFIGURATION
# =============================================================================

"""
Virtual obstacle configuration.

The project uses simple simulated obstacles.

No computer-vision obstacle detection is performed.

Obstacle geometry and collision detection are implemented in:

    analysis/obstacle_analysis.py
"""


# Minimum distance that should be maintained from an obstacle boundary.
#
# Units: meters.

MIN_SAFE_CLEARANCE: float = 0.10


# Distance at which the robot is considered to be entering a warning
# region.
#
# Units: meters.

OBSTACLE_WARNING_DISTANCE: float = 0.25


# Distance at or below which the robot is considered to be colliding
# with the obstacle.
#
# Units: meters.

COLLISION_DISTANCE: float = 0.02


# =============================================================================
# A10. SINGULARITY CONFIGURATION
# =============================================================================

"""
Jacobian-based singularity interpretation thresholds.

The singularity-analysis module calculates the actual metric.

This file ONLY stores the interpretation thresholds.

Higher manipulability / larger smallest singular value generally
indicates that the robot is farther from a singular configuration.

Threshold interpretation:

    metric < HIGH_RISK_BELOW
        -> HIGH

    HIGH_RISK_BELOW <= metric < LOW_RISK_AT_OR_ABOVE
        -> MEDIUM

    metric >= LOW_RISK_AT_OR_ABOVE
        -> LOW
"""

SINGULARITY_THRESHOLDS: Dict[str, float] = {
    "HIGH_RISK_BELOW": 0.05,
    "LOW_RISK_AT_OR_ABOVE": 0.15,
}


# =============================================================================
# A11. RECOVERABILITY CONFIGURATION
# =============================================================================

"""
Candidate recovery actions.

These values do NOT execute recovery themselves.

They are candidate parameters used by the recovery module.

The recovery module:

    1. identifies the main failure reason
    2. selects an appropriate corrective action
    3. modifies the trajectory
    4. re-runs the complete analysis
    5. calculates the new failure risk
    6. determines whether recovery succeeded
"""


# Maximum number of recovery attempts for one trajectory.

MAX_RECOVERY_ATTEMPTS: int = 3


# Candidate speed reductions.
#
# Example:
#
#     0.10 -> reduce trajectory speed by 10%
#     0.20 -> reduce trajectory speed by 20%
#     0.30 -> reduce trajectory speed by 30%

SPEED_REDUCTION_OPTIONS: List[float] = [
    0.10,
    0.20,
    0.30,
]


# Candidate increases in obstacle clearance.
#
# Units: meters.

CLEARANCE_ADJUSTMENT_OPTIONS: List[float] = [
    0.05,
    0.10,
    0.15,
]


# Candidate joint-configuration adjustments.
#
# Units: radians.

JOINT_CONFIGURATION_ADJUSTMENT_OPTIONS: List[float] = [
    0.05,
    0.10,
    0.20,
]


# =============================================================================
# A12. OPTIMIZATION CONFIGURATION
# =============================================================================

"""
Weights used by the final path-selection / optimization stage.

The optimizer compares FINAL paths after recovery attempts have been
performed.

Safety and reliability receive the highest weights.

Execution time receives the smallest weight so that the optimizer
does not simply select the fastest path.
"""

OPTIMIZATION_WEIGHTS: Dict[str, float] = {
    "safety_reliability": 0.35,
    "failure_risk": 0.30,
    "energy": 0.15,
    "path_smoothness": 0.12,
    "execution_time": 0.08,
}


# =============================================================================
# A13. APPLICATION CONFIGURATION
# =============================================================================

PROJECT_NAME: str = (
    "Intelligent Robot Motion Planning with Failure Prediction, "
    "Explainable Risk Analysis and Recoverability"
)


# Reuse ROBOT_NAME rather than duplicating the robot name.

APP_ROBOT_NAME: str = ROBOT_NAME


# Default visualization/simulation playback speed.
#
#     1.0 -> real-time
#     2.0 -> twice real-time
#     0.5 -> half-speed

DEFAULT_SIMULATION_SPEED: float = 1.0


# Default plotting line width.

PLOT_LINE_WIDTH: float = 2.0


# Risk disclaimer shown by the application/UI.

RISK_DISCLAIMER: str = (
    "The displayed failure risk is a simulation-based, rule-based "
    "estimated score derived from measurable robot conditions. It is "
    "NOT a statistically validated or scientifically proven probability "
    "of real-world failure."
)


# =============================================================================
# A14. CONFIGURATION VALIDATION
# =============================================================================

"""
Lightweight fail-fast validation.

The purpose of these checks is to catch configuration mistakes early.

The configuration is NOT silently modified or normalized.

Invalid values raise ValueError.
"""


_WEIGHT_SUM_TOLERANCE: float = 1e-6


def _validate_weight_dict_sums_to_one(
    weights: Dict[str, float],
    name: str,
) -> None:
    """
    Raise ValueError if the values in a weight dictionary do not sum
    to approximately 1.0.

    Args:
        weights:
            Dictionary of category weights.

        name:
            Human-readable configuration name used in the error message.

    Raises:
        ValueError:
            If the weights do not sum to 1.0 within the configured
            tolerance.
    """

    total = sum(float(value) for value in weights.values())

    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"{name} must sum to 1.0 "
            f"(within {_WEIGHT_SUM_TOLERANCE}), "
            f"but got {total}. "
            f"Values: {weights}"
        )


def _validate_robot_dimensions() -> None:
    """
    Validate that all per-joint and per-link configuration arrays
    contain exactly ROBOT_DOF entries.
    """

    expected_lengths = {
        "LINK_LENGTHS": len(LINK_LENGTHS),
        "JOINT_LIMITS_RAD": len(JOINT_LIMITS_RAD),
        "MAX_JOINT_VELOCITIES": len(MAX_JOINT_VELOCITIES),
        "MAX_JOINT_TORQUES": len(MAX_JOINT_TORQUES),
        "LINK_MASSES": len(LINK_MASSES),
        "LINK_INERTIAS": len(LINK_INERTIAS),
    }

    for config_name, length in expected_lengths.items():
        if length != ROBOT_DOF:
            raise ValueError(
                f"{config_name} has length {length}, "
                f"expected ROBOT_DOF ({ROBOT_DOF})."
            )


def _validate_physical_values() -> None:
    """
    Validate that physical robot parameters and configured limits
    contain physically meaningful positive values.
    """

    # ------------------------------------------------------------------
    # Link lengths
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(LINK_LENGTHS)):
        raise ValueError(
            f"LINK_LENGTHS contains NaN or infinite values: "
            f"{LINK_LENGTHS}"
        )

    if np.any(LINK_LENGTHS <= 0):
        raise ValueError(
            f"All LINK_LENGTHS must be > 0. "
            f"Got: {LINK_LENGTHS}"
        )

    # ------------------------------------------------------------------
    # Link masses
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(LINK_MASSES)):
        raise ValueError(
            f"LINK_MASSES contains NaN or infinite values: "
            f"{LINK_MASSES}"
        )

    if np.any(LINK_MASSES <= 0):
        raise ValueError(
            f"All LINK_MASSES must be > 0. "
            f"Got: {LINK_MASSES}"
        )

    # ------------------------------------------------------------------
    # Joint torque limits
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(MAX_JOINT_TORQUES)):
        raise ValueError(
            f"MAX_JOINT_TORQUES contains NaN or infinite values: "
            f"{MAX_JOINT_TORQUES}"
        )

    if np.any(MAX_JOINT_TORQUES <= 0):
        raise ValueError(
            f"All MAX_JOINT_TORQUES must be > 0. "
            f"Got: {MAX_JOINT_TORQUES}"
        )

    # ------------------------------------------------------------------
    # Joint velocity limits
    # ------------------------------------------------------------------

    if not np.all(np.isfinite(MAX_JOINT_VELOCITIES)):
        raise ValueError(
            f"MAX_JOINT_VELOCITIES contains NaN or infinite values: "
            f"{MAX_JOINT_VELOCITIES}"
        )

    if np.any(MAX_JOINT_VELOCITIES <= 0):
        raise ValueError(
            f"All MAX_JOINT_VELOCITIES must be > 0. "
            f"Got: {MAX_JOINT_VELOCITIES}"
        )

    # ------------------------------------------------------------------
    # Gravity
    # ------------------------------------------------------------------

    if not np.isfinite(GRAVITY):
        raise ValueError(
            f"GRAVITY must be finite. Got: {GRAVITY}"
        )

    if GRAVITY <= 0:
        raise ValueError(
            f"GRAVITY must be > 0. Got: {GRAVITY}"
        )

    # ------------------------------------------------------------------
    # Center-of-mass ratio
    # ------------------------------------------------------------------

    if not np.isfinite(LINK_COM_RATIO):
        raise ValueError(
            f"LINK_COM_RATIO must be finite. "
            f"Got: {LINK_COM_RATIO}"
        )

    if not 0.0 <= LINK_COM_RATIO <= 1.0:
        raise ValueError(
            f"LINK_COM_RATIO must be between 0 and 1. "
            f"Got: {LINK_COM_RATIO}"
        )


def _validate_joint_limits() -> None:
    """
    Validate that every joint has a valid lower and upper limit.
    """

    if len(JOINT_LIMITS_RAD) != ROBOT_DOF:
        raise ValueError(
            "JOINT_LIMITS_RAD length does not match ROBOT_DOF."
        )

    for joint_index, (lower, upper) in enumerate(JOINT_LIMITS_RAD):
        lower = float(lower)
        upper = float(upper)

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(
                f"Joint {joint_index + 1} limits must be finite. "
                f"Got ({lower}, {upper})."
            )

        if lower >= upper:
            raise ValueError(
                f"Joint {joint_index + 1} lower limit must be "
                f"less than its upper limit. "
                f"Got ({lower}, {upper})."
            )


def _validate_obstacle_configuration() -> None:
    """
    Validate obstacle-related configuration values.
    """

    obstacle_values = {
        "MIN_SAFE_CLEARANCE": MIN_SAFE_CLEARANCE,
        "OBSTACLE_WARNING_DISTANCE": OBSTACLE_WARNING_DISTANCE,
        "COLLISION_DISTANCE": COLLISION_DISTANCE,
    }

    for name, value in obstacle_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite. Got {value}."
            )

        if value < 0.0:
            raise ValueError(
                f"{name} must be >= 0. Got {value}."
            )

    if COLLISION_DISTANCE > MIN_SAFE_CLEARANCE:
        raise ValueError(
            "COLLISION_DISTANCE must be <= MIN_SAFE_CLEARANCE. "
            f"Got COLLISION_DISTANCE={COLLISION_DISTANCE}, "
            f"MIN_SAFE_CLEARANCE={MIN_SAFE_CLEARANCE}."
        )

    if MIN_SAFE_CLEARANCE > OBSTACLE_WARNING_DISTANCE:
        raise ValueError(
            "MIN_SAFE_CLEARANCE must be <= "
            "OBSTACLE_WARNING_DISTANCE. "
            f"Got MIN_SAFE_CLEARANCE={MIN_SAFE_CLEARANCE}, "
            f"OBSTACLE_WARNING_DISTANCE={OBSTACLE_WARNING_DISTANCE}."
        )


def _validate_singularity_configuration() -> None:
    """
    Validate singularity threshold ordering.
    """

    required_keys = {
        "HIGH_RISK_BELOW",
        "LOW_RISK_AT_OR_ABOVE",
    }

    missing_keys = required_keys.difference(SINGULARITY_THRESHOLDS)

    if missing_keys:
        raise ValueError(
            "SINGULARITY_THRESHOLDS is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    high_risk_below = float(
        SINGULARITY_THRESHOLDS["HIGH_RISK_BELOW"]
    )

    low_risk_at_or_above = float(
        SINGULARITY_THRESHOLDS["LOW_RISK_AT_OR_ABOVE"]
    )

    if not np.isfinite(high_risk_below):
        raise ValueError(
            "HIGH_RISK_BELOW must be finite."
        )

    if not np.isfinite(low_risk_at_or_above):
        raise ValueError(
            "LOW_RISK_AT_OR_ABOVE must be finite."
        )

    if high_risk_below < 0.0:
        raise ValueError(
            "HIGH_RISK_BELOW must be >= 0."
        )

    if low_risk_at_or_above < 0.0:
        raise ValueError(
            "LOW_RISK_AT_OR_ABOVE must be >= 0."
        )

    if high_risk_below >= low_risk_at_or_above:
        raise ValueError(
            "SINGULARITY_THRESHOLDS must satisfy "
            "HIGH_RISK_BELOW < LOW_RISK_AT_OR_ABOVE. "
            f"Got HIGH_RISK_BELOW={high_risk_below}, "
            f"LOW_RISK_AT_OR_ABOVE={low_risk_at_or_above}."
        )


def _validate_risk_level_thresholds() -> None:
    """
    Validate overall risk-level thresholds.

    Expected relationship:

        0 <= LOW_MAX < MEDIUM_MAX <= 1
    """

    required_keys = {
        "LOW_MAX",
        "MEDIUM_MAX",
    }

    missing_keys = required_keys.difference(RISK_LEVEL_THRESHOLDS)

    if missing_keys:
        raise ValueError(
            "RISK_LEVEL_THRESHOLDS is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    low_max = float(RISK_LEVEL_THRESHOLDS["LOW_MAX"])
    medium_max = float(RISK_LEVEL_THRESHOLDS["MEDIUM_MAX"])

    if not np.isfinite(low_max) or not np.isfinite(medium_max):
        raise ValueError(
            "RISK_LEVEL_THRESHOLDS values must be finite."
        )

    if not 0.0 <= low_max < medium_max <= 1.0:
        raise ValueError(
            "RISK_LEVEL_THRESHOLDS must satisfy "
            "0 <= LOW_MAX < MEDIUM_MAX <= 1. "
            f"Got LOW_MAX={low_max}, "
            f"MEDIUM_MAX={medium_max}."
        )


def _validate_recovery_configuration() -> None:
    """
    Validate recovery configuration values.
    """

    if MAX_RECOVERY_ATTEMPTS < 1:
        raise ValueError(
            "MAX_RECOVERY_ATTEMPTS must be >= 1."
        )

    if not SPEED_REDUCTION_OPTIONS:
        raise ValueError(
            "SPEED_REDUCTION_OPTIONS cannot be empty."
        )

    if not CLEARANCE_ADJUSTMENT_OPTIONS:
        raise ValueError(
            "CLEARANCE_ADJUSTMENT_OPTIONS cannot be empty."
        )

    if not JOINT_CONFIGURATION_ADJUSTMENT_OPTIONS:
        raise ValueError(
            "JOINT_CONFIGURATION_ADJUSTMENT_OPTIONS cannot be empty."
        )

    for value in SPEED_REDUCTION_OPTIONS:
        if not np.isfinite(value):
            raise ValueError(
                f"Speed reduction option must be finite. Got {value}."
            )

        if not 0.0 < value < 1.0:
            raise ValueError(
                "Speed reduction options must be between 0 and 1 "
                "(exclusive). "
                f"Got {value}."
            )

    for value in CLEARANCE_ADJUSTMENT_OPTIONS:
        if not np.isfinite(value):
            raise ValueError(
                f"Clearance adjustment must be finite. Got {value}."
            )

        if value <= 0.0:
            raise ValueError(
                "Clearance adjustment options must be > 0. "
                f"Got {value}."
            )

    for value in JOINT_CONFIGURATION_ADJUSTMENT_OPTIONS:
        if not np.isfinite(value):
            raise ValueError(
                "Joint configuration adjustment must be finite. "
                f"Got {value}."
            )

        if value <= 0.0:
            raise ValueError(
                "Joint configuration adjustment options must be > 0. "
                f"Got {value}."
            )


def _validate_trajectory_configuration() -> None:
    """
    Validate default trajectory-generation configuration.
    """

    if DEFAULT_NUM_PATHS < 1:
        raise ValueError(
            "DEFAULT_NUM_PATHS must be >= 1."
        )

    if DEFAULT_TRAJECTORY_DURATION <= 0.0:
        raise ValueError(
            "DEFAULT_TRAJECTORY_DURATION must be > 0."
        )

    if DEFAULT_NUM_SAMPLES < 2:
        raise ValueError(
            "DEFAULT_NUM_SAMPLES must be >= 2."
        )

    if not np.isfinite(DEFAULT_TRAJECTORY_DURATION):
        raise ValueError(
            "DEFAULT_TRAJECTORY_DURATION must be finite."
        )


def _validate_optimization_configuration() -> None:
    """
    Validate optimization weights.
    """

    required_keys = {
        "safety_reliability",
        "failure_risk",
        "energy",
        "path_smoothness",
        "execution_time",
    }

    missing_keys = required_keys.difference(OPTIMIZATION_WEIGHTS)

    if missing_keys:
        raise ValueError(
            "OPTIMIZATION_WEIGHTS is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    for name, value in OPTIMIZATION_WEIGHTS.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Optimization weight '{name}' must be finite. "
                f"Got {value}."
            )

        if value < 0.0:
            raise ValueError(
                f"Optimization weight '{name}' must be >= 0. "
                f"Got {value}."
            )


def _validate_risk_configuration() -> None:
    """
    Validate the primary failure-risk configuration.

    This explicitly checks that the intended five categories are
    present. This prevents a future accidental rename such as
    'joint_limit' from silently breaking the prediction module.
    """

    required_categories = {
        "torque_overload",
        "excessive_velocity",
        "joint_limit_violation",
        "insufficient_clearance",
        "singularity_proximity",
    }

    actual_categories = set(RISK_WEIGHTS.keys())

    missing_categories = required_categories - actual_categories

    if missing_categories:
        raise ValueError(
            "RISK_WEIGHTS is missing required failure-risk "
            f"categories: {sorted(missing_categories)}"
        )

    unexpected_categories = actual_categories - required_categories

    if unexpected_categories:
        raise ValueError(
            "RISK_WEIGHTS contains unexpected failure-risk "
            f"categories: {sorted(unexpected_categories)}"
        )

    for category, weight in RISK_WEIGHTS.items():
        if not np.isfinite(weight):
            raise ValueError(
                f"Risk weight '{category}' must be finite. "
                f"Got {weight}."
            )

        if weight < 0.0:
            raise ValueError(
                f"Risk weight '{category}' must be >= 0. "
                f"Got {weight}."
            )


def validate_config() -> None:
    """
    Run all configuration validation checks.

    This function is called automatically when config.py is imported.

    Raises:
        ValueError:
            If any configuration value is invalid or inconsistent.
    """

    # ------------------------------------------------------------------
    # Risk configuration
    # ------------------------------------------------------------------

    _validate_risk_configuration()

    _validate_weight_dict_sums_to_one(
        RISK_WEIGHTS,
        "RISK_WEIGHTS",
    )

    # ------------------------------------------------------------------
    # Optimization configuration
    # ------------------------------------------------------------------

    _validate_optimization_configuration()

    _validate_weight_dict_sums_to_one(
        OPTIMIZATION_WEIGHTS,
        "OPTIMIZATION_WEIGHTS",
    )

    # ------------------------------------------------------------------
    # Robot configuration
    # ------------------------------------------------------------------

    _validate_robot_dimensions()

    _validate_physical_values()

    _validate_joint_limits()

    # ------------------------------------------------------------------
    # Trajectory configuration
    # ------------------------------------------------------------------

    _validate_trajectory_configuration()

    # ------------------------------------------------------------------
    # Obstacle configuration
    # ------------------------------------------------------------------

    _validate_obstacle_configuration()

    # ------------------------------------------------------------------
    # Singularity configuration
    # ------------------------------------------------------------------

    _validate_singularity_configuration()

    # ------------------------------------------------------------------
    # Risk-level configuration
    # ------------------------------------------------------------------

    _validate_risk_level_thresholds()

    # ------------------------------------------------------------------
    # Recovery configuration
    # ------------------------------------------------------------------

    _validate_recovery_configuration()


# =============================================================================
# AUTOMATIC VALIDATION
# =============================================================================

# Validate immediately on import.
#
# This means that if config.py contains an invalid configuration,
# the project fails early rather than allowing an invalid configuration
# to propagate through several downstream modules.

validate_config()