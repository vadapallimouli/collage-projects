"""
models/data_models.py
======================

Shared Data Contracts
------------------------
Project: Intelligent Robot Motion Planning with Failure Prediction,
         Explainable Risk Analysis and Recoverability

This module is a DATA-MODEL layer only. It defines the dataclasses that
every other module in the project uses to pass structured information
between pipeline stages:

    Start + Goal
        -> Trajectory
        -> KinematicResult
        -> DynamicResult
        -> ObstacleResult
        -> SingularityResult
        -> TrajectoryFeatures
        -> RiskResult (initial)
        -> ExplanationResult (initial)
        -> RecoveryEvaluation (RecoveryAttempt x N -> selected attempt)
        -> Modified Trajectory
        -> re-run Kinematic/Dynamic/Obstacle/Singularity analysis
        -> final TrajectoryFeatures
        -> final RiskResult
        -> final ExplanationResult
        -> PathEvaluation
        -> OptimizationResult

This file intentionally contains NO algorithms: no kinematics, dynamics,
trajectory generation, collision detection, singularity calculation,
risk scoring, explainability logic, recovery logic, optimization, or
plotting/UI code. The dataclasses here are plain containers only.

Numerical constants (robot dimensions, limits, weights, thresholds,
etc.) live exclusively in `config.py` and must NOT be duplicated here.
Other modules are responsible for populating these dataclasses using
values imported from `config.py`.

UNIT CONTRACT
-------------
    Joint angle        -> radians
    Joint velocity     -> radians / second
    Joint acceleration -> radians / second^2
    Torque             -> N*m
    Power              -> Watts
    Energy             -> Joules
    Distance            -> meters
    Time               -> seconds
    Risk               -> 0.0 - 1.0 (estimated, rule-based; see RiskResult)
    Risk percentage    -> 0 - 100 (display only)

ARRAY SHAPE CONTRACT (3-DOF planar robot, N = number of trajectory samples)
----------------------------------------------------------------------------
    time                     -> (N,)
    q, qd, qdd               -> (N, 3)      joint position/velocity/accel.
    end-effector position    -> (N, 2)      planar Cartesian [x, y]
    end-effector velocity    -> (N, 2)
    start_position/goal_position -> (2,)    planar Cartesian [x, y]
    max torque per joint     -> (3,)
    max velocity per joint   -> (3,)
    max acceleration/joint   -> (3,)

Cartesian (task-space) quantities always have 2 components [x, y].
Joint-space quantities always have 3 components [q1, q2, q3] for this
3-DOF robot. Do not confuse the two.

RISK SCORE DISCLAIMER
----------------------
All "risk" values in this project (see RiskResult) are a SIMULATION-
BASED, RULE-BASED estimated risk score derived from measurable robot
conditions. They are NOT a statistically validated or scientifically
proven probability of real-world failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# B1. RobotConfig
# =============================================================================


@dataclass
class RobotConfig:
    """
    Static description of the robot's kinematic/dynamic parameters.

    Produced by: the robot-model module, populated from `config.py`
    (this dataclass does not itself define or duplicate any numeric
    values from config.py).

    Consumed by: kinematic analysis, dynamic analysis, singularity
    analysis, and any module that needs robot parameters.
    """

    name: str
    dof: int
    link_lengths: List[float]
    joint_limits_rad: List[Tuple[float, float]]
    max_joint_velocities: np.ndarray  # shape (dof,)
    max_joint_torques: np.ndarray  # shape (dof,)
    link_masses: List[float]
    link_inertias: List[float]
    gravity: float


# =============================================================================
# B2. Trajectory
# =============================================================================


@dataclass
class Trajectory:
    """
    Represents ONE candidate (or recovered) robot trajectory.

    Produced by: the trajectory-planning module (original trajectories)
    and the recovery module (modified/recovered trajectories).

    Consumed by: kinematic analysis, dynamic analysis, obstacle analysis,
    singularity analysis, visualization, and the recovery module.

    Array shapes (N = number of samples):
        time -> (N,)
        q, qd, qdd -> (N, 3)   joint position/velocity/acceleration
        start_position, goal_position -> (2,)   Cartesian [x, y]

    Note the distinction:
        Cartesian position  -> 2 values [x, y]
        Joint configuration -> 3 values [q1, q2, q3]
    """

    path_id: str
    name: str
    description: str
    time: np.ndarray  # shape (N,)
    q: np.ndarray  # shape (N, 3) joint positions
    qd: np.ndarray  # shape (N, 3) joint velocities
    qdd: np.ndarray  # shape (N, 3) joint accelerations
    start_position: np.ndarray  # shape (2,)
    goal_position: np.ndarray  # shape (2,)
    duration: float
    path_length: float

    # Optional / useful fields
    waypoints: Optional[np.ndarray] = None
    joint_space_length: Optional[float] = None
    is_recovered: bool = False
    parent_path_id: Optional[str] = None
    recovery_action: Optional["RecoveryAction"] = None


# =============================================================================
# B3. KinematicResult
# =============================================================================


@dataclass
class KinematicResult:
    """
    Results of kinematic analysis for ONE trajectory.

    Produced by: the kinematic-analysis module.
    Consumed by: feature extraction, risk prediction, explanation, and
    visualization modules.

    Array shapes (N = number of samples):
        joint_positions, joint_velocities, joint_accelerations -> (N, 3)
        end_effector_positions, end_effector_velocities -> (N, 2)
        max_joint_velocity, max_joint_acceleration -> (3,)  (per joint,
            NOT a single scalar — required so explainability can name
            the specific joint responsible for a problem)
    """

    joint_positions: np.ndarray  # (N, 3)
    joint_velocities: np.ndarray  # (N, 3)
    joint_accelerations: np.ndarray  # (N, 3)
    end_effector_positions: np.ndarray  # (N, 2)
    end_effector_velocities: np.ndarray  # (N, 2)
    max_joint_velocity: np.ndarray  # (3,) per-joint, not a single float
    max_joint_acceleration: np.ndarray  # (3,) per-joint, not a single float
    manipulability: float
    singularity_metric: float


# =============================================================================
# B3a. KinematicAnalysisResult
# =============================================================================


@dataclass
class KinematicAnalysisResult:
    """
    Results of kinematic SAFETY analysis for ONE trajectory: whether the
    kinematic measurements in a `KinematicResult` violate or approach
    the robot's configured kinematic limits (`config.py`:
    `MAX_JOINT_VELOCITIES`, `JOINT_LIMITS_RAD`).

    Produced by: `analysis/kinematic_analysis.py`
    (`KinematicSafetyAnalyzer`).

    Consumed by: feature extraction and the failure-risk prediction
    module. This dataclass is structured evidence only -- it does NOT
    contain a failure probability, risk score, or risk level.

    Array shapes (3-DOF planar robot):
        velocity_violation_count, max_velocity_utilization,
        max_velocity_excess -> (3,)
        joint_limit_violated, joint_limit_violation_count,
        min_joint_limit_margin -> (3,)
        max_joint_acceleration -> (3,)
    """

    velocity_limit_violated: bool
    velocity_violation_count: np.ndarray  # (3,) int
    max_velocity_utilization: np.ndarray  # (3,) float
    max_velocity_excess: np.ndarray  # (3,) float, rad/s
    worst_velocity_joint: int  # 1-based
    worst_velocity_sample: int  # 0-based
    joint_limit_violated: np.ndarray  # (3,) bool
    joint_limit_violation_count: np.ndarray  # (3,) int
    min_joint_limit_margin: np.ndarray  # (3,) float, radians
    minimum_manipulability: float
    minimum_singularity_metric: float
    max_joint_acceleration: np.ndarray  # (3,) float, rad/s^2


# =============================================================================
# B4. DynamicResult
# =============================================================================


@dataclass
class DynamicResult:
    """
    Results of dynamic analysis for ONE trajectory.

    Produced by: the dynamic-analysis module.
    Consumed by: feature extraction, risk prediction, explanation, and
    visualization modules.

    Array shapes (N = number of samples):
        joint_torques -> (N, 3)
        power -> (N,)
        max_joint_torque -> (3,) per-joint, NOT a single float — required
            so explainability can name the specific joint that overloaded
            (e.g. "Joint 2 torque overload") instead of only reporting a
            single maximum value.
    """

    joint_torques: np.ndarray  # (N, 3)
    max_joint_torque: np.ndarray  # (3,) per-joint, not a single float
    power: np.ndarray  # (N,)
    total_energy: float
    average_torque: float

    # Optional per-joint energy breakdown
    energy_per_joint: Optional[np.ndarray] = None  # (3,)


# =============================================================================
# B4a. DynamicAnalysisResult
# =============================================================================


@dataclass
class DynamicAnalysisResult:
    """
    Results of dynamic SAFETY analysis for ONE trajectory: whether the
    dynamic measurements in a `DynamicResult` violate or approach the
    robot's configured torque limits (`config.py`: `MAX_JOINT_TORQUES`).

    Produced by: `analysis/dynamic_analysis.py`
    (`DynamicSafetyAnalyzer`).

    Consumed by: feature extraction and the failure-risk prediction
    module. This dataclass is structured evidence only -- it does NOT
    contain a failure probability, risk score, or risk level, and it
    does NOT contain an energy violation (no configured energy limit
    exists in `config.py`).

    Array shapes (3-DOF planar robot):
        torque_violation_count, max_torque_utilization,
        max_torque_excess -> (3,)
        energy_per_joint -> (3,)
    """

    torque_limit_violated: bool
    torque_violation_count: np.ndarray  # (3,) int
    max_torque_utilization: np.ndarray  # (3,) float
    max_torque_excess: np.ndarray  # (3,) float, N*m
    worst_torque_joint: int  # 1-based
    worst_torque_sample: int  # 0-based
    peak_power: float  # W
    average_torque: float  # N*m
    total_energy: float  # J
    energy_per_joint: np.ndarray  # (3,) J


# =============================================================================
# B5. ObstacleResult
# =============================================================================


@dataclass
class ObstacleResult:
    """
    Results of obstacle-clearance analysis for ONE trajectory, evaluated
    against virtual simulated obstacles (no computer vision is involved).

    Produced by: the obstacle-analysis module.
    Consumed by: feature extraction, risk prediction, and explanation
    modules.

    If no obstacles exist in the scene:
        minimum_clearance   = float("inf")
        collision_detected  = False
        closest_obstacle_id = None
    """

    minimum_clearance: float
    collision_detected: bool
    obstacle_risk_indicator: float

    # Optional fields
    closest_point: Optional[np.ndarray] = None  # (2,) Cartesian [x, y]
    closest_obstacle_id: Optional[str] = None
    obstacle_clearances: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# B6. SingularityResult
# =============================================================================


@dataclass
class SingularityResult:
    """
    Results of singularity-proximity analysis for ONE trajectory.

    Produced by: the singularity-analysis module.
    Consumed by: feature extraction, risk prediction, and explanation
    modules.

    This dataclass only stores results; the manipulability/Jacobian-based
    metric itself is calculated elsewhere.
    """

    minimum_manipulability: float
    worst_condition_metric: float
    risk_level: str  # e.g. "LOW", "MEDIUM", "HIGH"
    near_singularity: bool

    # Optional: index into the trajectory samples where the worst
    # singularity condition occurs, useful for visualization/explanation.
    worst_sample_index: Optional[int] = None


# =============================================================================
# B7. TrajectoryFeatures
# =============================================================================


@dataclass
class TrajectoryFeatures:
    """
    Scalar and per-joint measurements extracted from a trajectory's
    kinematic/dynamic/obstacle/singularity analysis results. This is the
    feature set consumed directly by the failure-risk engine.

    Produced by: the feature-extraction module (from KinematicResult,
    DynamicResult, ObstacleResult, SingularityResult).

    Consumed by: the failure-risk prediction module and (via normalized
    fields, populated later) the Explainable AI module.
    """

    max_torque: float
    max_velocity: float
    max_acceleration: float
    total_energy: float
    execution_time: float
    path_length: float
    min_obstacle_clearance: float
    singularity_metric: float
    manipulability: float

    # Optional
    joint_space_length: Optional[float] = None

    # Per-joint values, required for detailed explanations.
    max_torque_per_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # (3,)
    max_velocity_per_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # (3,)
    max_acceleration_per_joint: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # (3,)

    # Optional normalized features. These are NOT calculated by this
    # dataclass — they are populated later by the risk-prediction module.
    normalized_max_torque: Optional[float] = None
    normalized_max_velocity: Optional[float] = None
    normalized_max_acceleration: Optional[float] = None
    normalized_total_energy: Optional[float] = None
    normalized_min_obstacle_clearance: Optional[float] = None
    normalized_singularity_metric: Optional[float] = None


# =============================================================================
# B8. RiskResult
# =============================================================================


@dataclass
class RiskResult:
    """
    The estimated failure-risk result for ONE trajectory (original or
    post-recovery).

    Produced by: the failure-risk prediction module.
    Consumed by: the Explainable AI module, the recovery module, and the
    optimization module.

    IMPORTANT DISCLAIMER: `overall_risk`, `risk_percentage`, and every
    per-category risk value stored here represent an ESTIMATED,
    SIMULATION-BASED, RULE-BASED risk score. They are NOT a statistically
    validated or scientifically proven probability of real-world failure.

    Each individual risk category and `overall_risk` are on a 0.0-1.0
    scale, where 0.0 = no significant risk and 1.0 = very high risk.
    `risk_percentage` is purely a display convenience equal to
    `overall_risk * 100`.
    """

    overall_risk: float  # 0.0 - 1.0 (estimated, not a statistical probability)
    risk_percentage: float  # overall_risk * 100, for display only
    risk_level: str  # e.g. "LOW", "MEDIUM", "HIGH"

    torque_risk: float
    velocity_risk: float
    energy_risk: float
    obstacle_risk: float
    singularity_risk: float

    main_contributors: List[str] = field(default_factory=list)
    detected_problems: List[str] = field(default_factory=list)
    is_safe: bool = False

    # Optional structured breakdown, e.g.
    #   {"torque": 0.82, "velocity": 0.41, "energy": 0.32,
    #    "obstacle": 0.71, "singularity": 0.08}
    # Not calculated by this dataclass — populated by the risk-prediction
    # module so the Explainable AI module can rank failure reasons.
    risk_contributions: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# B9. ExplanationResult
# =============================================================================


@dataclass
class ExplanationResult:
    """
    The Explainable AI output describing WHY a trajectory received its
    estimated risk score.

    Produced by: the Explainable AI module (from RiskResult and
    TrajectoryFeatures).

    Consumed by: the recovery module (to target the right corrective
    action) and the visualization/UI layer.
    """

    summary: str
    detailed_explanation: str
    main_reason: str

    # e.g. {"torque": "...", "obstacle": "...", "velocity": "..."}
    factor_explanations: Dict[str, str] = field(default_factory=dict)

    recommended_action: str = ""

    # Optional structured evidence used by the explanation engine, e.g.
    #   {"joint": 2, "actual_torque": 12.4, "torque_limit": 10.0,
    #    "minimum_clearance": 0.038}
    evidence: Dict[str, object] = field(default_factory=dict)


# =============================================================================
# B10. RecoveryAction
# =============================================================================


@dataclass
class RecoveryAction:
    """
    Describes ONE candidate corrective action that could be applied to a
    risky trajectory. This dataclass only describes the action — it does
    not execute it.

    Produced by: the recovery module, when generating candidate actions
    (using option values from `config.py`, e.g. SPEED_REDUCTION_OPTIONS).

    Consumed by: the recovery module (to apply the action) and
    RecoveryAttempt/RecoveryEvaluation (to record what was tried).

    Supported action_type values:
        REDUCE_SPEED
        INCREASE_CLEARANCE
        CHANGE_JOINT_CONFIGURATION
        GENERATE_ALTERNATIVE_TRAJECTORY

    Example: action_type="REDUCE_SPEED", parameter=0.20 means "reduce
    trajectory speed by 20%".
    """

    action_id: str
    action_type: str
    description: str
    parameter: float
    target_failure: str


# =============================================================================
# B11. RecoveryAttempt
# =============================================================================


@dataclass
class RecoveryAttempt:
    """
    Represents ONE attempted recovery action and its measured effect on
    risk.

    Produced by: the recovery module, once per candidate RecoveryAction
    that is applied and re-evaluated.

    Consumed by: RecoveryEvaluation (to compare all attempts and select
    the best one).

    Example sequence for one path:
        Attempt 1: Reduce speed 10%      -> risk 68% -> 39%
        Attempt 2: Reduce speed 20%      -> risk 68% -> 14%
        Attempt 3: Alternative trajectory -> risk 68% -> 22%
    """

    attempt_id: str
    action: RecoveryAction
    original_risk: float
    new_risk: float
    risk_reduction: float
    successful: bool
    reason: str

    modified_trajectory: Optional[Trajectory] = None
    new_risk_result: Optional[RiskResult] = None


# =============================================================================
# B12. RecoveryEvaluation
# =============================================================================


@dataclass
class RecoveryEvaluation:
    """
    The COMPLETE recoverability analysis for one original (risky) path:
    every candidate recovery attempt that was tried, and which one (if
    any) was selected as the best.

    Produced by: the recovery module.
    Consumed by: PathEvaluation, and the optimization module.

    Supported recoverability_level values:
        HIGH
        MEDIUM
        LOW
        NOT_RECOVERABLE

    This dataclass only stores the result; it does not decide which
    recoverability level applies.
    """

    original_risk: float
    candidate_attempts: List[RecoveryAttempt] = field(default_factory=list)
    selected_attempt: Optional[RecoveryAttempt] = None
    successful: bool = False
    recoverability_level: str = "NOT_RECOVERABLE"
    reason: str = ""


# =============================================================================
# B13. PathEvaluation
# =============================================================================


@dataclass
class PathEvaluation:
    """
    The CENTRAL DATA OBJECT for one candidate path: holds the complete
    original analysis, the recoverability evaluation, and the complete
    post-recovery ("final") re-analysis.

    Produced by: assembled incrementally across the whole pipeline —
    populated first by the original analysis modules (kinematics,
    dynamics, obstacle, singularity, features, risk, explanation), then
    by the recovery module, then by the same analysis modules run again
    on the recovered trajectory.

    Consumed by: the optimization module, which compares the FINAL
    versions of all paths (not the original ones) to select the safest
    and most reliable trajectory.

    All `final_*` fields are Optional because a PathEvaluation may exist
    before recovery/re-analysis has completed (e.g. a path that was
    already safe and never needed recovery).
    """

    # ---- Original analysis ----
    trajectory: Trajectory
    kinematic_result: KinematicResult
    dynamic_result: DynamicResult
    obstacle_result: ObstacleResult
    singularity_result: SingularityResult
    features: TrajectoryFeatures
    initial_risk: RiskResult
    initial_explanation: ExplanationResult

    # ---- Recovery information ----
    recovery_evaluation: Optional[RecoveryEvaluation] = None

    # ---- Final post-recovery analysis (re-run after recovery) ----
    final_trajectory: Optional[Trajectory] = None
    final_kinematic_result: Optional[KinematicResult] = None
    final_dynamic_result: Optional[DynamicResult] = None
    final_obstacle_result: Optional[ObstacleResult] = None
    final_singularity_result: Optional[SingularityResult] = None
    final_features: Optional[TrajectoryFeatures] = None
    final_risk: Optional[RiskResult] = None
    final_explanation: Optional[ExplanationResult] = None

    # ---- Summary fields ----
    final_score: Optional[float] = None
    status: str = "PENDING"  # PENDING | SAFE | RECOVERED | REJECTED


# =============================================================================
# B14. OptimizationResult
# =============================================================================


@dataclass
class OptimizationResult:
    """
    The final output of the optimization stage: the selected path (using
    its final, post-recovery analysis) along with all candidate paths
    ranked for comparison.

    Produced by: the optimization module, which compares the FINAL
    versions of all PathEvaluations (not their original/pre-recovery
    versions).

    Consumed by: the visualization/UI layer.
    """

    selected_path_id: str
    selected_path: PathEvaluation
    ranked_paths: List[PathEvaluation] = field(default_factory=list)
    selection_reason: str = ""