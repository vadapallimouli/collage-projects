"""
explainability/explanation_engine.py
=====================================

Explainable AI Module
----------------------
Project:
    Intelligent Robot Motion Planning with Failure Prediction,
    Explainable Risk Analysis and Recoverability

This module answers exactly one question:

    "What exactly is wrong with this trajectory, and why?"

It takes the ALREADY-COMPUTED evidence produced upstream:

    KinematicAnalysisResult
    DynamicAnalysisResult
    ObstacleResult
    SingularityResult
    RiskResult

and turns it into a clear, detailed, human-readable explanation.

This module intentionally does NOT:

    - calculate robot kinematics, dynamics, Jacobians, singular values,
      torque, power, or energy
    - calculate obstacle/collision geometry
    - calculate failure probability or change risk weights
    - re-rank main contributors (uses RiskResult.main_contributors as-is)
    - modify trajectories, perform recovery, or optimize paths
    - perform machine learning
    - call an LLM or external API

It ONLY reads already-computed evidence and explains it in plain
language, connecting:

    numerical evidence -> physical meaning -> failure consequence

DATA CONTRACT
-------------
This module uses the EXISTING project dataclasses from
`models/data_models.py` exactly as defined -- no new or competing
dataclasses are created:

    ExplanationResult (returned)
    RiskResult, KinematicAnalysisResult, DynamicAnalysisResult,
    ObstacleResult, SingularityResult (consumed)

CONFIG USAGE
------------
Only one existing config value is used, to describe the obstacle
safety threshold in plain language:

    config.MIN_SAFE_CLEARANCE

No new configuration values are invented.

ARCHITECTURAL BOUNDARY
-----------------------
    prediction/     -> "What is the estimated failure risk?"
    explainability/ -> "Why is it risky?"                     (THIS FILE)
    recovery/       -> "Can we fix it?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MIN_SAFE_CLEARANCE

from models.data_models import (
    DynamicAnalysisResult,
    ExplanationResult,
    KinematicAnalysisResult,
    ObstacleResult,
    RiskResult,
    SingularityResult,
)


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})

# Human-readable translations of the internal risk-category names used
# by config.RISK_WEIGHTS / RiskResult.risk_contributions /
# RiskResult.main_contributors. This is presentation-only; it does not
# change, re-rank, or reinterpret the underlying risk values.
_CATEGORY_LABELS: Dict[str, str] = {
    "torque_overload": "Torque overload",
    "excessive_velocity": "Excessive joint velocity",
    "joint_limit_violation": "Joint-limit violation",
    "insufficient_clearance": "Insufficient obstacle clearance",
    "singularity_proximity": "Singularity proximity",
}


# =============================================================================
# INTERNAL HELPER STRUCTURE
# =============================================================================
#
# This is a small, private, internal helper used only to assemble the
# existing ExplanationResult. It is NOT a replacement or competitor for
# any project dataclass and is never returned to callers directly.


@dataclass
class _FactorFinding:
    """
    One structured, human-readable finding for a single risk category,
    built entirely from already-computed evidence.
    """

    category: str
    problem: str
    evidence_lines: List[str] = field(default_factory=list)
    meaning: str = ""
    why_it_matters: str = ""
    recovery_direction: str = ""
    evidence: Dict[str, object] = field(default_factory=dict)


# =============================================================================
# EXPLANATION ENGINE
# =============================================================================


class ExplanationEngine:
    """
    Converts already-computed analysis and risk evidence into a
    structured, human-readable explanation.

    This class performs explanation only. It does not recompute any
    kinematic, dynamic, obstacle, singularity, or risk quantity.
    """

    def __init__(self) -> None:
        """
        Initialize the explanation engine.

        No configuration values are required beyond
        `config.MIN_SAFE_CLEARANCE`, which is read directly at
        explanation time (not duplicated as a class attribute here so
        that it always reflects the single source of truth).
        """

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def explain(
        self,
        kinematic_result: KinematicAnalysisResult,
        dynamic_result: DynamicAnalysisResult,
        obstacle_result: ObstacleResult,
        singularity_result: SingularityResult,
        risk_result: RiskResult,
    ) -> ExplanationResult:
        """
        Produce the project's existing `ExplanationResult` for one
        trajectory's analysis results.

        Args:
            kinematic_result: Structured kinematic-safety evidence.
            dynamic_result: Structured dynamic-safety evidence.
            obstacle_result: Structured obstacle-clearance evidence.
            singularity_result: Structured singularity-proximity
                evidence.
            risk_result: The already-computed estimated failure risk.

        Returns:
            ExplanationResult.

        Raises:
            ValueError: if any input is missing, malformed, or
                internally inconsistent (see `_validate_inputs`).
        """
        explanation, _details = self.explain_with_details(
            kinematic_result,
            dynamic_result,
            obstacle_result,
            singularity_result,
            risk_result,
        )
        return explanation

    # -------------------------------------------------------------------------

    def explain_with_details(
        self,
        kinematic_result: KinematicAnalysisResult,
        dynamic_result: DynamicAnalysisResult,
        obstacle_result: ObstacleResult,
        singularity_result: SingularityResult,
        risk_result: RiskResult,
    ) -> Tuple[ExplanationResult, Dict[str, object]]:
        """
        Same as `explain`, but also returns an internal details
        dictionary (ordered findings, per-category presence) useful
        for debugging, testing, and visualization.

        Returns:
            Tuple of:
                explanation: ExplanationResult
                details: Dict[str, object]
        """

        # ---------------------------------------------------------------------
        # 1. Validate inputs. Never repaired, never mutated.
        # ---------------------------------------------------------------------

        self._validate_inputs(
            kinematic_result,
            dynamic_result,
            obstacle_result,
            singularity_result,
            risk_result,
        )

        # ---------------------------------------------------------------------
        # 2. Build one structured finding per detected problem category,
        #    using ONLY already-computed evidence.
        # ---------------------------------------------------------------------

        findings_by_category: Dict[str, _FactorFinding] = {}

        torque_finding = self._explain_torque(dynamic_result)
        if torque_finding is not None:
            findings_by_category[torque_finding.category] = torque_finding

        velocity_finding = self._explain_velocity(kinematic_result)
        if velocity_finding is not None:
            findings_by_category[velocity_finding.category] = velocity_finding

        joint_limit_finding = self._explain_joint_limits(kinematic_result)
        if joint_limit_finding is not None:
            findings_by_category[joint_limit_finding.category] = (
                joint_limit_finding
            )

        obstacle_finding = self._explain_obstacle(obstacle_result)
        if obstacle_finding is not None:
            findings_by_category[obstacle_finding.category] = obstacle_finding

        singularity_finding = self._explain_singularity(singularity_result)
        if singularity_finding is not None:
            findings_by_category[singularity_finding.category] = (
                singularity_finding
            )

        # ---------------------------------------------------------------------
        # 3. Order findings using the ALREADY-COMPUTED ranking from
        #    RiskResult.main_contributors. The ranking itself is never
        #    recalculated here.
        # ---------------------------------------------------------------------

        ordered_findings = [
            findings_by_category[category]
            for category in risk_result.main_contributors
            if category in findings_by_category
        ]

        # ---------------------------------------------------------------------
        # 4. Build the pieces of ExplanationResult.
        # ---------------------------------------------------------------------

        summary = self._build_summary(risk_result, ordered_findings)

        detailed_explanation = self._build_detailed_explanation(
            ordered_findings,
            dynamic_result,
        )

        main_reason = self._build_main_reason(ordered_findings)

        factor_explanations = {
            finding.category: f"{finding.meaning} {finding.why_it_matters}"
            for finding in ordered_findings
        }

        recommended_action = self._build_recommended_action(ordered_findings)

        evidence = self._build_evidence(
            risk_result,
            dynamic_result,
            ordered_findings,
        )

        # ---------------------------------------------------------------------
        # 5. Build the project's existing ExplanationResult dataclass.
        # ---------------------------------------------------------------------

        explanation = ExplanationResult(
            summary=summary,
            detailed_explanation=detailed_explanation,
            main_reason=main_reason,
            factor_explanations=factor_explanations,
            recommended_action=recommended_action,
            evidence=evidence,
        )

        details: Dict[str, object] = {
            "ordered_categories": [f.category for f in ordered_findings],
            "findings_by_category": dict(findings_by_category),
        }

        return explanation, details

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @staticmethod
    def _validate_inputs(
        kinematic_result: KinematicAnalysisResult,
        dynamic_result: DynamicAnalysisResult,
        obstacle_result: ObstacleResult,
        singularity_result: SingularityResult,
        risk_result: RiskResult,
    ) -> None:
        """
        Validate all five inputs. Never silently repairs malformed
        data; raises ValueError with a clear message instead.
        """

        if kinematic_result is None:
            raise ValueError("kinematic_result must not be None.")
        if dynamic_result is None:
            raise ValueError("dynamic_result must not be None.")
        if obstacle_result is None:
            raise ValueError("obstacle_result must not be None.")
        if singularity_result is None:
            raise ValueError("singularity_result must not be None.")
        if risk_result is None:
            raise ValueError("risk_result must not be None.")

        # ---------------------------------------------------------------------
        # Kinematic analysis result
        # ---------------------------------------------------------------------

        for field_name in (
            "velocity_violation_count",
            "max_velocity_utilization",
            "max_velocity_excess",
            "joint_limit_violation_count",
            "min_joint_limit_margin",
        ):
            array = np.asarray(
                getattr(kinematic_result, field_name), dtype=float
            )
            if array.shape != (3,):
                raise ValueError(
                    f"kinematic_result.{field_name} must have shape (3,), "
                    f"got {array.shape}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"kinematic_result.{field_name} contains NaN or "
                    "infinite values."
                )

        joint_limit_violated = np.asarray(
            kinematic_result.joint_limit_violated, dtype=bool
        )
        if joint_limit_violated.shape != (3,):
            raise ValueError(
                "kinematic_result.joint_limit_violated must have shape (3,)."
            )

        if not (1 <= int(kinematic_result.worst_velocity_joint) <= 3):
            raise ValueError(
                "kinematic_result.worst_velocity_joint must be between "
                "1 and 3."
            )

        # ---------------------------------------------------------------------
        # Dynamic analysis result
        # ---------------------------------------------------------------------

        for field_name in (
            "torque_violation_count",
            "max_torque_utilization",
            "max_torque_excess",
        ):
            array = np.asarray(
                getattr(dynamic_result, field_name), dtype=float
            )
            if array.shape != (3,):
                raise ValueError(
                    f"dynamic_result.{field_name} must have shape (3,), "
                    f"got {array.shape}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"dynamic_result.{field_name} contains NaN or "
                    "infinite values."
                )

        if not (1 <= int(dynamic_result.worst_torque_joint) <= 3):
            raise ValueError(
                "dynamic_result.worst_torque_joint must be between 1 and 3."
            )

        if not np.isfinite(dynamic_result.total_energy):
            raise ValueError("dynamic_result.total_energy must be finite.")

        # ---------------------------------------------------------------------
        # Obstacle result
        # ---------------------------------------------------------------------

        minimum_clearance = float(obstacle_result.minimum_clearance)
        if np.isnan(minimum_clearance):
            raise ValueError(
                "obstacle_result.minimum_clearance cannot be NaN."
            )

        # ---------------------------------------------------------------------
        # Singularity result
        # ---------------------------------------------------------------------

        if not np.isfinite(singularity_result.worst_condition_metric):
            raise ValueError(
                "singularity_result.worst_condition_metric must be finite."
            )
        if singularity_result.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(
                "singularity_result.risk_level must be LOW, MEDIUM, or "
                "HIGH."
            )

        # ---------------------------------------------------------------------
        # Risk result
        # ---------------------------------------------------------------------

        if not np.isfinite(risk_result.overall_risk):
            raise ValueError("risk_result.overall_risk must be finite.")
        if not (0.0 <= risk_result.overall_risk <= 1.0):
            raise ValueError(
                "risk_result.overall_risk must be between 0 and 1."
            )
        if not np.isfinite(risk_result.risk_percentage):
            raise ValueError("risk_result.risk_percentage must be finite.")
        if not (0.0 <= risk_result.risk_percentage <= 100.0):
            raise ValueError(
                "risk_result.risk_percentage must be between 0 and 100."
            )
        if risk_result.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(
                "risk_result.risk_level must be LOW, MEDIUM, or HIGH."
            )
        if not risk_result.main_contributors:
            raise ValueError(
                "risk_result.main_contributors must not be empty."
            )
        for category in risk_result.main_contributors:
            if category not in _CATEGORY_LABELS:
                raise ValueError(
                    "risk_result.main_contributors contains an unknown "
                    f"category: '{category}'."
                )
        for category, value in risk_result.risk_contributions.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"risk_result.risk_contributions['{category}'] must "
                    "be finite."
                )

    # =========================================================================
    # TORQUE EXPLANATION
    # =========================================================================

    @staticmethod
    def _explain_torque(
        dynamic_result: DynamicAnalysisResult,
    ) -> Optional[_FactorFinding]:
        """
        Build a torque-overload finding using DynamicAnalysisResult.
        Returns None if no torque overload was detected.
        """

        if not dynamic_result.torque_limit_violated:
            return None

        joint = int(dynamic_result.worst_torque_joint)
        idx = joint - 1

        utilization = float(dynamic_result.max_torque_utilization[idx])
        excess = float(dynamic_result.max_torque_excess[idx])
        violation_count = int(dynamic_result.torque_violation_count[idx])

        lines = [
            f"Joint {joint} exceeds its configured torque capability.",
            f"Maximum torque utilization: {utilization:.2f}x the "
            "configured limit.",
            f"Maximum torque excess: {excess:.2f} N*m above the "
            "configured limit.",
        ]
        if violation_count > 0:
            lines.append(
                f"This occurred at {violation_count} trajectory sample(s)."
            )

        other_joints = [
            j + 1
            for j in range(3)
            if j != idx and int(dynamic_result.torque_violation_count[j]) > 0
        ]
        if other_joints:
            lines.append(
                "Other joints also exceeded their configured torque "
                "limits: " + ", ".join(f"Joint {j}" for j in other_joints)
                + "."
            )

        return _FactorFinding(
            category="torque_overload",
            problem=f"Joint {joint} torque overload",
            evidence_lines=lines,
            meaning=(
                f"The trajectory requires substantially more torque from "
                f"Joint {joint} than the actuator is configured to "
                "produce."
            ),
            why_it_matters=(
                "The actuator may not be able to execute the requested "
                "motion reliably, which increases the possibility that "
                "the trajectory cannot be physically executed as planned."
            ),
            recovery_direction=(
                "Reduce the required joint torque by lowering trajectory "
                "speed or aggressiveness, choosing another joint "
                "configuration, or selecting another trajectory."
            ),
            evidence={
                "joint": joint,
                "max_torque_utilization": utilization,
                "max_torque_excess_nm": excess,
                "torque_violation_count": violation_count,
            },
        )

    # =========================================================================
    # VELOCITY EXPLANATION
    # =========================================================================

    @staticmethod
    def _explain_velocity(
        kinematic_result: KinematicAnalysisResult,
    ) -> Optional[_FactorFinding]:
        """
        Build an excessive-velocity finding using
        KinematicAnalysisResult. Returns None if no velocity violation
        was detected.
        """

        if not kinematic_result.velocity_limit_violated:
            return None

        joint = int(kinematic_result.worst_velocity_joint)
        idx = joint - 1

        utilization = float(kinematic_result.max_velocity_utilization[idx])
        excess = float(kinematic_result.max_velocity_excess[idx])
        violation_count = int(kinematic_result.velocity_violation_count[idx])

        lines = [
            f"Joint {joint} is required to move faster than its "
            "configured maximum velocity.",
            f"Maximum velocity utilization: {utilization:.2f}x the "
            "configured limit.",
            f"Maximum velocity excess: {excess:.3f} rad/s above the "
            "configured limit.",
        ]
        if violation_count > 0:
            lines.append(
                f"This occurred at {violation_count} trajectory sample(s)."
            )

        other_joints = [
            j + 1
            for j in range(3)
            if j != idx
            and int(kinematic_result.velocity_violation_count[j]) > 0
        ]
        if other_joints:
            lines.append(
                "Other joints also exceeded their configured velocity "
                "limits: " + ", ".join(f"Joint {j}" for j in other_joints)
                + "."
            )

        return _FactorFinding(
            category="excessive_velocity",
            problem=f"Joint {joint} velocity limit exceeded",
            evidence_lines=lines,
            meaning=(
                f"The trajectory requires Joint {joint} to move faster "
                "than its allowed operating speed."
            ),
            why_it_matters=(
                "This can make the planned motion difficult or "
                "impossible for the actuator to execute reliably."
            ),
            recovery_direction=(
                "Reduce joint speed, use a slower trajectory, or "
                "generate another trajectory."
            ),
            evidence={
                "joint": joint,
                "max_velocity_utilization": utilization,
                "max_velocity_excess_rad_s": excess,
                "velocity_violation_count": violation_count,
            },
        )

    # =========================================================================
    # JOINT-LIMIT EXPLANATION
    # =========================================================================

    @staticmethod
    def _explain_joint_limits(
        kinematic_result: KinematicAnalysisResult,
    ) -> Optional[_FactorFinding]:
        """
        Build a joint-limit-violation finding using
        KinematicAnalysisResult. Returns None if no joint exceeds its
        configured angular range.
        """

        violated = np.asarray(
            kinematic_result.joint_limit_violated, dtype=bool
        )
        if not np.any(violated):
            return None

        margins = kinematic_result.min_joint_limit_margin
        counts = kinematic_result.joint_limit_violation_count

        violated_joints = [j + 1 for j in range(3) if violated[j]]

        lines: List[str] = []
        joint_evidence: Dict[str, object] = {}

        for j in range(3):
            if not violated[j]:
                continue
            joint = j + 1
            margin = float(margins[j])
            count = int(counts[j])
            lines.append(
                f"Joint {joint} reaches or exceeds its allowed angular "
                f"range: the trajectory crosses the configured limit by "
                f"{abs(margin):.4f} rad, violated at {count} trajectory "
                "sample(s)."
            )
            joint_evidence[f"joint_{joint}_margin_rad"] = margin
            joint_evidence[f"joint_{joint}_violation_count"] = count

        problem = (
            "Joint limit violation ("
            + ", ".join(f"Joint {j}" for j in violated_joints)
            + ")"
        )

        return _FactorFinding(
            category="joint_limit_violation",
            problem=problem,
            evidence_lines=lines,
            meaning=(
                "One or more joints are commanded to positions outside "
                "their configured allowed angular range."
            ),
            why_it_matters=(
                "The robot cannot physically move a joint beyond its "
                "mechanical range, so the trajectory as planned cannot "
                "be executed exactly as specified."
            ),
            recovery_direction=(
                "Move the trajectory away from the joint limit, choose "
                "another joint configuration, or regenerate the "
                "trajectory."
            ),
            evidence=joint_evidence,
        )

    # =========================================================================
    # OBSTACLE EXPLANATION
    # =========================================================================

    @staticmethod
    def _explain_obstacle(
        obstacle_result: ObstacleResult,
    ) -> Optional[_FactorFinding]:
        """
        Build an obstacle-clearance finding using ObstacleResult.
        Returns None if there is no collision and clearance is at or
        above the configured safety margin.
        """

        collision_detected = bool(obstacle_result.collision_detected)
        clearance = float(obstacle_result.minimum_clearance)

        is_finite_clearance = np.isfinite(clearance)
        is_low_clearance = (
            is_finite_clearance and clearance < MIN_SAFE_CLEARANCE
        )

        if not collision_detected and not is_low_clearance:
            return None

        lines: List[str] = []

        if collision_detected:
            lines.append(
                "The robot's link geometry comes into contact with or "
                "penetrates the defined obstacle region."
            )
            if clearance < 0.0:
                lines.append(
                    f"Minimum clearance is {clearance:.4f} m (negative), "
                    "indicating geometric penetration of the obstacle "
                    "boundary."
                )
            elif clearance == 0.0:
                lines.append(
                    "Minimum clearance is exactly 0 m: the robot is "
                    "exactly touching the obstacle boundary."
                )
        else:
            lines.append(
                f"The robot remains outside the obstacle boundary but "
                f"does not maintain sufficient clearance (minimum "
                f"clearance = {clearance:.4f} m, below the configured "
                f"safety threshold of {MIN_SAFE_CLEARANCE:.2f} m)."
            )

        if obstacle_result.closest_obstacle_id:
            lines.append(
                f"The closest obstacle is '{obstacle_result.closest_obstacle_id}'."
            )

        problem = (
            "Obstacle collision"
            if collision_detected
            else "Insufficient obstacle clearance"
        )

        evidence: Dict[str, object] = {
            "collision_detected": collision_detected,
            "closest_obstacle_id": obstacle_result.closest_obstacle_id,
        }
        if is_finite_clearance:
            evidence["minimum_clearance_m"] = clearance

        return _FactorFinding(
            category="insufficient_clearance",
            problem=problem,
            evidence_lines=lines,
            meaning=(
                "The robot's link geometry -- not only the "
                "end-effector -- is too close to, touching, or "
                "penetrating a defined obstacle region."
            ),
            why_it_matters=(
                "Executing the trajectory as planned risks contact "
                "with the obstacle, which the robot must avoid."
            ),
            recovery_direction=(
                "Increase obstacle clearance, choose another path, or "
                "move the trajectory around the obstacle."
            ),
            evidence=evidence,
        )

    # =========================================================================
    # SINGULARITY EXPLANATION
    # =========================================================================

    @staticmethod
    def _explain_singularity(
        singularity_result: SingularityResult,
    ) -> Optional[_FactorFinding]:
        """
        Build a singularity-proximity finding using SingularityResult.
        Returns None if the trajectory is not near a singular
        configuration and its risk level is not HIGH.
        """

        near = bool(singularity_result.near_singularity)
        high = singularity_result.risk_level == "HIGH"

        if not near and not high:
            return None

        metric = float(singularity_result.worst_condition_metric)
        manipulability = float(singularity_result.minimum_manipulability)

        lines = [
            "The robot approaches a kinematic singularity (a "
            "configuration where some end-effector movements become "
            "difficult to produce efficiently).",
            f"Worst-case singularity metric (smallest singular value) "
            f"observed: {metric:.5f}.",
            f"Minimum manipulability observed: {manipulability:.5f}.",
        ]
        if singularity_result.worst_sample_index is not None:
            lines.append(
                "This occurs at trajectory sample index "
                f"{int(singularity_result.worst_sample_index)}."
            )

        return _FactorFinding(
            category="singularity_proximity",
            problem="Singularity proximity",
            evidence_lines=lines,
            meaning=(
                "Near a singular configuration, the robot can lose "
                "effective freedom in certain motion directions, and "
                "small changes in the desired end-effector motion may "
                "require very large joint motions."
            ),
            why_it_matters=(
                "This can make the trajectory difficult to execute "
                "precisely and increases estimated failure risk, "
                "although it does not necessarily mean the robot "
                "becomes uncontrollable."
            ),
            recovery_direction=(
                "Move the trajectory away from the singular "
                "configuration, choose another joint configuration, or "
                "select another trajectory."
            ),
            evidence={
                "worst_condition_metric": metric,
                "minimum_manipulability": manipulability,
                "risk_level": singularity_result.risk_level,
                "worst_sample_index": singularity_result.worst_sample_index,
            },
        )

    # =========================================================================
    # SUMMARY / DETAILED EXPLANATION / MAIN REASON / RECOVERY
    # =========================================================================

    @staticmethod
    def _build_summary(
        risk_result: RiskResult,
        ordered_findings: List[_FactorFinding],
    ) -> str:
        """
        Build a one-paragraph summary connecting overall risk to the
        primary contributing factor(s).
        """

        if ordered_findings:
            top_label = _CATEGORY_LABELS.get(
                ordered_findings[0].category, ordered_findings[0].category
            )
            if len(ordered_findings) > 1:
                other_labels = ", ".join(
                    _CATEGORY_LABELS.get(f.category, f.category)
                    for f in ordered_findings[1:]
                )
                contributor_text = (
                    f"The primary contributing factor is {top_label}. "
                    f"Additional contributing factor(s): {other_labels}."
                )
            else:
                contributor_text = (
                    f"The primary contributing factor is {top_label}."
                )
        else:
            contributor_text = (
                "No individual failure factor was strongly flagged; the "
                "estimated risk reflects the combined weighted evidence "
                "across all categories."
            )

        return (
            f"Estimated failure risk is {risk_result.risk_percentage:.1f}% "
            f"({risk_result.risk_level}). {contributor_text}"
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _build_detailed_explanation(
        ordered_findings: List[_FactorFinding],
        dynamic_result: DynamicAnalysisResult,
    ) -> str:
        """
        Build the full structured explanation: one Problem / Evidence /
        Meaning / Why it matters / Recovery direction block per
        detected factor, followed by supporting (non-risk-category)
        energy information.
        """

        if ordered_findings:
            blocks = []
            for finding in ordered_findings:
                evidence_text = " ".join(finding.evidence_lines)
                block = (
                    f"Problem: {finding.problem}\n"
                    f"Evidence: {evidence_text}\n"
                    f"Meaning: {finding.meaning}\n"
                    f"Why it matters: {finding.why_it_matters}\n"
                    f"Recovery direction: {finding.recovery_direction}"
                )
                blocks.append(block)
        else:
            blocks = [
                "No individual risk factor was flagged as significant "
                "for this trajectory based on the current evidence."
            ]

        energy_note = (
            "Supporting information: the trajectory requires "
            f"{float(dynamic_result.total_energy):.2f} J of signed "
            "mechanical work. Energy is not treated as an independent "
            "failure-risk category in this project; it is provided only "
            "as supporting context."
        )
        blocks.append(energy_note)

        return "\n\n".join(blocks)

    # -------------------------------------------------------------------------

    @staticmethod
    def _build_main_reason(ordered_findings: List[_FactorFinding]) -> str:
        """
        Build a short human-readable label for the single most
        important detected factor.
        """

        if not ordered_findings:
            return (
                "No individual failure factor was flagged as "
                "significant; the estimated risk reflects the combined "
                "weighted evidence."
            )

        top_category = ordered_findings[0].category
        return _CATEGORY_LABELS.get(top_category, top_category)

    # -------------------------------------------------------------------------

    @staticmethod
    def _build_recommended_action(
        ordered_findings: List[_FactorFinding],
    ) -> str:
        """
        Build the combined recovery-direction recommendation text.

        These are RECOMMENDATIONS ONLY. This method does not modify
        the trajectory, and the recovery module later decides whether
        a suggested action can actually fix the trajectory.
        """

        if not ordered_findings:
            return (
                "No specific recovery action is currently suggested; "
                "the trajectory does not show a significant individual "
                "failure factor."
            )

        lines = []
        for finding in ordered_findings:
            label = _CATEGORY_LABELS.get(finding.category, finding.category)
            lines.append(f"{label}: {finding.recovery_direction}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------

    @staticmethod
    def _build_evidence(
        risk_result: RiskResult,
        dynamic_result: DynamicAnalysisResult,
        ordered_findings: List[_FactorFinding],
    ) -> Dict[str, object]:
        """
        Build the structured supporting-evidence dictionary attached
        to ExplanationResult.evidence.
        """

        evidence: Dict[str, object] = {
            "overall_risk": float(risk_result.overall_risk),
            "risk_percentage": float(risk_result.risk_percentage),
            "risk_level": risk_result.risk_level,
            "main_contributors": list(risk_result.main_contributors),
            "total_energy_j": float(dynamic_result.total_energy),
        }

        for finding in ordered_findings:
            for key, value in finding.evidence.items():
                evidence[f"{finding.category}.{key}"] = value

        return evidence


# =============================================================================
# DEMO / SANITY TEST
# =============================================================================

if __name__ == "__main__":

    import copy

    from robot.robot_model import Planar3RRobot
    from robot.kinematics import KinematicAnalyzer
    from robot.dynamics import DynamicAnalyzer

    from analysis.kinematic_analysis import KinematicSafetyAnalyzer
    from analysis.dynamic_analysis import DynamicSafetyAnalyzer
    from analysis.obstacle_analysis import (
        CircularObstacle,
        ObstacleSafetyAnalyzer,
    )
    from analysis.singularity_analysis import SingularityAnalyzer

    from prediction.failure_predictor import FailurePredictor

    from models.data_models import Trajectory

    # =========================================================================
    # 1. Create deterministic demo trajectory (same style as the
    #    existing failure_predictor.py demo).
    # =========================================================================

    n_samples = 100
    duration = 4.0

    time = np.linspace(0.0, duration, n_samples)

    q1_target = 2.6
    q2_target = 1.0
    q3_target = 0.5

    fraction = time / duration

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
        path_id="demo-explanation-001",
        name="Explanation engine demo",
        description=(
            "Synthetic trajectory for testing the explanation engine."
        ),
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        start_position=robot.end_effector_position(q[0]),
        goal_position=robot.end_effector_position(q[-1]),
        duration=duration,
        path_length=0.0,
    )

    # =========================================================================
    # 2. Run the existing analysis pipeline.
    # =========================================================================

    kinematic_analyzer = KinematicAnalyzer(robot)
    kinematic_result = kinematic_analyzer.analyze(demo_trajectory)

    kinematic_safety_analyzer = KinematicSafetyAnalyzer()
    kinematic_analysis_result = kinematic_safety_analyzer.analyze(
        kinematic_result
    )

    dynamic_analyzer = DynamicAnalyzer(robot)
    dynamic_result = dynamic_analyzer.analyze(demo_trajectory)

    dynamic_safety_analyzer = DynamicSafetyAnalyzer()
    dynamic_analysis_result = dynamic_safety_analyzer.analyze(dynamic_result)

    demo_obstacles = [
        CircularObstacle(
            obstacle_id="O1",
            center=np.array([1.3, 0.6], dtype=float),
            radius=0.20,
        )
    ]

    obstacle_analyzer = ObstacleSafetyAnalyzer(
        robot=robot,
        obstacles=demo_obstacles,
    )
    obstacle_result = obstacle_analyzer.analyze(demo_trajectory)

    singularity_analyzer = SingularityAnalyzer(robot)
    singularity_result = singularity_analyzer.analyze(demo_trajectory)

    predictor = FailurePredictor()
    risk_result = predictor.predict(
        kinematic_analysis_result,
        dynamic_analysis_result,
        obstacle_result,
        singularity_result,
    )

    # =========================================================================
    # 3. Explain.
    # =========================================================================

    engine = ExplanationEngine()

    # Keep copies so we can verify that explanation does not modify
    # any input.
    kinematic_before = copy.deepcopy(kinematic_analysis_result)
    dynamic_before = copy.deepcopy(dynamic_analysis_result)
    obstacle_before = copy.deepcopy(obstacle_result)
    singularity_before = copy.deepcopy(singularity_result)
    risk_before = copy.deepcopy(risk_result)

    explanation, details = engine.explain_with_details(
        kinematic_analysis_result,
        dynamic_analysis_result,
        obstacle_result,
        singularity_result,
        risk_result,
    )

    # =========================================================================
    # 4. Display results.
    # =========================================================================

    print(f"Number of samples: {n_samples}")
    print(f"Overall risk: {risk_result.risk_percentage:.1f}%")
    print(f"Risk level: {risk_result.risk_level}")

    print("\nMain contributors:")
    for rank, category in enumerate(risk_result.main_contributors, start=1):
        print(f"  {rank}. {category}")

    print("\nDetected problems:")
    if risk_result.detected_problems:
        for problem in risk_result.detected_problems:
            print(f"  - {problem}")
    else:
        print("  (none)")

    print("\nSummary:")
    print(explanation.summary)

    print("\nMain reason:")
    print(explanation.main_reason)

    print("\nDetailed explanation:")
    print(explanation.detailed_explanation)

    print("\nRecommended action (recovery directions):")
    print(explanation.recommended_action)

    # =========================================================================
    # 5. Sanity checks.
    # =========================================================================

    assert isinstance(explanation, ExplanationResult)
    assert explanation.summary.strip() != ""
    assert explanation.detailed_explanation.strip() != ""
    assert explanation.main_reason.strip() != ""

    for key, value in explanation.evidence.items():
        if isinstance(value, float):
            assert np.isfinite(value), f"Non-finite evidence value: {key}"

    # Determinism: repeated calls with the same inputs produce the
    # same explanation.
    explanation_repeat, _details_repeat = engine.explain_with_details(
        kinematic_analysis_result,
        dynamic_analysis_result,
        obstacle_result,
        singularity_result,
        risk_result,
    )

    assert explanation_repeat.summary == explanation.summary
    assert (
        explanation_repeat.detailed_explanation
        == explanation.detailed_explanation
    )
    assert explanation_repeat.main_reason == explanation.main_reason
    assert (
        explanation_repeat.recommended_action
        == explanation.recommended_action
    )
    assert explanation_repeat.factor_explanations == (
        explanation.factor_explanations
    )

    # Risk level / percentage consistency: the explanation must not
    # invent its own numbers -- it must reflect RiskResult exactly.
    assert f"{risk_result.risk_percentage:.1f}%" in explanation.summary
    assert risk_result.risk_level in explanation.summary
    assert explanation.evidence["overall_risk"] == risk_result.overall_risk
    assert (
        explanation.evidence["risk_percentage"]
        == risk_result.risk_percentage
    )

    # Input objects must not be modified.
    assert np.array_equal(
        kinematic_analysis_result.min_joint_limit_margin,
        kinematic_before.min_joint_limit_margin,
    )
    assert np.array_equal(
        dynamic_analysis_result.max_torque_utilization,
        dynamic_before.max_torque_utilization,
    )
    assert (
        obstacle_result.collision_detected
        == obstacle_before.collision_detected
    )
    assert (
        singularity_result.worst_condition_metric
        == singularity_before.worst_condition_metric
    )
    assert risk_result.overall_risk == risk_before.overall_risk
    assert risk_result.detected_problems == risk_before.detected_problems

    print("\nAll sanity checks passed.")