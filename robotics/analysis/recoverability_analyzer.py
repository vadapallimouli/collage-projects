"""
Recoverability Analyzer

Purpose
-------
Determine whether a failed/risky candidate trajectory has an alternative
feasible trajectory among the generated candidate paths.

The recoverability module is deliberately separate from the ML failure
predictor.

Pipeline:

    Candidate trajectories
            |
            v
    Physical validation
            |
       +----+----+
       |         |
    feasible    failed
       |         |
       |         v
       |   search other candidates
       |         |
       |         v
       +--> feasible alternative
                    |
                    v
              recovery option


Important
---------
CollisionChecker remains the physical ground truth.

This module does NOT:
    - create a hand-designed failure score
    - replace the ML model
    - use collision labels as ML features
    - assume exactly 3 candidate trajectories
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class RecoverabilityAnalyzer:
    """
    Analyze whether failed candidate trajectories can be replaced by
    other feasible candidate trajectories.

    Parameters
    ----------
    collision_checker:
        Existing CollisionChecker instance.

    trajectory_generator:
        Existing TrajectoryGenerator instance. It is stored for future
        integration with trajectory regeneration/recovery.
    """

    def __init__(
        self,
        collision_checker,
        trajectory_generator=None,
    ):
        self.collision_checker = collision_checker
        self.trajectory_generator = trajectory_generator

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_trajectory(
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Validate one trajectory.

        Expected shape:
            (N, 7)

        Returns
        -------
        np.ndarray
            Validated trajectory.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "Trajectory must be a 2D array. "
                f"Received shape: {trajectory.shape}"
            )

        if trajectory.shape[1] != 7:
            raise ValueError(
                "Trajectory must contain 7 joint values per "
                f"configuration. Received shape: {trajectory.shape}"
            )

        if trajectory.shape[0] < 2:
            raise ValueError(
                "Trajectory must contain at least 2 configurations."
            )

        if not np.all(np.isfinite(trajectory)):
            raise ValueError(
                "Trajectory contains NaN or infinite values."
            )

        return trajectory

    @classmethod
    def _validate_trajectories(
        cls,
        trajectories: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Validate a collection of candidate trajectories.
        """

        if trajectories is None:
            raise ValueError(
                "trajectories cannot be None."
            )

        trajectories = list(trajectories)

        if len(trajectories) == 0:
            raise ValueError(
                "At least one candidate trajectory is required."
            )

        return [
            cls._validate_trajectory(trajectory)
            for trajectory in trajectories
        ]

    # ==================================================================
    # SINGLE TRAJECTORY VALIDATION
    # ==================================================================

    def validate_trajectory(
        self,
        trajectory: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Physically validate a trajectory using CollisionChecker.

        CollisionChecker is treated as the ground-truth feasibility
        check.
        """

        trajectory = self._validate_trajectory(
            trajectory
        )

        result = self.collision_checker.check_trajectory(
            trajectory
        )

        return {
            "collision_free": bool(
                result["collision_free"]
            ),
            "collision_count": int(
                result["collision_count"]
            ),
            "self_collision_count": int(
                result["self_collision_count"]
            ),
            "obstacle_collision_count": int(
                result["obstacle_collision_count"]
            ),
            "joint_limit_violation_count": int(
                result["joint_limit_violation_count"]
            ),
            "num_configurations": int(
                result["num_configurations"]
            ),
        }

    # ==================================================================
    # VALIDATE ALL CANDIDATES
    # ==================================================================

    def analyze_candidates(
        self,
        trajectories: Sequence[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Validate all candidate trajectories.

        Returns
        -------
        list of dict
            One dictionary per candidate.
        """

        trajectories = self._validate_trajectories(
            trajectories
        )

        candidate_results = []

        for candidate_index, trajectory in enumerate(
            trajectories
        ):

            validation = self.validate_trajectory(
                trajectory
            )

            candidate_results.append(
                {
                    "candidate_index": candidate_index,
                    "trajectory": trajectory,
                    "validation": validation,
                    "feasible": validation["collision_free"],
                }
            )

        return candidate_results

    # ==================================================================
    # FILTER CANDIDATES
    # ==================================================================

    @staticmethod
    def get_feasible_candidates(
        candidate_results: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return all collision-free candidates.
        """

        return [
            candidate
            for candidate in candidate_results
            if candidate["feasible"]
        ]

    @staticmethod
    def get_failed_candidates(
        candidate_results: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Return all candidates that failed physical validation.
        """

        return [
            candidate
            for candidate in candidate_results
            if not candidate["feasible"]
        ]

    # ==================================================================
    # TRAJECTORY DISTANCE
    # ==================================================================

    @staticmethod
    def _resample_trajectory(
        trajectory: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """
        Resample a trajectory to a specified number of points.

        Linear interpolation is performed independently for each
        joint.
        """

        trajectory = RecoverabilityAnalyzer._validate_trajectory(
            trajectory
        )

        if num_points < 2:
            raise ValueError(
                "num_points must be at least 2."
            )

        if len(trajectory) == num_points:
            return trajectory.copy()

        old_parameter = np.linspace(
            0.0,
            1.0,
            len(trajectory),
        )

        new_parameter = np.linspace(
            0.0,
            1.0,
            num_points,
        )

        resampled = np.empty(
            (num_points, 7),
            dtype=float,
        )

        for joint_index in range(7):

            resampled[:, joint_index] = np.interp(
                new_parameter,
                old_parameter,
                trajectory[:, joint_index],
            )

        return resampled

    @classmethod
    def trajectory_distance(
        cls,
        trajectory_a: np.ndarray,
        trajectory_b: np.ndarray,
    ) -> float:
        """
        Calculate average joint-space distance between two trajectories.

        This is used only to describe how different a recovery
        trajectory is from the failed trajectory.

        It is NOT a failure/risk score.
        """

        trajectory_a = cls._validate_trajectory(
            trajectory_a
        )

        trajectory_b = cls._validate_trajectory(
            trajectory_b
        )

        if len(trajectory_a) != len(trajectory_b):

            num_points = max(
                len(trajectory_a),
                len(trajectory_b),
            )

            trajectory_a = cls._resample_trajectory(
                trajectory_a,
                num_points,
            )

            trajectory_b = cls._resample_trajectory(
                trajectory_b,
                num_points,
            )

        point_distances = np.linalg.norm(
            trajectory_a - trajectory_b,
            axis=1,
        )

        return float(
            np.mean(point_distances)
        )

    # ==================================================================
    # FIND ALTERNATIVES
    # ==================================================================

    def find_alternatives(
        self,
        failed_candidate_index: int,
        candidate_results: Sequence[Dict[str, Any]],
        max_alternatives: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find feasible alternatives for a failed candidate.

        Only candidates that are physically collision-free are
        considered alternatives.

        Alternatives are ordered by trajectory distance from the
        failed trajectory.

        Parameters
        ----------
        failed_candidate_index:
            Index of the failed candidate.

        candidate_results:
            Output of analyze_candidates().

        max_alternatives:
            Optional maximum number of alternatives to return.
        """

        if not isinstance(
            failed_candidate_index,
            (int, np.integer),
        ):
            raise TypeError(
                "failed_candidate_index must be an integer."
            )

        candidate_results = list(
            candidate_results
        )

        if len(candidate_results) == 0:
            raise ValueError(
                "candidate_results cannot be empty."
            )

        valid_indices = [
            candidate["candidate_index"]
            for candidate in candidate_results
        ]

        if failed_candidate_index not in valid_indices:
            raise IndexError(
                "failed_candidate_index does not exist "
                "in candidate_results."
            )

        failed_candidate = next(
            candidate
            for candidate in candidate_results
            if candidate["candidate_index"]
            == failed_candidate_index
        )

        failed_trajectory = failed_candidate[
            "trajectory"
        ]

        alternatives = []

        for candidate in candidate_results:

            candidate_index = candidate[
                "candidate_index"
            ]

            # Do not use the failed trajectory itself.
            if candidate_index == failed_candidate_index:
                continue

            # Only physically feasible trajectories can be
            # recovery alternatives.
            if not candidate["feasible"]:
                continue

            distance = self.trajectory_distance(
                failed_trajectory,
                candidate["trajectory"],
            )

            alternatives.append(
                {
                    "candidate_index": candidate_index,
                    "trajectory": candidate["trajectory"],
                    "validation": candidate["validation"],
                    "trajectory_distance": distance,
                }
            )

        alternatives.sort(
            key=lambda item:
            item["trajectory_distance"]
        )

        if max_alternatives is not None:

            if max_alternatives < 0:
                raise ValueError(
                    "max_alternatives cannot be negative."
                )

            alternatives = alternatives[
                :max_alternatives
            ]

        return alternatives

    # ==================================================================
    # COMPLETE RECOVERABILITY ANALYSIS
    # ==================================================================

    def analyze_recoverability(
        self,
        trajectories: Sequence[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Perform complete recoverability analysis.

        A failed candidate is considered recoverable if at least one
        other candidate trajectory is physically feasible.

        Returns
        -------
        dict
            Complete analysis result.
        """

        trajectories = self._validate_trajectories(
            trajectories
        )

        candidate_results = self.analyze_candidates(
            trajectories
        )

        feasible_candidates = (
            self.get_feasible_candidates(
                candidate_results
            )
        )

        failed_candidates = (
            self.get_failed_candidates(
                candidate_results
            )
        )

        recoverable_candidates = []
        unrecoverable_candidates = []

        for failed_candidate in failed_candidates:

            failed_index = failed_candidate[
                "candidate_index"
            ]

            alternatives = self.find_alternatives(
                failed_candidate_index=failed_index,
                candidate_results=candidate_results,
            )

            if len(alternatives) > 0:

                recoverable_candidates.append(
                    {
                        "failed_candidate_index": failed_index,
                        "alternatives": alternatives,
                    }
                )

            else:

                unrecoverable_candidates.append(
                    failed_index
                )

        total_failed = len(
            failed_candidates
        )

        if total_failed > 0:

            recoverability_rate = (
                len(recoverable_candidates)
                / total_failed
            )

        else:

            # If there are no failed candidates, there is
            # nothing that needs recovery.
            recoverability_rate = 1.0

        summary = {
            "total_candidates": len(
                candidate_results
            ),
            "feasible_candidates": len(
                feasible_candidates
            ),
            "failed_candidates": total_failed,
            "recoverable_failed_candidates": len(
                recoverable_candidates
            ),
            "unrecoverable_failed_candidates": len(
                unrecoverable_candidates
            ),
            "recoverability_rate": float(
                recoverability_rate
            ),
        }

        return {
            "candidate_results": candidate_results,
            "feasible_candidates": feasible_candidates,
            "failed_candidates": failed_candidates,
            "recoverable_candidates": recoverable_candidates,
            "unrecoverable_candidates": unrecoverable_candidates,
            "summary": summary,
        }

    # ==================================================================
    # SELECT RECOVERY TRAJECTORY
    # ==================================================================

    def select_recovery_trajectory(
        self,
        recoverability_result: Dict[str, Any],
        failed_candidate_index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the closest feasible alternative for a failed candidate.

        Returns
        -------
        dict or None
            Selected recovery candidate, or None when no feasible
            alternative exists.
        """

        recoverable_candidates = (
            recoverability_result[
                "recoverable_candidates"
            ]
        )

        for recovery_case in recoverable_candidates:

            if (
                recovery_case[
                    "failed_candidate_index"
                ]
                == failed_candidate_index
            ):

                alternatives = recovery_case[
                    "alternatives"
                ]

                if len(alternatives) == 0:
                    return None

                # Alternatives are already sorted by trajectory
                # distance, so the first one is the closest
                # feasible alternative.
                return alternatives[0]

        return None

    # ==================================================================
    # HUMAN-READABLE SUMMARY
    # ==================================================================

    @staticmethod
    def format_summary(
        recoverability_result: Dict[str, Any],
    ) -> str:
        """
        Create a readable text summary.
        """

        summary = recoverability_result[
            "summary"
        ]

        lines = [
            "",
            "=" * 68,
            "RECOVERABILITY ANALYSIS",
            "=" * 68,
            f"Total candidates             : {summary['total_candidates']}",
            f"Physically feasible          : {summary['feasible_candidates']}",
            f"Physically failed            : {summary['failed_candidates']}",
            f"Recoverable failed candidates: {summary['recoverable_failed_candidates']}",
            f"Unrecoverable failed         : {summary['unrecoverable_failed_candidates']}",
            f"Recoverability rate          : {summary['recoverability_rate']:.2%}",
        ]

        recoverable_cases = recoverability_result.get("recoverable_candidates", [])
        if recoverable_cases:
            lines.extend(["", "AVAILABLE FEASIBLE ALTERNATIVES:"])
            for case in recoverable_cases:
                failed_idx = int(case["failed_candidate_index"]) + 1
                alternatives = case.get("alternatives", [])
                lines.append(f"  If candidate {failed_idx} fails:")
                for alternative in alternatives[:3]:
                    alt_idx = int(alternative["candidate_index"]) + 1
                    distance = float(alternative.get("trajectory_distance", float("nan")))
                    lines.append(
                        f"    Candidate {alt_idx}: physically feasible; "
                        f"average joint-space distance={distance:.4f} rad"
                    )
        elif int(summary.get("failed_candidates", 0)) == 0:
            lines.extend([
                "",
                "No candidate failed physical validation, so no recovery was needed.",
                "Other physically feasible candidates remain alternatives.",
            ])
        else:
            lines.extend([
                "",
                "No physically feasible alternative was found for the failed candidates.",
            ])

        lines.append("=" * 68)
        return "\n".join(lines)