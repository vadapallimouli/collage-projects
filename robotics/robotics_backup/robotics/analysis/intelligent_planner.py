# analysis/intelligent_planner.py

"""
Integrated Intelligent Motion Planning Pipeline.

Pipeline:

    Start + Goal
        ↓
    Generate many candidate trajectories
        ↓
    ML failure prediction
        ↓
    Physical collision validation
        ↓
    Recoverability analysis
        ↓
    Select a feasible trajectory
        ↓
    Final result

Important:
- ML prediction is used for failure prediction.
- CollisionChecker is the physical ground truth.
- ML predictions never override physical collision validation.
- No hand-designed weighted risk score is used.
"""

import numpy as np


class IntelligentMotionPlanner:
    """
    Integrates:

    - obstacle-aware trajectory planning
    - trajectory feature extraction
    - ML failure prediction
    - physical collision validation
    - recoverability analysis
    - final trajectory selection
    """

    def __init__(
        self,
        planner,
        feature_extractor,
        failure_predictor,
        recoverability_analyzer,
        collision_checker,
    ):
        self.planner = planner
        self.feature_extractor = feature_extractor
        self.failure_predictor = failure_predictor
        self.recoverability_analyzer = (
            recoverability_analyzer
        )
        self.collision_checker = collision_checker

    # ================================================================
    # VALIDATE TRAJECTORY FORMAT
    # ================================================================

    def _validate_trajectory(
        self,
        trajectory,
    ):
        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "Trajectory must be a 2D array."
            )

        if trajectory.shape[1] != 7:
            raise ValueError(
                "Trajectory must have 7 joint dimensions."
            )

        if trajectory.shape[0] < 2:
            raise ValueError(
                "Trajectory must contain at least 2 configurations."
            )

        if not np.all(
            np.isfinite(trajectory)
        ):
            raise ValueError(
                "Trajectory contains NaN or infinite values."
            )

        return trajectory

    # ================================================================
    # PHYSICAL VALIDATION
    # ================================================================

    def _physical_validation(
        self,
        trajectory,
    ):
        """
        Physically validate a trajectory.

        CollisionChecker is the ground truth.
        """

        trajectory = self._validate_trajectory(
            trajectory
        )

        result = (
            self.collision_checker.check_trajectory(
                trajectory
            )
        )

        collision_free = bool(
            result["collision_free"]
        )

        return {
            "valid": collision_free,
            "collision_free": collision_free,
            "collision_count": int(
                result["collision_count"]
            ),
            "self_collision_count": int(
                result["self_collision_count"]
            ),
            "obstacle_collision_count": int(
                result[
                    "obstacle_collision_count"
                ]
            ),
            "joint_limit_violation_count": int(
                result[
                    "joint_limit_violation_count"
                ]
            ),
            "num_configurations": int(
                result[
                    "num_configurations"
                ]
            ),
        }

    # ================================================================
    # GENERATE CANDIDATES
    # ================================================================

    def generate_candidates(
        self,
        start,
        goal,
        num_candidates=10,
    ):
        """
        Generate multiple obstacle-aware
        candidate trajectories.
        """

        start = np.asarray(
            start,
            dtype=float,
        )

        goal = np.asarray(
            goal,
            dtype=float,
        )

        if start.shape != (7,):
            raise ValueError(
                "Start configuration must have shape (7,)."
            )

        if goal.shape != (7,):
            raise ValueError(
                "Goal configuration must have shape (7,)."
            )

        if num_candidates < 1:
            raise ValueError(
                "num_candidates must be at least 1."
            )

        results = (
            self.planner.plan_multiple(
                start,
                goal,
                num_candidates,
            )
        )

        trajectories = []

        for result in results:

            trajectory = result.get(
                "trajectory"
            )

            if trajectory is None:
                continue

            trajectory = (
                self._validate_trajectory(
                    trajectory
                )
            )

            trajectories.append(
                trajectory
            )

        return trajectories

    # ================================================================
    # PREDICT CANDIDATES
    # ================================================================

    def predict_candidates(
        self,
        trajectories,
        top_k=5,
    ):
        """
        Run ML failure prediction for every
        candidate trajectory.
        """

        if trajectories is None:
            raise ValueError(
                "trajectories cannot be None."
            )

        predictions = []

        for index, trajectory in enumerate(
            trajectories
        ):

            trajectory = (
                self._validate_trajectory(
                    trajectory
                )
            )

            features = (
                self.feature_extractor.extract(
                    trajectory
                )
            )

            prediction = (
                self.failure_predictor.predict_from_features(
                    features
                )
            )

            prediction = dict(
                prediction
            )

            prediction[
                "trajectory_index"
            ] = index

            predictions.append(
                prediction
            )

        return predictions

    # ================================================================
    # PHYSICAL VALIDATION OF ALL CANDIDATES
    # ================================================================

    def validate_candidates(
        self,
        trajectories,
    ):
        """
        Physically validate every candidate.
        """

        if trajectories is None:
            raise ValueError(
                "trajectories cannot be None."
            )

        validations = []

        for index, trajectory in enumerate(
            trajectories
        ):

            validation = (
                self._physical_validation(
                    trajectory
                )
            )

            validation[
                "trajectory_index"
            ] = index

            validations.append(
                validation
            )

        return validations

    # ================================================================
    # COMBINE RESULTS
    # ================================================================

    def combine_results(
        self,
        trajectories,
        predictions,
        validations,
    ):
        """
        Combine trajectory, ML prediction,
        and physical validation information.
        """

        if not (
            len(trajectories)
            == len(predictions)
            == len(validations)
        ):
            raise ValueError(
                "Trajectories, predictions, and "
                "validations must have the same length."
            )

        combined = []

        for index in range(
            len(trajectories)
        ):

            prediction = dict(
                predictions[index]
            )

            validation = dict(
                validations[index]
            )

            result = {
                "trajectory_index": index,
                "trajectory": trajectories[
                    index
                ],
                "prediction": prediction,
                "validation": validation,
                "predicted_class": prediction.get(
                    "predicted_class"
                ),
                "success_probability": float(
                    prediction.get(
                        "success_probability",
                        0.0,
                    )
                ),
                "failure_probability": float(
                    prediction.get(
                        "failure_probability",
                        1.0,
                    )
                ),
                "valid": bool(
                    validation.get(
                        "valid",
                        False,
                    )
                ),
                "collision_free": bool(
                    validation.get(
                        "collision_free",
                        False,
                    )
                ),
            }

            combined.append(
                result
            )

        return combined

    # ================================================================
    # GET FEASIBLE CANDIDATES
    # ================================================================

    def get_feasible_candidates(
        self,
        combined_results,
    ):
        """
        Return only physically feasible candidates.
        """

        return [
            result
            for result in combined_results
            if result["validation"].get(
                "valid",
                False,
            )
        ]

    # ================================================================
    # GET FAILED CANDIDATES
    # ================================================================

    def get_failed_candidates(
        self,
        combined_results,
    ):
        """
        Return physically failed candidates.
        """

        return [
            result
            for result in combined_results
            if not result["validation"].get(
                "valid",
                False,
            )
        ]

    # ================================================================
    # RECOVERABILITY ANALYSIS
    # ================================================================

    def analyze_recoverability(
        self,
        trajectories,
    ):
        """
        Analyze recoverability of candidate trajectories.
        """

        if trajectories is None:
            raise ValueError(
                "trajectories cannot be None."
            )

        validated_trajectories = []

        for trajectory in trajectories:
            validated_trajectories.append(
                self._validate_trajectory(
                    trajectory
                )
            )

        return (
            self.recoverability_analyzer
            .analyze_recoverability(
                validated_trajectories
            )
        )

    # ================================================================
    # SELECT FINAL TRAJECTORY
    # ================================================================

        # ================================================================
    # SELECT FINAL TRAJECTORY
    # ================================================================

    def select_final_trajectory(
        self,
        combined_results,
    ):
        """
        Select the final trajectory.

        Only physically valid candidates are considered.

        Among physically valid candidates, the candidate
        with the highest ML success probability is selected.

        Physical validation remains the final authority.
        """

        feasible = (
            self.get_feasible_candidates(
                combined_results
            )
        )

        if len(feasible) == 0:
            return {
                "success": False,
                "selected_index": None,
                "trajectory": None,
                "selected_trajectory": None,
                "selected_candidate": None,
                "success_probability": None,
                "failure_probability": None,
            }

        selected = max(
            feasible,
            key=lambda result: float(
                result.get(
                    "success_probability",
                    0.0,
                )
            ),
        )

        trajectory = selected[
            "trajectory"
        ]

        selected_index = int(
            selected[
                "trajectory_index"
            ]
        )

        success_probability = float(
            selected.get(
                "success_probability",
                0.0,
            )
        )

        failure_probability = float(
            selected.get(
                "failure_probability",
                1.0,
            )
        )

        return {
            "success": True,

            # Expected by the integration test
            "selected_index": selected_index,
            "trajectory": trajectory,

            # Kept as an explicit descriptive alias
            "selected_trajectory": trajectory,

            # Full selected candidate
            "selected_candidate": selected,

            # ML information
            "success_probability": success_probability,
            "failure_probability": failure_probability,
        }
        
    # ================================================================
    # COMPLETE PLANNING PIPELINE
    # ================================================================

    def plan(
        self,
        start_configuration,
        goal_configuration,
        num_candidates=10,
    ):
        """
        Run the complete intelligent planning pipeline.
        """

        trajectories = (
            self.generate_candidates(
                start_configuration,
                goal_configuration,
                num_candidates,
            )
        )

        if len(trajectories) == 0:
            return {
                "success": False,
                "trajectories": [],
                "predictions": [],
                "validations": [],
                "candidates": [],
                "feasible_candidates": [],
                "failed_candidates": [],
                "recoverability": None,
                "selected": {
                    "success": False,
                    "selected_index": None,
                    "selected_trajectory": None,
                    "selected_candidate": None,
                    "success_probability": None,
                    "failure_probability": None,
                },
            }

        predictions = (
            self.predict_candidates(
                trajectories
            )
        )

        validations = (
            self.validate_candidates(
                trajectories
            )
        )

        combined = (
            self.combine_results(
                trajectories,
                predictions,
                validations,
            )
        )

        feasible = (
            self.get_feasible_candidates(
                combined
            )
        )

        failed = (
            self.get_failed_candidates(
                combined
            )
        )

        try:

            recoverability = (
                self.analyze_recoverability(
                    trajectories
                )
            )

        except Exception as exc:

            recoverability = {
                "success": False,
                "error": str(exc),
            }

        selected = (
            self.select_final_trajectory(
                combined
            )
        )

        return {
            "success": bool(
                selected["success"]
            ),
            "trajectories": trajectories,
            "predictions": predictions,
            "validations": validations,
            "candidates": combined,
            "feasible_candidates": feasible,
            "failed_candidates": failed,
            "recoverability": recoverability,
            "selected": selected,
        }

    # ================================================================
    # SUMMARY
    # ================================================================

    def summarize(
        self,
        result,
    ):
        """
        Generate a compact summary.
        """

        if result is None:
            return {
                "success": False,
                "message": "No planning result.",
            }

        trajectories = result.get(
            "trajectories",
            [],
        )

        candidates = result.get(
            "candidates",
            [],
        )

        feasible = result.get(
            "feasible_candidates",
            [],
        )

        failed = result.get(
            "failed_candidates",
            [],
        )

        selected = result.get(
            "selected"
        )

        summary = {
            "success": bool(
                result.get(
                    "success",
                    False,
                )
            ),
            "num_candidates": len(
                trajectories
            ),
            "num_feasible": len(
                feasible
            ),
            "num_failed": len(
                failed
            ),
            "selected_index": None,
            "selected_success_probability": None,
            "selected_failure_probability": None,
        }

        if selected is not None:

            summary[
                "selected_index"
            ] = selected.get(
                "selected_index"
            )

            summary[
                "selected_success_probability"
            ] = selected.get(
                "success_probability"
            )

            summary[
                "selected_failure_probability"
            ] = selected.get(
                "failure_probability"
            )

        return summary