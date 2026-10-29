# prediction/failure_predictor.py

from pathlib import Path
from typing import Any, Dict, List

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


class FailurePredictor:
    """
    XGBoost-based trajectory failure predictor.

    Responsibilities
    ----------------
    1. Load the trained localized XGBoost model.
    2. Load the exact 17 ML feature names.
    3. Predict success/failure probabilities.
    4. Generate native XGBoost local feature contributions.
    5. Convert contributions into human-readable explanations.
    6. Predict single or multiple trajectories.

    Important
    ---------
    The ML model predicts failure risk.

    Physical collision validation remains separate and is
    handled by the collision checker.
    """

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(
        self,
        model_path=None,
        feature_names_path=None,
    ):
        """
        Initialize the failure predictor.

        Parameters
        ----------
        model_path:
            Path to the trained XGBoost model.

        feature_names_path:
            Path to the JSON file containing the 17 ML feature names.
        """

        project_root = Path(__file__).resolve().parent.parent

        if model_path is None:
            model_path = (
                project_root
                / "data"
                / "failure_prediction"
                / "models"
                / "xgboost_localized.joblib"
            )

        if feature_names_path is None:
            feature_names_path = (
                project_root
                / "data"
                / "failure_prediction"
                / "models"
                / "feature_names_localized.json"
            )

        self.model_path = Path(model_path)
        self.feature_names_path = Path(
            feature_names_path
        )

        # ------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Failure prediction model not found:\n"
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        # ------------------------------------------------------------
        # Validate model
        # ------------------------------------------------------------

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            raise TypeError(
                "Loaded model does not support "
                "predict_proba()."
            )

        if type(self.model).__name__ != "XGBClassifier":
            raise TypeError(
                "The localized failure predictor expects "
                "an XGBClassifier.\n"
                f"Loaded model type: "
                f"{type(self.model).__name__}"
            )

        # ------------------------------------------------------------
        # Load feature names
        # ------------------------------------------------------------

        if not self.feature_names_path.exists():
            raise FileNotFoundError(
                "Feature names file not found:\n"
                f"{self.feature_names_path}"
            )

        with open(
            self.feature_names_path,
            "r",
            encoding="utf-8",
        ) as file:

            loaded_features = json.load(
                file
            )

        # Support:
        #
        # ["feature1", "feature2", ...]
        #
        # or:
        #
        # {"feature_names": ["feature1", ...]}
        #

        if isinstance(
            loaded_features,
            dict,
        ):

            if "feature_names" in loaded_features:

                self.feature_names = list(
                    loaded_features[
                        "feature_names"
                    ]
                )

            else:

                raise ValueError(
                    "Feature names JSON dictionary "
                    "does not contain 'feature_names'."
                )

        elif isinstance(
            loaded_features,
            list,
        ):

            self.feature_names = list(
                loaded_features
            )

        else:

            raise ValueError(
                "Unsupported feature_names_localized.json "
                "format."
            )

        # ------------------------------------------------------------
        # Validate feature count
        # ------------------------------------------------------------

        if len(self.feature_names) != 17:

            raise ValueError(
                "Localized model must use exactly "
                "17 features.\n"
                f"Found: {len(self.feature_names)}"
            )

        # ------------------------------------------------------------
        # Validate model feature count
        # ------------------------------------------------------------

        model_feature_count = getattr(
            self.model,
            "n_features_in_",
            None,
        )

        if model_feature_count is not None:

            if int(model_feature_count) != len(
                self.feature_names
            ):

                raise ValueError(
                    "Model feature count does not "
                    "match feature names.\n"
                    f"Model: {model_feature_count}\n"
                    f"Features: {len(self.feature_names)}"
                )

    # ================================================================
    # FEATURE DATAFRAME
    # ================================================================

    def _prepare_feature_dataframe(
        self,
        features: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert extracted features into the exact
        dataframe expected by the trained model.
        """

        missing = [
            feature
            for feature in self.feature_names
            if feature not in features
        ]

        if missing:

            raise ValueError(
                "Missing required ML features:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing
                )
            )

        values = []

        for feature in self.feature_names:

            value = features[feature]

            try:

                value = float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"Feature '{feature}' is not numeric: "
                    f"{value!r}"
                ) from exc

            if not np.isfinite(value):

                raise ValueError(
                    f"Feature '{feature}' is not finite: "
                    f"{value}"
                )

            values.append(value)

        dataframe = pd.DataFrame(
            [values],
            columns=self.feature_names,
        )

        return dataframe

    # ================================================================
    # PREDICTION FROM FEATURES
    # ================================================================

    def predict_from_features(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict failure from an extracted feature dictionary.

        Returns
        -------
        dict
            Prediction probabilities plus native XGBoost
            local feature contributions and explanations.
        """

        dataframe = (
            self._prepare_feature_dataframe(
                features
            )
        )

        # ------------------------------------------------------------
        # Prediction probabilities
        # ------------------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                dataframe
            )[0]
        )

        classes = np.asarray(
            self.model.classes_
        )

        # ------------------------------------------------------------
        # Find class probabilities
        # ------------------------------------------------------------

        success_probability = None
        failure_probability = None

        for (
            class_value,
            probability,
        ) in zip(
            classes,
            probabilities,
        ):

            class_value = int(
                class_value
            )

            if class_value == 0:

                success_probability = float(
                    probability
                )

            elif class_value == 1:

                failure_probability = float(
                    probability
                )

        if success_probability is None:
            success_probability = 0.0

        if failure_probability is None:
            failure_probability = 0.0

        # ------------------------------------------------------------
        # Predicted class
        # ------------------------------------------------------------

        predicted_class_value = int(
            self.model.predict(
                dataframe
            )[0]
        )

        if predicted_class_value == 1:

            predicted_class = "failure"

        else:

            predicted_class = "success"

        # ------------------------------------------------------------
        # Native XGBoost local contributions
        # ------------------------------------------------------------

        feature_contributions = (
            self._get_feature_contributions(
                dataframe
            )
        )

        # One row because one trajectory
        # is being predicted.

        contribution_vector = (
            feature_contributions[0]
        )

        # ------------------------------------------------------------
        # Store contribution by feature
        # ------------------------------------------------------------

        local_contributions = {}

        for (
            feature_name,
            contribution,
        ) in zip(
            self.feature_names,
            contribution_vector,
        ):

            local_contributions[
                feature_name
            ] = float(
                contribution
            )

        # ------------------------------------------------------------
        # Human-readable explanations
        # ------------------------------------------------------------

        explanations = []

        for (
            feature_name,
            contribution,
        ) in zip(
            self.feature_names,
            contribution_vector,
        ):

            value = float(
                dataframe.iloc[
                    0
                ][feature_name]
            )

            contribution = float(
                contribution
            )

            explanations.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "contribution": contribution,
                    "direction": (
                        "failure"
                        if contribution > 0
                        else "success"
                    ),
                    "description": (
                        self._describe_feature(
                            feature_name,
                            value,
                            contribution,
                        )
                    ),
                }
            )

        # ------------------------------------------------------------
        # Most influential features first
        # ------------------------------------------------------------

        explanations.sort(
            key=lambda item: abs(
                item["contribution"]
            ),
            reverse=True,
        )

        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------

        return {
            "predicted_class": predicted_class,
            "predicted_class_value": (
                predicted_class_value
            ),
            "success_probability": (
                success_probability
            ),
            "failure_probability": (
                failure_probability
            ),
            "features": features,
            "feature_dataframe": dataframe,

            # Native XGBoost explanations
            "local_contributions": (
                local_contributions
            ),

            # Compatibility alias
            "feature_contributions": (
                local_contributions
            ),

            # Sorted human-readable explanations
            "explanations": explanations,
        }

    # ================================================================
    # XGBOOST LOCAL CONTRIBUTIONS
    # ================================================================

    def _get_feature_contributions(
        self,
        feature_dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Get local feature contributions using XGBoost's
        native pred_contribs mechanism.

        XGBoost returns:

            17 feature contributions + 1 bias contribution

        The final bias contribution is removed.

        Returns
        -------
        np.ndarray
            Shape: (1, 17)
        """

        booster = (
            self.model.get_booster()
        )

        # ------------------------------------------------------------
        # Convert DataFrame to DMatrix
        # ------------------------------------------------------------

        dmatrix = xgb.DMatrix(
            feature_dataframe,
            feature_names=list(
                feature_dataframe.columns
            ),
        )

        # ------------------------------------------------------------
        # Native XGBoost contributions
        # ------------------------------------------------------------

        contributions = (
            booster.predict(
                dmatrix,
                pred_contribs=True,
            )
        )

        contributions = np.asarray(
            contributions,
            dtype=float,
        )

        # ------------------------------------------------------------
        # Validate dimensions
        # ------------------------------------------------------------

        if contributions.ndim != 2:

            raise ValueError(
                "Unexpected contribution array shape: "
                f"{contributions.shape}"
            )

        expected_width = (
            len(self.feature_names) + 1
        )

        if (
            contributions.shape[1]
            != expected_width
        ):

            raise ValueError(
                "Unexpected number of XGBoost "
                "contributions.\n"
                f"Received: "
                f"{contributions.shape[1]}\n"
                f"Expected: "
                f"{expected_width}\n"
                f"Features: "
                f"{len(self.feature_names)}"
            )

        # ------------------------------------------------------------
        # Remove bias/base-score contribution
        # ------------------------------------------------------------

        feature_contributions = (
            contributions[:, :-1]
        )

        return feature_contributions

    # ================================================================
    # FEATURE DESCRIPTION
    # ================================================================

    def _describe_feature(
        self,
        feature_name: str,
        value: float,
        contribution: float,
    ) -> str:
        """
        Convert a feature and its local contribution into
        a human-readable physical explanation.
        """

        direction = (
            "failure"
            if contribution > 0
            else "success"
        )

        # ------------------------------------------------------------
        # Clearance
        # ------------------------------------------------------------

        if feature_name == (
            "interior_mean_clearance"
        ):

            if value < 0.02:

                return (
                    "Very low average clearance "
                    "suggests that the robot stays "
                    "close to obstacles."
                )

            elif value < 0.05:

                return (
                    "Low average clearance indicates "
                    "that the trajectory passes "
                    "relatively close to obstacles."
                )

            elif value < 0.10:

                return (
                    "Moderate clearance is present "
                    "along the trajectory."
                )

            else:

                return (
                    "High average clearance indicates "
                    "good separation from obstacles."
                )

        if feature_name == (
            "interior_clearance_std"
        ):

            if value > 0.05:

                return (
                    "Large clearance variation indicates "
                    "uneven obstacle proximity."
                )

            else:

                return (
                    "Clearance remains relatively "
                    "consistent along the trajectory."
                )

        if feature_name == (
            "interior_maximum_clearance"
        ):

            return (
                "Maximum interior clearance describes "
                "the largest obstacle separation "
                "along the trajectory."
            )

        if feature_name == (
            "interior_low_clearance_count"
        ):

            if value > 20:

                return (
                    "Many low-clearance configurations "
                    "indicate repeated proximity to "
                    "obstacles."
                )

            elif value > 0:

                return (
                    "Some configurations pass close "
                    "to obstacles."
                )

            else:

                return (
                    "No low-clearance configurations "
                    "were detected."
                )

        if feature_name == (
            "interior_low_clearance_fraction"
        ):

            if value > 0.25:

                return (
                    "A large fraction of the trajectory "
                    "has low obstacle clearance."
                )

            elif value > 0.05:

                return (
                    "Part of the trajectory passes "
                    "through low-clearance regions."
                )

            else:

                return (
                    "Only a small fraction of the "
                    "trajectory has low clearance."
                )

        if feature_name == (
            "number_of_low_clearance_runs"
        ):

            if value > 3:

                return (
                    "Several separate low-clearance "
                    "regions occur along the path."
                )

            elif value > 0:

                return (
                    "The trajectory contains "
                    "low-clearance regions."
                )

            else:

                return (
                    "No low-clearance regions were detected."
                )

        if feature_name == (
            "longest_low_clearance_run"
        ):

            if value > 20:

                return (
                    "A long continuous low-clearance "
                    "segment occurs along the trajectory."
                )

            elif value > 0:

                return (
                    "A short continuous low-clearance "
                    "segment is present."
                )

            else:

                return (
                    "There is no continuous "
                    "low-clearance segment."
                )

        # ------------------------------------------------------------
        # Motion
        # ------------------------------------------------------------

        if feature_name == (
            "joint_space_length"
        ):

            return (
                "Joint-space length describes the "
                "overall amount of joint movement."
            )

        if feature_name == (
            "max_joint_step"
        ):

            return (
                "Maximum joint step measures the "
                "largest configuration change "
                "between consecutive trajectory points."
            )

        if feature_name == (
            "max_joint_velocity"
        ):

            return (
                "Maximum joint velocity indicates "
                "the fastest joint movement."
            )

        if feature_name == (
            "mean_joint_velocity"
        ):

            return (
                "Mean joint velocity represents the "
                "average joint motion rate."
            )

        if feature_name == (
            "max_joint_acceleration"
        ):

            return (
                "Maximum joint acceleration indicates "
                "the strongest change in joint velocity."
            )

        if feature_name == (
            "mean_joint_acceleration"
        ):

            return (
                "Mean joint acceleration represents "
                "the average change in joint velocity."
            )

        # ------------------------------------------------------------
        # Joint limits
        # ------------------------------------------------------------

        if feature_name == (
            "minimum_joint_limit_distance"
        ):

            if value < 0.05:

                return (
                    "At least one joint approaches "
                    "its physical limit closely."
                )

            else:

                return (
                    "The closest joint remains away "
                    "from its physical limit."
                )

        if feature_name == (
            "mean_joint_limit_distance"
        ):

            return (
                "Mean joint-limit distance describes "
                "the average margin from joint limits."
            )

        if feature_name == (
            "minimum_normalized_joint_limit_distance"
        ):

            if value < 0.05:

                return (
                    "A joint operates very close to "
                    "its normalized physical limit."
                )

            else:

                return (
                    "The minimum normalized joint-limit "
                    "margin is reasonably large."
                )

        if feature_name == (
            "mean_normalized_joint_limit_distance"
        ):

            return (
                "This measures the average normalized "
                "distance from joint limits."
            )

        # ------------------------------------------------------------
        # Generic fallback
        # ------------------------------------------------------------

        return (
            f"{feature_name} contributes toward "
            f"{direction} prediction."
        )

    # ================================================================
    # EXPLAIN PREDICTION
    # ================================================================

    def explain_prediction(
        self,
        prediction: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the strongest local feature contributions.

        Parameters
        ----------
        prediction:
            Result returned by predict_from_features().

        top_k:
            Number of strongest explanations to return.
        """

        explanations = prediction.get(
            "explanations",
            [],
        )

        if not explanations:
            return []

        top_k = max(
            1,
            int(top_k),
        )

        return explanations[
            :top_k
        ]

    # ================================================================
    # PREDICT AND EXPLAIN
    # ================================================================

    def predict_and_explain(
        self,
        features: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict and return the strongest explanations.
        """

        prediction = (
            self.predict_from_features(
                features
            )
        )

        prediction[
            "top_explanations"
        ] = self.explain_prediction(
            prediction,
            top_k=top_k,
        )

        return prediction

    # ================================================================
    # PREDICT TRAJECTORY
    # ================================================================

    def predict_trajectory(
        self,
        trajectory,
        feature_extractor,
    ) -> Dict[str, Any]:
        """
        Extract trajectory features and perform prediction.

        Parameters
        ----------
        trajectory:
            Joint trajectory with shape (N, 7).

        feature_extractor:
            TrajectoryFeatureExtractor instance.
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 7
        ):

            raise ValueError(
                "Trajectory must have shape (N, 7). "
                f"Received: {trajectory.shape}"
            )

        features = (
            feature_extractor.extract(
                trajectory
            )
        )

        prediction = (
            self.predict_from_features(
                features
            )
        )

        prediction[
            "trajectory"
        ] = trajectory

        return prediction

    # ================================================================
    # PREDICT MULTIPLE TRAJECTORIES
    # ================================================================

    def predict_multiple(
        self,
        trajectories,
        feature_extractor,
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple trajectories.
        """

        results = []

        for index, trajectory in enumerate(
            trajectories
        ):

            prediction = (
                self.predict_trajectory(
                    trajectory,
                    feature_extractor,
                )
            )

            prediction[
                "trajectory_index"
            ] = index

            results.append(
                prediction
            )

        return results

    # ================================================================
    # MODEL INFORMATION
    # ================================================================

    def get_model_info(
        self,
    ) -> Dict[str, Any]:
        """
        Return information about the loaded model.
        """

        return {
            "model_path": str(
                self.model_path
            ),
            "feature_names_path": str(
                self.feature_names_path
            ),
            "model_type": type(
                self.model
            ).__name__,
            "num_features": len(
                self.feature_names
            ),
            "feature_names": list(
                self.feature_names
            ),
            "model_classes": [
                int(value)
                for value in self.model.classes_
            ],
            "explanation_method": (
                "XGBoost native pred_contribs"
            ),
        }