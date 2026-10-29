from pathlib import Path
import sys

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# TRAJECTORY FEATURE EXTRACTOR
# ============================================================

class TrajectoryFeatureExtractor:
    """
    Extract ML features from robot trajectories.

    The extractor separates:

    1. Diagnostic / ground-truth collision information
    2. Clearance features
    3. Trajectory motion features
    4. Joint-limit features
    5. ML features used by the failure prediction model

    Collision outcome features are intentionally excluded from
    the ML feature vector to avoid target leakage.
    """

    # --------------------------------------------------------
    # ML FEATURES
    # --------------------------------------------------------

    ML_FEATURE_NAMES = [
        # Clearance distribution
        "interior_mean_clearance",
        "interior_clearance_std",
        "interior_maximum_clearance",
        "interior_low_clearance_count",
        "interior_low_clearance_fraction",

        # NEW localized low-clearance features
        "number_of_low_clearance_runs",
        "longest_low_clearance_run",

        # Trajectory motion
        "joint_space_length",
        "max_joint_step",
        "max_joint_velocity",
        "mean_joint_velocity",
        "max_joint_acceleration",
        "mean_joint_acceleration",

        # Joint-limit proximity
        "minimum_joint_limit_distance",
        "mean_joint_limit_distance",
        "minimum_normalized_joint_limit_distance",
        "mean_normalized_joint_limit_distance",
    ]

    LOW_CLEARANCE_THRESHOLD = 0.02

    def __init__(
        self,
        collision_checker,
        clearance_analyzer,
        trajectory_generator,
    ):
        self.collision_checker = collision_checker
        self.clearance_analyzer = clearance_analyzer
        self.trajectory_generator = trajectory_generator

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_trajectory(self, trajectory):

        trajectory = np.asarray(
            trajectory,
            dtype=float
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "Trajectory must have shape (N, 7)."
            )

        if trajectory.shape[1] != 7:
            raise ValueError(
                f"Trajectory must have 7 joints, "
                f"got shape {trajectory.shape}."
            )

        if trajectory.shape[0] < 2:
            raise ValueError(
                "Trajectory must contain at least "
                "2 configurations."
            )

        if not np.all(np.isfinite(trajectory)):
            raise ValueError(
                "Trajectory contains non-finite values."
            )

        return trajectory

    # ========================================================
    # LOW-CLEARANCE RUN ANALYSIS
    # ========================================================

    def _find_low_clearance_runs(
        self,
        clearances
    ):
        """
        Find contiguous regions where clearance is at or below
        the low-clearance threshold.

        Example:

            False False True True True False True True False

        gives:

            runs = [(2, 4), (6, 7)]

        Returns:
            list of dictionaries with:
                start_index
                end_index
                length
        """

        clearances = np.asarray(
            clearances,
            dtype=float
        )

        low_mask = (
            clearances
            <= self.LOW_CLEARANCE_THRESHOLD
        )

        runs = []

        start = None

        for index, is_low in enumerate(low_mask):

            if is_low and start is None:

                start = index

            elif not is_low and start is not None:

                end = index - 1

                runs.append({
                    "start_index": start,
                    "end_index": end,
                    "length": end - start + 1,
                })

                start = None

        # Handle a run reaching the final point
        if start is not None:

            end = len(low_mask) - 1

            runs.append({
                "start_index": start,
                "end_index": end,
                "length": end - start + 1,
            })

        return runs

    # ========================================================
    # COLLISION FEATURES
    # ========================================================

    def _extract_collision_features(
        self,
        trajectory
    ):

        result = self.collision_checker.check_trajectory(
            trajectory
        )

        features = {
            # Ground-truth / diagnostic fields
            "collision": int(
                not result["collision_free"]
            ),

            "collision_free": int(
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
        }

        return features

    # ========================================================
    # CLEARANCE FEATURES
    # ========================================================

    def _extract_clearance_features(
        self,
        trajectory
    ):

        result = (
            self.clearance_analyzer
            .calculate_trajectory_clearance(
                trajectory
            )
        )

        features = {
            "minimum_clearance": float(
                result["minimum_clearance"]
            ),

            "interior_minimum_clearance": float(
                result["interior_minimum_clearance"]
            ),

            "interior_mean_clearance": float(
                result["interior_mean_clearance"]
            ),

            "interior_clearance_std": float(
                result["interior_clearance_std"]
            ),

            "interior_maximum_clearance": float(
                result["interior_maximum_clearance"]
            ),

            "interior_low_clearance_count": int(
                result["interior_low_clearance_count"]
            ),

            "interior_low_clearance_fraction": float(
                result["interior_low_clearance_fraction"]
            ),

            # Diagnostic metadata
            "closest_obstacle_index": result.get(
                "closest_obstacle_index"
            ),

            "closest_robot_link": result.get(
                "closest_robot_link"
            ),
        }

        # ----------------------------------------------------
        # NEW:
        #
        # Reconstruct the actual clearance sequence so we can
        # describe how low-clearance points are distributed.
        # ----------------------------------------------------

        clearances = []

        for configuration in trajectory:

            configuration_result = (
                self.clearance_analyzer
                .calculate_configuration_clearance(
                    configuration
                )
            )

            clearances.append(
                float(
                    configuration_result[
                        "minimum_clearance"
                    ]
                )
            )

        clearances = np.asarray(
            clearances,
            dtype=float
        )

        # ----------------------------------------------------
        # Find contiguous low-clearance regions
        # ----------------------------------------------------

        low_clearance_runs = (
            self._find_low_clearance_runs(
                clearances
            )
        )

        features[
            "number_of_low_clearance_runs"
        ] = int(
            len(low_clearance_runs)
        )

        if low_clearance_runs:

            features[
                "longest_low_clearance_run"
            ] = int(
                max(
                    run["length"]
                    for run in low_clearance_runs
                )
            )

        else:

            features[
                "longest_low_clearance_run"
            ] = 0

        return features

    # ========================================================
    # TRAJECTORY MOTION FEATURES
    # ========================================================

    def _extract_motion_features(
        self,
        trajectory
    ):

        joint_space_length = (
            self.trajectory_generator
            .calculate_joint_space_length(
                trajectory
            )
        )

        max_joint_step = (
            self.trajectory_generator
            .calculate_max_joint_step(
                trajectory
            )
        )

        velocities = (
            self.trajectory_generator
            .calculate_joint_velocities(
                trajectory
            )
        )

        accelerations = (
            self.trajectory_generator
            .calculate_joint_accelerations(
                trajectory
            )
        )

        features = {
            "joint_space_length": float(
                joint_space_length
            ),

            "max_joint_step": float(
                max_joint_step
            ),

            "max_joint_velocity": float(
                np.max(
                    np.abs(velocities)
                )
            ),

            "mean_joint_velocity": float(
                np.mean(
                    np.abs(velocities)
                )
            ),

            "max_joint_acceleration": float(
                np.max(
                    np.abs(accelerations)
                )
            ),

            "mean_joint_acceleration": float(
                np.mean(
                    np.abs(accelerations)
                )
            ),
        }

        return features

    # ========================================================
    # JOINT-LIMIT FEATURES
    # ========================================================

    def _extract_joint_limit_features(
        self,
        trajectory
    ):

        joint_limits = np.asarray(
            self.trajectory_generator.joint_limits,
            dtype=float
        )

        lower_limits = joint_limits[:, 0]
        upper_limits = joint_limits[:, 1]

        # Distance from every configuration to the nearest
        # joint limit.
        lower_distance = (
            trajectory
            - lower_limits
        )

        upper_distance = (
            upper_limits
            - trajectory
        )

        joint_limit_distances = np.minimum(
            lower_distance,
            upper_distance
        )

        minimum_joint_limit_distance = float(
            np.min(
                joint_limit_distances
            )
        )

        mean_joint_limit_distance = float(
            np.mean(
                joint_limit_distances
            )
        )

        # ----------------------------------------------------
        # Normalize distance by each joint's available range.
        # ----------------------------------------------------

        joint_ranges = (
            upper_limits
            - lower_limits
        )

        normalized_distances = (
            joint_limit_distances
            / joint_ranges
        )

        minimum_normalized_joint_limit_distance = float(
            np.min(
                normalized_distances
            )
        )

        mean_normalized_joint_limit_distance = float(
            np.mean(
                normalized_distances
            )
        )

        features = {
            "minimum_joint_limit_distance":
                minimum_joint_limit_distance,

            "mean_joint_limit_distance":
                mean_joint_limit_distance,

            "minimum_normalized_joint_limit_distance":
                minimum_normalized_joint_limit_distance,

            "mean_normalized_joint_limit_distance":
                mean_normalized_joint_limit_distance,
        }

        return features

    # ========================================================
    # SINGLE TRAJECTORY
    # ========================================================

    def extract(
        self,
        trajectory,
        include_label=True
    ):

        trajectory = self._validate_trajectory(
            trajectory
        )

        features = {}

        # ----------------------------------------------------
        # Collision information
        # ----------------------------------------------------

        collision_features = (
            self._extract_collision_features(
                trajectory
            )
        )

        features.update(
            collision_features
        )

        # ----------------------------------------------------
        # Clearance
        # ----------------------------------------------------

        clearance_features = (
            self._extract_clearance_features(
                trajectory
            )
        )

        features.update(
            clearance_features
        )

        # ----------------------------------------------------
        # Motion
        # ----------------------------------------------------

        motion_features = (
            self._extract_motion_features(
                trajectory
            )
        )

        features.update(
            motion_features
        )

        # ----------------------------------------------------
        # Joint limits
        # ----------------------------------------------------

        joint_limit_features = (
            self._extract_joint_limit_features(
                trajectory
            )
        )

        features.update(
            joint_limit_features
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        features["num_points"] = int(
            len(trajectory)
        )

        # ----------------------------------------------------
        # Ground-truth failure label
        #
        # IMPORTANT:
        #
        # This is derived from actual collision checking.
        # It is NOT an ML input.
        # ----------------------------------------------------

        if include_label:

            features["failure"] = int(
                features["collision"] == 1
            )

        return features

    # ========================================================
    # MULTIPLE TRAJECTORIES
    # ========================================================

    def extract_multiple(
        self,
        trajectories,
        include_label=True
    ):

        results = []

        for index, trajectory in enumerate(
            trajectories
        ):

            features = self.extract(
                trajectory,
                include_label=include_label
            )

            features["trajectory_index"] = (
                index
            )

            results.append(
                features
            )

        return results

    # ========================================================
    # ML FEATURE NAMES
    # ========================================================

    def get_feature_names(self):

        return list(
            self.ML_FEATURE_NAMES
        )

    # ========================================================
    # ML FEATURE VECTOR
    # ========================================================

    def get_ml_feature_vector(
        self,
        features
    ):

        return np.asarray(
            [
                features[name]
                for name in self.ML_FEATURE_NAMES
            ],
            dtype=float
        )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

FeatureExtractor = TrajectoryFeatureExtractor