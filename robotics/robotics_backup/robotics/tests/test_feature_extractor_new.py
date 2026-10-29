from pathlib import Path
import sys
import pickle
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT COMPONENTS
# ============================================================

from robot.panda_robot import PandaRobot
from environment.environment import Environment
from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer
from planning.trajectory_generator import TrajectoryGenerator
from analysis.feature_extractor import TrajectoryFeatureExtractor

import pybullet as p
import pybullet_data


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)


problem = dataset[
    "tabletop"
]["task_oriented"][0]


# ============================================================
# START PYBULLET
# ============================================================

client_id = p.connect(p.DIRECT)

try:

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.loadURDF(
        "plane.urdf"
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True
    )

    robot = PandaRobot(robot_id)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = Environment.from_mpinets_problem(
        problem
    )

    environment.create_pybullet_obstacles(
        client_id
    )

    # --------------------------------------------------------
    # Analysis components
    # --------------------------------------------------------

    collision_checker = CollisionChecker(
        panda_robot=robot,
        environment=environment,
        client_id=client_id
    )

    clearance_analyzer = ClearanceAnalyzer(
        panda_robot=robot,
        environment=environment,
        client_id=client_id
    )

    trajectory_generator = TrajectoryGenerator(
        panda_robot=robot
    )

    feature_extractor = TrajectoryFeatureExtractor(
        collision_checker=collision_checker,
        clearance_analyzer=clearance_analyzer,
        trajectory_generator=trajectory_generator
    )

    # ========================================================
    # CREATE SIMPLE TEST TRAJECTORY
    # ========================================================

    start_configuration = np.asarray(
        problem.q0,
        dtype=float
    )

    goal_configuration = np.asarray(
        [
            0.969308251,
            0.776347574,
            -2.325633062,
            -2.457559241,
            -0.022686111,
            2.261069429,
            -2.341595461,
        ],
        dtype=float
    )

    trajectory = (
        trajectory_generator
        .generate_joint_trajectory(
            start_configuration,
            goal_configuration,
            num_points=50
        )
    )

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    print("=" * 70)
    print("TESTING FEATURE EXTRACTOR")
    print("=" * 70)

    print(
        f"Trajectory shape: {trajectory.shape}"
    )

    features = feature_extractor.extract(
        trajectory
    )

    # ========================================================
    # PRINT IMPORTANT FEATURES
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE RESULTS")
    print("=" * 70)

    print(
        f"Total features extracted: "
        f"{len(features)}"
    )

    print(
        f"\nNumber of low-clearance runs: "
        f"{features['number_of_low_clearance_runs']}"
    )

    print(
        f"Longest low-clearance run: "
        f"{features['longest_low_clearance_run']}"
    )

    print(
        f"Low-clearance count: "
        f"{features['interior_low_clearance_count']}"
    )

    print(
        f"Low-clearance fraction: "
        f"{features['interior_low_clearance_fraction']:.6f}"
    )

    print(
        f"\nCollision: "
        f"{features['collision']}"
    )

    print(
        f"Collision count: "
        f"{features['collision_count']}"
    )

    print(
        f"Joint-space length: "
        f"{features['joint_space_length']:.6f}"
    )

    # ========================================================
    # ML FEATURE LIST
    # ========================================================

    print("\n" + "=" * 70)
    print("ML FEATURE NAMES")
    print("=" * 70)

    feature_names = (
        feature_extractor
        .get_feature_names()
    )

    print(
        f"Number of ML features: "
        f"{len(feature_names)}"
    )

    for index, name in enumerate(
        feature_names,
        start=1
    ):
        print(
            f"{index:2d}. {name}"
        )

    # ========================================================
    # ML VECTOR
    # ========================================================

    ml_vector = (
        feature_extractor
        .get_ml_feature_vector(
            features
        )
    )

    print("\n" + "=" * 70)
    print("ML FEATURE VECTOR")
    print("=" * 70)

    print(
        f"Shape: {ml_vector.shape}"
    )

    print(
        ml_vector
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    assert (
        len(feature_names) == 17
    ), (
        "Expected 17 ML features."
    )

    assert (
        ml_vector.shape == (17,)
    ), (
        "ML vector must have shape (17,)."
    )

    assert (
        "number_of_low_clearance_runs"
        in features
    )

    assert (
        "longest_low_clearance_run"
        in features
    )

    assert (
        features[
            "number_of_low_clearance_runs"
        ] >= 0
    )

    assert (
        features[
            "longest_low_clearance_run"
        ] >= 0
    )

    assert np.all(
        np.isfinite(ml_vector)
    )

    print(
        "All feature extractor checks PASSED."
    )

finally:

    p.disconnect(client_id)