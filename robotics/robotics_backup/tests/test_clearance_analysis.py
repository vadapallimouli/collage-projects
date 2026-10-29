"""
Diagnostic test for candidate trajectory clearance.

Purpose:
1. Generate multiple candidate trajectories.
2. Verify that the trajectories are actually different.
3. Compare selected configurations between candidates.
4. Calculate clearance for individual configurations.
5. Determine exactly where the identical clearance originates.
"""

import pickle
import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment
from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer
from planning.trajectory_generator import TrajectoryGenerator
from planning.goal_configuration import GoalConfigurationGenerator
from planning.obstacle_aware_planner import ObstacleAwarePlanner


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("CANDIDATE TRAJECTORY / CLEARANCE DIAGNOSTIC")
print("=" * 70)

print("\nLoading MπNets dataset...")

DATASET_PATH = "data/mpinets/both_solvable_problems.pkl"

with open(DATASET_PATH, "rb") as f:
    dataset = pickle.load(f)

problem = dataset["tabletop"]["task_oriented"][0]

print("Dataset loaded.")
print("Selected tabletop/task_oriented problem 0.")


# ============================================================
# PYBULLET
# ============================================================

physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(0, 0, -9.81)

p.loadURDF(
    "plane.urdf"
)

robot_id = p.loadURDF(
    "franka_panda/panda.urdf",
    useFixedBase=True
)

robot = PandaRobot(robot_id)

print("\nPanda loaded.")
print("DOF:", robot.DOF)


# ============================================================
# ENVIRONMENT
# ============================================================

environment = Environment.from_mpinets_problem(
    problem
)

environment.create_pybullet_obstacles(
    physics_client
)

print("\nEnvironment created.")
print(
    "Number of obstacles:",
    environment.num_obstacles
)


# ============================================================
# ANALYSIS OBJECTS
# ============================================================

collision_checker = CollisionChecker(
    panda_robot=robot,
    environment=environment,
    client_id=physics_client
)

clearance_analyzer = ClearanceAnalyzer(
    panda_robot=robot,
    environment=environment,
    client_id=physics_client
)

trajectory_generator = TrajectoryGenerator(
    panda_robot=robot
)


# ============================================================
# START
# ============================================================

start_configuration = np.asarray(
    problem.q0,
    dtype=float
)

print("\n" + "=" * 70)
print("START CONFIGURATION")
print("=" * 70)

print(
    "Start:",
    np.array2string(
        start_configuration,
        precision=6
    )
)

start_clearance = (
    clearance_analyzer.calculate_configuration_clearance(
        start_configuration
    )
)

print(
    "Start clearance:",
    start_clearance["minimum_clearance"]
)

print(
    "Start closest obstacle:",
    start_clearance["closest_obstacle_index"]
)


# ============================================================
# GOAL
# ============================================================

print("\nGenerating goal...")

goal_generator = GoalConfigurationGenerator(
    panda_robot=robot,
    environment=environment,
    client_id=physics_client
)

goal_result = goal_generator.solve()

if isinstance(goal_result, dict):

    goal_configuration = np.asarray(
        goal_result["configuration"],
        dtype=float
    )

else:

    goal_configuration = np.asarray(
        goal_result[0]["configuration"],
        dtype=float
    )

print(
    "Goal:",
    np.array2string(
        goal_configuration,
        precision=6
    )
)

goal_clearance = (
    clearance_analyzer.calculate_configuration_clearance(
        goal_configuration
    )
)

print(
    "Goal clearance:",
    goal_clearance["minimum_clearance"]
)

print(
    "Goal closest obstacle:",
    goal_clearance["closest_obstacle_index"]
)


# ============================================================
# PLANNER
# ============================================================

print("\n" + "=" * 70)
print("GENERATING CANDIDATES")
print("=" * 70)

planner = ObstacleAwarePlanner(
    panda_robot=robot,
    collision_checker=collision_checker,
    trajectory_generator=trajectory_generator,
    num_samples=700,
    k_neighbors=15,
    edge_points=25,
    seed=42
)

num_candidates = 5

planning_results = planner.plan_multiple(
    start_configuration=start_configuration,
    goal_configuration=goal_configuration,
    num_candidates=num_candidates
)

candidates = [
    result["trajectory"]
    for result in planning_results
    if result.get("success", False)
]

print(
    "\nGenerated candidates:",
    len(candidates)
)


# ============================================================
# BASIC TRAJECTORY CHECK
# ============================================================

print("\n" + "=" * 70)
print("TRAJECTORY DIFFERENCE CHECK")
print("=" * 70)

for i, trajectory in enumerate(candidates):

    print(
        f"\nCandidate {i + 1}:"
    )

    print(
        "Shape:",
        trajectory.shape
    )

    print(
        "Start:",
        np.array2string(
            trajectory[0],
            precision=6
        )
    )

    print(
        "Middle:",
        np.array2string(
            trajectory[len(trajectory) // 2],
            precision=6
        )
    )

    print(
        "End:",
        np.array2string(
            trajectory[-1],
            precision=6
        )
    )

    print(
        "Length:",
        trajectory.shape[0]
    )


# ============================================================
# COMPARE CANDIDATES DIRECTLY
# ============================================================

print("\n" + "=" * 70)
print("PAIRWISE TRAJECTORY DIFFERENCES")
print("=" * 70)

for i in range(len(candidates)):

    for j in range(i + 1, len(candidates)):

        difference = np.abs(
            candidates[i] - candidates[j]
        )

        max_difference = np.max(
            difference
        )

        mean_difference = np.mean(
            difference
        )

        print(
            f"Candidate {i + 1} vs "
            f"Candidate {j + 1}:"
        )

        print(
            f"  Maximum joint difference: "
            f"{max_difference:.9f}"
        )

        print(
            f"  Mean joint difference: "
            f"{mean_difference:.9f}"
        )


# ============================================================
# INDIVIDUAL CONFIGURATION CLEARANCE
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL CONFIGURATION CLEARANCE")
print("=" * 70)

test_indices = [
    1,
    25,
    50,
    100,
    150,
    198
]

for trajectory_index, trajectory in enumerate(
    candidates
):

    print(
        f"\nCandidate {trajectory_index + 1}"
    )

    print("-" * 50)

    for index in test_indices:

        q = trajectory[index]

        clearance = (
            clearance_analyzer
            .calculate_configuration_clearance(q)
        )

        print(
            f"Configuration {index:3d}: "
            f"clearance = "
            f"{clearance['minimum_clearance']:.9f} m, "
            f"obstacle = "
            f"{clearance['closest_obstacle_index']}"
        )


# ============================================================
# CONFIGURATION DIFFERENCE + CLEARANCE
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION VS CLEARANCE DIAGNOSTIC")
print("=" * 70)

if len(candidates) >= 2:

    q1 = candidates[0][100]
    q2 = candidates[1][100]

    print("\nCandidate 1 configuration at index 100:")

    print(
        np.array2string(
            q1,
            precision=9
        )
    )

    print("\nCandidate 2 configuration at index 100:")

    print(
        np.array2string(
            q2,
            precision=9
        )
    )

    configuration_difference = np.max(
        np.abs(q1 - q2)
    )

    print(
        "\nMaximum configuration difference:",
        configuration_difference
    )

    clearance_1 = (
        clearance_analyzer
        .calculate_configuration_clearance(q1)
    )

    clearance_2 = (
        clearance_analyzer
        .calculate_configuration_clearance(q2)
    )

    print(
        "\nCandidate 1 clearance:",
        clearance_1["minimum_clearance"]
    )

    print(
        "Candidate 2 clearance:",
        clearance_2["minimum_clearance"]
    )

    print(
        "\nCandidate 1 closest obstacle:",
        clearance_1["closest_obstacle_index"]
    )

    print(
        "Candidate 2 closest obstacle:",
        clearance_2["closest_obstacle_index"]
    )


# ============================================================
# FULL TRAJECTORY CLEARANCE
# ============================================================

print("\n" + "=" * 70)
print("FULL TRAJECTORY CLEARANCE")
print("=" * 70)

for i, trajectory in enumerate(candidates):

    result = (
        clearance_analyzer
        .calculate_trajectory_clearance(
            trajectory
        )
    )

    print(
        f"\nCandidate {i + 1}"
    )

    print(
        "Full minimum:",
        f"{result['minimum_clearance']:.9f} m"
    )

    print(
        "Interior minimum:",
        f"{result['interior_minimum_clearance']:.9f} m"
    )

    print(
        "Mean interior:",
        f"{result['interior_mean_clearance']:.9f} m"
    )

    print(
        "Interior std:",
        f"{result['interior_clearance_std']:.9f} m"
    )


# ============================================================
# ALTERED CONFIGURATION CONTROL TEST
# ============================================================

print("\n" + "=" * 70)
print("CONTROL TEST")
print("=" * 70)

original = start_configuration.copy()

altered = start_configuration.copy()

altered[0] += 1.0

original_result = (
    clearance_analyzer
    .calculate_configuration_clearance(
        original
    )
)

altered_result = (
    clearance_analyzer
    .calculate_configuration_clearance(
        altered
    )
)

print(
    "Original q1:",
    original[0]
)

print(
    "Altered q1:",
    altered[0]
)

print(
    "Original clearance:",
    original_result[
        "minimum_clearance"
    ]
)

print(
    "Altered clearance:",
    altered_result[
        "minimum_clearance"
    ]
)


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSTIC")
print("=" * 70)

if len(candidates) >= 2:

    max_difference = np.max(
        np.abs(
            candidates[0]
            - candidates[1]
        )
    )

    if max_difference < 1e-8:

        print(
            "PROBLEM FOUND:"
        )

        print(
            "Candidate trajectories are "
            "numerically identical."
        )

    else:

        print(
            "Candidate trajectories ARE different."
        )

        print(
            "Therefore the problem is inside "
            "the clearance calculation or "
            "PyBullet geometry update."
        )

else:

    print(
        "Not enough candidate trajectories "
        "for comparison."
    )


print(
    "\nDIAGNOSTIC COMPLETED."
)


# ============================================================
# CLEANUP
# ============================================================

p.disconnect(physics_client)