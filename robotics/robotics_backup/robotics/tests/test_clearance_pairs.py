"""
PyBullet closest-point diagnostic.

Uses the same goal-generation flow as the previously
working clearance test.

Purpose:
1. Generate two genuinely different trajectories.
2. Inspect the actual PyBullet obstacle body IDs.
3. Inspect robot links.
4. Compare closest-point records for different configurations.
"""

import pickle
import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment
from analysis.clearance_analyzer import ClearanceAnalyzer
from analysis.collision_checker import CollisionChecker
from planning.trajectory_generator import TrajectoryGenerator
from planning.goal_configuration import GoalConfigurationGenerator
from planning.obstacle_aware_planner import ObstacleAwarePlanner


print("=" * 70)
print("PYBULLET CLOSEST-POINT DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading MπNets dataset...")

with open(
    "data/mpinets/both_solvable_problems.pkl",
    "rb"
) as f:
    dataset = pickle.load(f)

problem = dataset["tabletop"]["task_oriented"][0]

print("Dataset loaded.")
print("Selected tabletop/task_oriented problem 0.")


# ============================================================
# PYBULLET SETUP
# ============================================================

client_id = p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(
    0,
    0,
    -9.81
)

p.loadURDF(
    "plane.urdf"
)

robot_id = p.loadURDF(
    "franka_panda/panda.urdf",
    useFixedBase=True
)

robot = PandaRobot(robot_id)

print("\nPanda loaded.")
print("Robot ID:", robot_id)
print("DOF:", robot.DOF)


# ============================================================
# ENVIRONMENT
# ============================================================

environment = Environment.from_mpinets_problem(
    problem
)

environment.create_pybullet_obstacles(
    client_id
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


# ============================================================
# START CONFIGURATION
# ============================================================

start_configuration = np.asarray(
    problem.q0,
    dtype=float
)

print("\n" + "=" * 70)
print("START CONFIGURATION")
print("=" * 70)

print(
    np.array2string(
        start_configuration,
        precision=9
    )
)

start_collision = (
    collision_checker.check_configuration(
        start_configuration
    )
)

start_clearance = (
    clearance_analyzer
    .calculate_configuration_clearance(
        start_configuration
    )
)

print(
    "Collision:",
    start_collision["collision"]
)

print(
    "Clearance:",
    start_clearance["minimum_clearance"]
)

print(
    "Closest obstacle:",
    start_clearance[
        "closest_obstacle_index"
    ]
)


# ============================================================
# GOAL GENERATION
# ============================================================

print("\nGenerating goal...")

goal_generator = GoalConfigurationGenerator(
    panda_robot=robot,
    environment=environment,
    client_id=client_id
)

# Use the same solve_multiple() method that was
# already working in the previous test.

goal_solutions = (
    goal_generator.solve_multiple(
        num_solutions=1
    )
)

if not goal_solutions:
    raise RuntimeError(
        "Goal generator did not produce "
        "a valid goal configuration."
    )

goal_configuration = np.asarray(
    goal_solutions[0]["configuration"],
    dtype=float
)

print(
    "Goal:",
    np.array2string(
        goal_configuration,
        precision=9
    )
)

goal_collision = (
    collision_checker.check_configuration(
        goal_configuration
    )
)

goal_clearance = (
    clearance_analyzer
    .calculate_configuration_clearance(
        goal_configuration
    )
)

print(
    "Goal collision:",
    goal_collision["collision"]
)

print(
    "Goal clearance:",
    goal_clearance[
        "minimum_clearance"
    ]
)

print(
    "Goal closest obstacle:",
    goal_clearance[
        "closest_obstacle_index"
    ]
)

if goal_collision["collision"]:
    raise RuntimeError(
        "Goal configuration is in collision."
    )


# ============================================================
# GENERATE CANDIDATES
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

planning_results = (
    planner.plan_multiple(
        start_configuration=start_configuration,
        goal_configuration=goal_configuration,
        num_candidates=5
    )
)

trajectories = [
    result["trajectory"]
    for result in planning_results
    if result.get("success", False)
]

print(
    "Generated trajectories:",
    len(trajectories)
)

if len(trajectories) < 2:
    raise RuntimeError(
        "Need at least two successful "
        "candidate trajectories."
    )


# ============================================================
# TRAJECTORY DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("TRAJECTORY DIFFERENCE")
print("=" * 70)

for i in range(len(trajectories)):

    for j in range(i + 1, len(trajectories)):

        max_difference = np.max(
            np.abs(
                trajectories[i]
                - trajectories[j]
            )
        )

        print(
            f"Candidate {i + 1} vs "
            f"Candidate {j + 1}: "
            f"{max_difference:.9f}"
        )


# ============================================================
# OBSTACLE IDs
# ============================================================

obstacle_ids = (
    clearance_analyzer
    ._get_obstacle_ids()
)

print("\n" + "=" * 70)
print("PYBULLET OBSTACLE IDs")
print("=" * 70)

print(
    "Obstacle IDs:",
    obstacle_ids
)

print(
    "Number of detected obstacle bodies:",
    len(obstacle_ids)
)


# ============================================================
# ALL PYBULLET BODIES
# ============================================================

print("\n" + "=" * 70)
print("ALL PYBULLET BODIES")
print("=" * 70)

num_bodies = p.getNumBodies(
    physicsClientId=client_id
)

print(
    "Total bodies:",
    num_bodies
)

for body_id in range(num_bodies):

    try:

        info = p.getBodyInfo(
            body_id,
            physicsClientId=client_id
        )

        decoded = []

        for item in info:

            if isinstance(item, bytes):

                decoded.append(
                    item.decode(
                        "utf-8",
                        errors="ignore"
                    )
                )

            else:

                decoded.append(
                    str(item)
                )

        print(
            f"Body {body_id}:",
            decoded
        )

    except Exception as error:

        print(
            f"Body {body_id}:",
            "ERROR",
            error
        )


# ============================================================
# ROBOT LINKS
# ============================================================

print("\n" + "=" * 70)
print("ROBOT LINKS")
print("=" * 70)

num_joints = p.getNumJoints(
    robot_id,
    physicsClientId=client_id
)

for joint_index in range(num_joints):

    info = p.getJointInfo(
        robot_id,
        joint_index,
        physicsClientId=client_id
    )

    link_name = info[12]

    if isinstance(
        link_name,
        bytes
    ):
        link_name = link_name.decode(
            "utf-8",
            errors="ignore"
        )

    print(
        f"Joint {joint_index}: "
        f"{link_name}"
    )


# ============================================================
# CLOSEST-POINT INSPECTION
# ============================================================

def inspect_configuration(
    label,
    q,
    obstacle_id
):

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    print(
        "Obstacle body ID:",
        obstacle_id
    )

    print(
        "Configuration:"
    )

    print(
        np.array2string(
            q,
            precision=9
        )
    )

    # --------------------------------------------------------
    # Set robot configuration.
    # --------------------------------------------------------

    for joint_index, joint_value in zip(
        robot.ARM_JOINT_INDICES,
        q
    ):

        p.resetJointState(
            robot_id,
            joint_index,
            float(joint_value),
            targetVelocity=0.0,
            physicsClientId=client_id
        )

    p.performCollisionDetection(
        physicsClientId=client_id
    )

    # --------------------------------------------------------
    # Get closest points.
    # --------------------------------------------------------

    points = p.getClosestPoints(
        bodyA=robot_id,
        bodyB=obstacle_id,
        distance=10.0,
        physicsClientId=client_id
    )

    print(
        "\nNumber of closest-point records:",
        len(points)
    )

    if not points:

        print(
            "No closest points returned."
        )

        return

    points = sorted(
        points,
        key=lambda x: float(x[8])
    )

    print(
        "\nClosest point records:"
    )

    for point in points[:15]:

        link_a = point[3]
        link_b = point[4]
        distance = float(point[8])

        print(
            f"Robot link: {link_a:2d} | "
            f"Obstacle link: {link_b:2d} | "
            f"Distance: {distance:.12f} m"
        )


# ============================================================
# INSPECT OBSTACLE INDEX 4
# ============================================================

if len(obstacle_ids) > 4:

    obstacle_4 = obstacle_ids[4]

    q_candidate_1 = (
        trajectories[0][100]
    )

    q_candidate_2 = (
        trajectories[1][100]
    )

    inspect_configuration(
        "CANDIDATE 1 — INDEX 100",
        q_candidate_1,
        obstacle_4
    )

    inspect_configuration(
        "CANDIDATE 2 — INDEX 100",
        q_candidate_2,
        obstacle_4
    )

    # --------------------------------------------------------
    # Altered configuration control.
    # --------------------------------------------------------

    altered = (
        start_configuration.copy()
    )

    altered[0] += 1.0

    inspect_configuration(
        "ALTERED START — JOINT 1 + 1 RAD",
        altered,
        obstacle_4
    )

else:

    print(
        "\nObstacle index 4 does not exist."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)

p.disconnect(client_id)