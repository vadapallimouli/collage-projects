import pickle

import numpy as np
import pybullet as p
import pybullet_data

from environment import Environment
from robot.panda_robot import PandaRobot
from analysis.collision_checker import CollisionChecker


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("COLLISION CHECKER TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Connect to PyBullet
    # ---------------------------------------------------------

    print("\nConnecting to PyBullet...")

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    # ---------------------------------------------------------
    # Load Panda
    # ---------------------------------------------------------

    print("Loading Panda...")

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id
    )

    panda = PandaRobot(robot_id)

    print(
        f"Panda loaded. ID = {robot_id}"
    )

    # ---------------------------------------------------------
    # Load MπNets dataset
    # ---------------------------------------------------------

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    # ---------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------

    print("Creating Environment...")

    environment = Environment.from_mpinets_problem(
        problem
    )

    # ---------------------------------------------------------
    # Create obstacles
    # ---------------------------------------------------------

    print("Creating obstacles...")

    environment.create_pybullet_obstacles(
        client_id
    )

    print(
        f"Created {environment.num_obstacles} obstacles."
    )

    # ---------------------------------------------------------
    # Create collision checker
    # ---------------------------------------------------------

    print("Creating CollisionChecker...")

    checker = CollisionChecker(
        panda_robot=panda,
        environment=environment,
        client_id=client_id
    )

    # =========================================================
    # TEST 1 — START CONFIGURATION
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 1: MπNets START CONFIGURATION")
    print("-" * 60)

    q_start = environment.start_configuration

    print("Start configuration:")
    print(q_start)

    result = checker.check_configuration(
        q_start
    )

    print("\nResult:")
    print(result)

    assert result["joint_limit_violation"] is False

    print("\nTEST 1 PASSED")

    # =========================================================
    # TEST 2 — JOINT LIMIT VIOLATION
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 2: JOINT LIMIT VIOLATION")
    print("-" * 60)

    q_invalid = q_start.copy()

    # Deliberately violate joint 1 limit.
    q_invalid[0] = 10.0

    result = checker.check_configuration(
        q_invalid
    )

    print("\nInvalid configuration:")
    print(q_invalid)

    print("\nResult:")
    print(result)

    assert result["joint_limit_violation"] is True
    assert result["collision"] is True

    print("\nTEST 2 PASSED")

    # =========================================================
    # TEST 3 — TRAJECTORY
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 3: TRAJECTORY CHECK")
    print("-" * 60)

    # Create a simple trajectory around the start
    # configuration.

    trajectory = np.linspace(
        q_start,
        q_start,
        20
    )

    trajectory_result = checker.check_trajectory(
        trajectory
    )

    print("\nTrajectory shape:")
    print(trajectory.shape)

    print("\nTrajectory result:")

    print(
        "Collision free:",
        trajectory_result["collision_free"]
    )

    print(
        "Collision count:",
        trajectory_result["collision_count"]
    )

    print(
        "Self-collision count:",
        trajectory_result[
            "self_collision_count"
        ]
    )

    print(
        "Obstacle collision count:",
        trajectory_result[
            "obstacle_collision_count"
        ]
    )

    print(
        "Joint-limit violations:",
        trajectory_result[
            "joint_limit_violation_count"
        ]
    )

    assert (
        trajectory_result["num_configurations"]
        == 20
    )

    print("\nTEST 3 PASSED")

    # =========================================================
    # SUCCESS
    # =========================================================

    print("\n" + "=" * 60)
    print("ALL COLLISION CHECKER TESTS PASSED")
    print("=" * 60)

    p.disconnect(
        physicsClientId=client_id
    )


if __name__ == "__main__":
    main()