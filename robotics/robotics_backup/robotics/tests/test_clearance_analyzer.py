import pickle

import numpy as np
import pybullet as p
import pybullet_data

from environment import Environment
from robot.panda_robot import PandaRobot
from analysis.clearance_analyzer import ClearanceAnalyzer


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("CLEARANCE ANALYZER TEST")
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

    # ---------------------------------------------------------
    # Load MπNets
    # ---------------------------------------------------------

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    # ---------------------------------------------------------
    # Environment
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
    # Clearance analyzer
    # ---------------------------------------------------------

    print("Creating ClearanceAnalyzer...")

    analyzer = ClearanceAnalyzer(
        panda_robot=panda,
        environment=environment,
        client_id=client_id
    )

    # =========================================================
    # TEST 1 — SINGLE CONFIGURATION
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 1: SINGLE CONFIGURATION")
    print("-" * 60)

    q_start = environment.start_configuration

    result = (
        analyzer.calculate_configuration_clearance(
            q_start
        )
    )

    print("\nStart configuration:")
    print(q_start)

    print("\nClearance result:")

    print(
        "Minimum clearance:",
        result["minimum_clearance"]
    )

    print(
        "Closest obstacle index:",
        result["closest_obstacle_index"]
    )

    print(
        "Closest obstacle ID:",
        result["closest_obstacle_id"]
    )

    assert result["valid"] is True

    assert np.isfinite(
        result["minimum_clearance"]
    )

    print("\nTEST 1 PASSED")

    # =========================================================
    # TEST 2 — TRAJECTORY
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 2: TRAJECTORY CLEARANCE")
    print("-" * 60)

    trajectory = np.linspace(
        q_start,
        q_start,
        20
    )

    trajectory_result = (
        analyzer.calculate_trajectory_clearance(
            trajectory
        )
    )

    print("\nTrajectory shape:")
    print(trajectory.shape)

    print(
        "\nMinimum trajectory clearance:",
        trajectory_result[
            "minimum_clearance"
        ]
    )

    print(
        "Configuration with minimum clearance:",
        trajectory_result[
            "minimum_clearance_configuration"
        ]
    )

    print(
        "Closest obstacle:",
        trajectory_result[
            "closest_obstacle_index"
        ]
    )

    print(
        "\nAll configuration clearances:"
    )

    print(
        trajectory_result["clearances"]
    )

    assert (
        trajectory_result["num_configurations"]
        == 20
    )

    assert (
        trajectory_result["clearances"].shape
        == (20,)
    )

    assert np.all(
        np.isfinite(
            trajectory_result["clearances"]
        )
    )

    print("\nTEST 2 PASSED")

    # =========================================================
    # SUCCESS
    # =========================================================

    print("\n" + "=" * 60)
    print("ALL CLEARANCE ANALYZER TESTS PASSED")
    print("=" * 60)

    p.disconnect(
        physicsClientId=client_id
    )


if __name__ == "__main__":
    main()