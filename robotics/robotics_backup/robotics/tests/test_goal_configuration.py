import pickle

import numpy as np
import pybullet as p
import pybullet_data

from environment import Environment
from robot.panda_robot import PandaRobot
from planning.goal_configuration import (
    GoalConfigurationGenerator
)


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("GOAL CONFIGURATION TEST")
    print("=" * 60)

    # =========================================================
    # Connect to PyBullet
    # =========================================================

    print("\nConnecting to PyBullet...")

    client_id = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    # =========================================================
    # Load Panda
    # =========================================================

    print("Loading Panda...")

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id
    )

    panda_robot = PandaRobot(robot_id)

    # =========================================================
    # Load MπNets dataset
    # =========================================================

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    # =========================================================
    # Create Environment
    # =========================================================

    print("Creating Environment...")

    environment = Environment.from_mpinets_problem(
        problem
    )

    # =========================================================
    # Create PyBullet obstacles
    # =========================================================

    print("Creating obstacles...")

    environment.create_pybullet_obstacles(
        client_id
    )

    # =========================================================
    # Create GoalConfigurationGenerator
    # =========================================================

    print(
        "Creating GoalConfigurationGenerator..."
    )

    generator = GoalConfigurationGenerator(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id
    )

    # =========================================================
    # TEST 1 — SINGLE GOAL
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 1: SINGLE GOAL CONFIGURATION")
    print("-" * 60)

    result = generator.solve()

    print("\nResult:")

    if result is None:

        print(
            "No valid goal configuration found."
        )

    else:

        print(
            "Valid goal configuration found."
        )

        print("\nJoint configuration:")

        print(
            result["configuration"]
        )

        print("\nEnd-effector position:")

        print(
            result["position"]
        )

        print("\nDistance from target center:")

        print(
            f'{result["position_error"]:.8f} m'
        )

        print("\nInside target volume:")

        print(
            generator.is_inside_target_volume(
                result["position"]
            )
        )

    # ---------------------------------------------------------
    # Validate single goal
    # ---------------------------------------------------------

    assert result is not None, (
        "No valid single goal configuration was generated."
    )

    q = result["configuration"]
    position = result["position"]

    assert q.shape == (7,), (
        f"Expected 7-DOF configuration, got {q.shape}"
    )

    assert np.all(np.isfinite(q)), (
        "Goal configuration contains non-finite values."
    )

    assert panda_robot.is_within_joint_limits(q), (
        "Goal configuration violates joint limits."
    )

    assert generator.is_inside_target_volume(
        position
    ), (
        "Goal end-effector position is outside "
        "the target volume."
    )

    print("\nTEST 1 PASSED")

    # =========================================================
    # TEST 2 — MULTIPLE GOALS
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 2: MULTIPLE GOAL CONFIGURATIONS")
    print("-" * 60)

    solutions = generator.solve_multiple(
        num_solutions=10,
        num_rest_poses=50,
        seed=42
    )

    print(
        "\nNumber of valid goal configurations:"
    )

    print(
        len(solutions)
    )

    # ---------------------------------------------------------
    # Display all solutions
    # ---------------------------------------------------------

    for i, solution in enumerate(
        solutions,
        start=1
    ):

        print(
            f"\nGoal {i}:"
        )

        print(
            "  Joint configuration:"
        )

        print(
            " ",
            solution["configuration"]
        )

        print(
            "  End-effector position:"
        )

        print(
            " ",
            solution["position"]
        )

        print(
            "  Position error:"
        )

        print(
            f'  {solution["position_error"]:.8f} m'
        )

        print(
            "  Inside target volume:"
        )

        print(
            " ",
            generator.is_inside_target_volume(
                solution["position"]
            )
        )

    # ---------------------------------------------------------
    # Validate multiple solutions
    # ---------------------------------------------------------

    assert len(solutions) > 0, (
        "No valid goal configurations were generated."
    )

    for solution in solutions:

        q = solution["configuration"]
        position = solution["position"]

        assert q.shape == (7,), (
            "Goal configuration is not 7-DOF."
        )

        assert np.all(np.isfinite(q)), (
            "Goal configuration contains non-finite values."
        )

        assert panda_robot.is_within_joint_limits(q), (
            "Goal configuration violates joint limits."
        )

        assert generator.is_inside_target_volume(
            position
        ), (
            "Goal position is outside target volume."
        )

    print(
        "\nTEST 2 PASSED"
    )

    # =========================================================
    # TEST 3 — GOAL DIVERSITY
    # =========================================================

    print("\n" + "-" * 60)
    print("TEST 3: GOAL DIVERSITY")
    print("-" * 60)

    if len(solutions) >= 2:

        distances = []

        for i in range(
            len(solutions)
        ):

            for j in range(
                i + 1,
                len(solutions)
            ):

                q1 = solutions[i][
                    "configuration"
                ]

                q2 = solutions[j][
                    "configuration"
                ]

                distance = np.linalg.norm(
                    q1 - q2
                )

                distances.append(
                    distance
                )

        max_distance = max(
            distances
        )

        min_distance = min(
            distances
        )

        print(
            "\nMinimum joint-space distance:"
        )

        print(
            f"{min_distance:.6f}"
        )

        print(
            "\nMaximum joint-space distance:"
        )

        print(
            f"{max_distance:.6f}"
        )

        assert max_distance > 1e-3, (
            "Goal configurations are not "
            "sufficiently different."
        )

        print(
            "\nMultiple distinct goal configurations "
            "confirmed."
        )

    else:

        print(
            "\nOnly one valid goal configuration "
            "was generated."
        )

        print(
            "The generator itself is still valid, "
            "but diversity could not be demonstrated."
        )

    print(
        "\nTEST 3 PASSED"
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    print("\n" + "=" * 60)
    print("ALL GOAL CONFIGURATION TESTS PASSED")
    print("=" * 60)

    # =========================================================
    # Cleanup
    # =========================================================

    p.disconnect(
        client_id
    )


if __name__ == "__main__":
    main()