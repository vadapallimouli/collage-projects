import pickle
import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment

from planning.goal_configuration import (
    GoalConfigurationGenerator
)

from planning.trajectory_generator import (
    TrajectoryGenerator
)


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("DIVERSE TRAJECTORY GENERATOR TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # SETUP
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

    print("Loading Panda...")

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    panda_robot = PandaRobot(
        robot_id
    )

    print("Loading MπNets dataset...")

    with open(
        DATASET_PATH,
        "rb"
    ) as f:

        data = pickle.load(f)

    problem = (
        data[
            "tabletop"
        ][
            "task_oriented"
        ][0]
    )

    print("Creating Environment...")

    environment = (
        Environment.from_mpinets_problem(
            problem
        )
    )

    print("Creating obstacles...")

    environment.create_pybullet_obstacles(
        client_id
    )

    # ---------------------------------------------------------
    # GOAL
    # ---------------------------------------------------------

    print(
        "Generating goal configuration..."
    )

    goal_generator = (
        GoalConfigurationGenerator(
            panda_robot=panda_robot,
            environment=environment,
            client_id=client_id,
        )
    )

    goals = (
        goal_generator.solve_multiple(
            num_solutions=1,
            num_rest_poses=50,
            seed=42,
        )
    )

    if len(goals) == 0:
        raise RuntimeError(
            "No valid goal configuration found."
        )

    goal = goals[0]

    q_start = np.asarray(
        problem.q0,
        dtype=float
    )

    q_goal = np.asarray(
        goal["configuration"],
        dtype=float
    )

    print("\nStart configuration:")
    print(q_start)

    print("\nGoal configuration:")
    print(q_goal)

    # ---------------------------------------------------------
    # GENERATOR
    # ---------------------------------------------------------

    print(
        "\nCreating TrajectoryGenerator..."
    )

    generator = TrajectoryGenerator(
        panda_robot
    )

    # ---------------------------------------------------------
    # TEST 1
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 1: DIVERSE TRAJECTORY GENERATION")
    print("-" * 60)

    trajectories = (
        generator.generate_diverse_trajectories(
            start_configuration=q_start,
            goal_configuration=q_goal,
            num_candidates=10,
            num_points=100,
            seed=42,
            perturbation_scale=0.8,
        )
    )

    print(
        f"\nGenerated trajectories: "
        f"{len(trajectories)}"
    )

    assert len(trajectories) == 10

    for i, trajectory in enumerate(
        trajectories
    ):

        assert trajectory.shape == (
            100,
            7
        )

        assert np.all(
            np.isfinite(
                trajectory
            )
        )

        assert np.allclose(
            trajectory[0],
            q_start,
            atol=1e-8
        )

        assert np.allclose(
            trajectory[-1],
            q_goal,
            atol=1e-8
        )

        for q in trajectory:

            assert (
                panda_robot
                .is_within_joint_limits(q)
            )

        print(
            f"Trajectory {i + 1}: "
            f"shape={trajectory.shape}, "
            f"valid=True"
        )

    print(
        "\nTEST 1 PASSED"
    )

    # ---------------------------------------------------------
    # TEST 2
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 2: TRAJECTORY DIVERSITY")
    print("-" * 60)

    # Compare complete trajectories.
    pairwise_distances = []

    for i in range(
        len(trajectories)
    ):

        for j in range(
            i + 1,
            len(trajectories)
        ):

            distance = np.linalg.norm(
                trajectories[i]
                - trajectories[j]
            )

            pairwise_distances.append(
                distance
            )

    minimum_distance = min(
        pairwise_distances
    )

    maximum_distance = max(
        pairwise_distances
    )

    print(
        f"\nMinimum trajectory distance: "
        f"{minimum_distance:.6f}"
    )

    print(
        f"Maximum trajectory distance: "
        f"{maximum_distance:.6f}"
    )

    assert maximum_distance > 1.0

    print(
        "\nSignificant trajectory diversity confirmed."
    )

    print(
        "\nTEST 2 PASSED"
    )

    # ---------------------------------------------------------
    # TEST 3
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 3: DIFFERENT INTERMEDIATE WAYPOINTS")
    print("-" * 60)

    midpoint_indices = [
        len(trajectory) // 2
        for trajectory in trajectories
    ]

    midpoints = [
        trajectories[i][
            midpoint_indices[i]
        ]
        for i in range(
            len(trajectories)
        )
    ]

    midpoint_distances = []

    for i in range(
        len(midpoints)
    ):

        for j in range(
            i + 1,
            len(midpoints)
        ):

            distance = np.linalg.norm(
                midpoints[i]
                - midpoints[j]
            )

            midpoint_distances.append(
                distance
            )

    max_midpoint_distance = max(
        midpoint_distances
    )

    print(
        f"\nMaximum midpoint distance: "
        f"{max_midpoint_distance:.6f} rad"
    )

    assert max_midpoint_distance > 0.1

    print(
        "\nIntermediate configurations are diverse."
    )

    print(
        "\nTEST 3 PASSED"
    )

    # ---------------------------------------------------------
    # TEST 4
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 4: SMOOTHNESS")
    print("-" * 60)

    for i, trajectory in enumerate(
        trajectories
    ):

        velocity = np.diff(
            trajectory,
            axis=0
        )

        max_step = np.max(
            np.abs(velocity)
        )

        assert np.isfinite(
            max_step
        )

        assert max_step < 0.5

        print(
            f"Trajectory {i + 1}: "
            f"max joint step="
            f"{max_step:.6f} rad"
        )

    print(
        "\nAll diverse trajectories are smooth."
    )

    print(
        "\nTEST 4 PASSED"
    )

    # ---------------------------------------------------------
    # FINISH
    # ---------------------------------------------------------

    p.disconnect(
        client_id
    )

    print("\n" + "=" * 60)
    print(
        "ALL DIVERSE TRAJECTORY TESTS PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()