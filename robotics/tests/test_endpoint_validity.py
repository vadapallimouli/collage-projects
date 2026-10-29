import pickle
import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment
from planning.goal_configuration import GoalConfigurationGenerator
from analysis.collision_checker import CollisionChecker
from analysis.clearance_analyzer import ClearanceAnalyzer


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("ENDPOINT VALIDITY TEST")
    print("=" * 60)

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    panda_robot = PandaRobot(robot_id)

    print("\nLoading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    environment = Environment.from_mpinets_problem(
        problem
    )

    environment.create_pybullet_obstacles(
        client_id
    )

    collision_checker = CollisionChecker(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    clearance_analyzer = ClearanceAnalyzer(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    # ---------------------------------------------------------
    # START
    # ---------------------------------------------------------

    q_start = np.asarray(
        problem.q0,
        dtype=float
    )

    print("\n" + "-" * 60)
    print("START CONFIGURATION")
    print("-" * 60)

    print("\nq_start:")
    print(q_start)

    start_collision = (
        collision_checker.check_configuration(
            q_start
        )
    )

    start_clearance = (
        clearance_analyzer.calculate_configuration_clearance(
            q_start
        )
    )

    print("\nCollision result:")
    print(start_collision)

    print("\nClearance result:")
    print(start_clearance)

    # ---------------------------------------------------------
    # GOAL
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("GENERATED GOAL CONFIGURATION")
    print("-" * 60)

    goal_generator = GoalConfigurationGenerator(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    goals = goal_generator.solve_multiple(
        num_solutions=10,
        num_rest_poses=50,
        seed=42,
    )

    print(
        f"\nGenerated {len(goals)} goals."
    )

    for i, goal in enumerate(goals):

        q_goal = np.asarray(
            goal["configuration"],
            dtype=float
        )

        collision = (
            collision_checker.check_configuration(
                q_goal
            )
        )

        clearance = (
            clearance_analyzer.calculate_configuration_clearance(
                q_goal
            )
        )

        print(
            f"\nGoal {i + 1}"
        )

        print(
            f"Position error: "
            f"{goal['position_error']:.6f} m"
        )

        print(
            f"Collision: "
            f"{collision['collision']}"
        )

        print(
            f"Obstacle collision: "
            f"{collision['obstacle_collision']}"
        )

        print(
            f"Self collision: "
            f"{collision['self_collision']}"
        )

        print(
            f"Minimum clearance: "
            f"{clearance['minimum_clearance']:.6f} m"
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("ENDPOINT VALIDITY SUMMARY")
    print("=" * 60)

    print(
        "\nStart collision:",
        start_collision["collision"]
    )

    print(
        "Start clearance:",
        f"{start_clearance['minimum_clearance']:.6f} m"
    )

    collision_free_goals = 0

    for goal in goals:

        q_goal = np.asarray(
            goal["configuration"],
            dtype=float
        )

        result = (
            collision_checker.check_configuration(
                q_goal
            )
        )

        if not result["collision"]:
            collision_free_goals += 1

    print(
        "\nCollision-free generated goals:",
        collision_free_goals,
        "/",
        len(goals)
    )

    p.disconnect(client_id)

    print("\n" + "=" * 60)
    print("ENDPOINT VALIDITY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()