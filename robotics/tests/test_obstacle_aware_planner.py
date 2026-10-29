import pickle

import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment
from analysis.collision_checker import CollisionChecker
from planning.trajectory_generator import TrajectoryGenerator
from planning.goal_configuration import GoalConfigurationGenerator
from planning.obstacle_aware_planner import ObstacleAwarePlanner


def main():

    # --------------------------------------------------
    # Load MπNets problem
    # --------------------------------------------------

    with open(
        "data/mpinets/both_solvable_problems.pkl",
        "rb",
    ) as f:
        dataset = pickle.load(f)

    print("\nDataset loaded.")

    print("Top-level keys:", dataset.keys())

    # Use tabletop task-oriented problems.
    problem = dataset["tabletop"]["task_oriented"][0]

    print(
        "Selected tabletop/task_oriented problem 0."
    )

    # --------------------------------------------------
    # Start PyBullet
    # --------------------------------------------------

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError(
            "Could not connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    # --------------------------------------------------
    # Load Panda
    # --------------------------------------------------

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
    )

    robot = PandaRobot(robot_id)

    print(
        "\nPanda loaded."
    )

    print(
        "DOF:",
        robot.get_dof(),
    )

    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------

    environment = Environment.from_mpinets_problem(
        problem
    )

    print(
        "\nEnvironment created."
    )

    print(
        "Number of obstacles:",
        environment.num_obstacles,
    )

    # --------------------------------------------------
    # Create PyBullet obstacles
    # --------------------------------------------------

    environment.create_pybullet_obstacles(
        client_id
    )

    print(
        "Obstacles loaded into PyBullet."
    )

    # --------------------------------------------------
    # Collision checker
    # --------------------------------------------------

    collision_checker = CollisionChecker(
        robot,
        environment,
        client_id,
    )

    # --------------------------------------------------
    # Trajectory generator
    # --------------------------------------------------

    trajectory_generator = TrajectoryGenerator(
        robot
    )

    # --------------------------------------------------
    # Start configuration
    # --------------------------------------------------

    start = np.asarray(
        problem.q0,
        dtype=float,
    )

    print("\nSTART CONFIGURATION:")
    print(start)

    start_result = collision_checker.check_configuration(
        start
    )

    print(
        "Start collision:",
        start_result["collision"],
    )

    if start_result["collision"]:
        raise RuntimeError(
            "Start configuration is in collision."
        )

    # --------------------------------------------------
    # Generate goal
    # --------------------------------------------------

    goal_generator = GoalConfigurationGenerator(
        panda_robot=robot,
        environment=environment,
        client_id=client_id,
    )

    print(
        "\nGenerating goal configuration..."
    )

    goals = goal_generator.solve_multiple(
        num_solutions=1,
        num_rest_poses=50,
        seed=42,
    )

    if not goals:
        raise RuntimeError(
            "Could not generate a valid goal."
        )

    goal = np.asarray(
        goals[0]["configuration"],
        dtype=float,
    )

    print("\nGOAL CONFIGURATION:")
    print(goal)

    goal_result = collision_checker.check_configuration(
        goal
    )

    print(
        "Goal collision:",
        goal_result["collision"],
    )

    if goal_result["collision"]:
        raise RuntimeError(
            "Generated goal is in collision."
        )

    # --------------------------------------------------
    # Create obstacle-aware planner
    # --------------------------------------------------

    planner = ObstacleAwarePlanner(
        panda_robot=robot,
        collision_checker=collision_checker,
        trajectory_generator=trajectory_generator,
        num_samples=100,
        k_neighbors=8,
        edge_points=15,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("OBSTACLE-AWARE PLANNER")
    print("=" * 60)

    print(
        "\nPlanning from start to goal..."
    )

    # --------------------------------------------------
    # Plan
    # --------------------------------------------------

    result = planner.plan(
        start,
        goal,
    )

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    if result is None:

        print(
            "\nNo collision-free path found."
        )

    else:

        trajectory = result["trajectory"]

        print(
            "\nPATH FOUND!"
        )

        print(
            "Trajectory shape:",
            trajectory.shape,
        )

        print(
            "Number of waypoints:",
            result["num_waypoints"],
        )

        print(
            "Joint-space path length:",
            result["path_length"],
        )

        # --------------------------------------------------
        # Validate complete trajectory
        # --------------------------------------------------

        validation = collision_checker.check_trajectory(
            trajectory
        )

        print("\nFINAL VALIDATION")
        print("-" * 40)

        print(
            "Collision free:",
            validation["collision_free"],
        )

        print(
            "Collision count:",
            validation["collision_count"],
        )

        print(
            "Obstacle collisions:",
            validation[
                "obstacle_collision_count"
            ],
        )

        print(
            "Self collisions:",
            validation[
                "self_collision_count"
            ],
        )

        print(
            "Joint-limit violations:",
            validation[
                "joint_limit_violation_count"
            ],
        )

    # --------------------------------------------------
    # Disconnect
    # --------------------------------------------------

    p.disconnect(client_id)


if __name__ == "__main__":
    main()