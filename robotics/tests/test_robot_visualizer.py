"""
tests/test_robot_visualizer.py

Robot visualization integration test.

Architecture:

    MπNets
       ↓
    DIRECT PyBullet
       ↓
    Planning
       ↓
    Physical validation
       ↓
    Close DIRECT server
       ↓
    GUI PyBullet
       ↓
    Display environment
       ↓
    Display trajectory
       ↓
    Smooth robot animation
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ================================================================
# IMPORTS
# ================================================================

from robot.panda_robot import PandaRobot

from environment.environment import Environment

from analysis.collision_checker import CollisionChecker

from planning.trajectory_generator import (
    TrajectoryGenerator,
)

from planning.goal_configuration import (
    GoalConfigurationGenerator,
)

from planning.obstacle_aware_planner import (
    ObstacleAwarePlanner,
)

from visualization.robot_visualizer import (
    RobotVisualizer,
)


# ================================================================
# DATASET
# ================================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)

ENVIRONMENT_NAME = "tabletop"
PROBLEM_SET_NAME = "task_oriented"
PROBLEM_INDEX = 0


# ================================================================
# PLANNER SETTINGS
# ================================================================

PLANNER_SAMPLES = 500
PLANNER_K_NEIGHBORS = 12
PLANNER_EDGE_POINTS = 20


# ================================================================
# LOAD PROBLEM
# ================================================================

def load_problem():

    print(
        "Loading MπNets dataset..."
    )

    with open(
        DATASET_PATH,
        "rb",
    ) as file:

        dataset = pickle.load(
            file
        )

    problem = (
        dataset
        [ENVIRONMENT_NAME]
        [PROBLEM_SET_NAME]
        [PROBLEM_INDEX]
    )

    print(
        f"Environment : "
        f"{ENVIRONMENT_NAME}"
    )

    print(
        f"Problem set : "
        f"{PROBLEM_SET_NAME}"
    )

    print(
        f"Problem      : "
        f"{PROBLEM_INDEX}"
    )

    return problem


# ================================================================
# CREATE PYBULLET
# ================================================================

def create_simulation(
    mode,
):

    physics_client = p.connect(
        mode
    )

    if physics_client < 0:

        raise RuntimeError(
            "Could not connect to PyBullet."
        )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81,
    )

    p.loadURDF(
        "plane.urdf"
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
    )

    panda_robot = PandaRobot(
        robot_id
    )

    return (
        physics_client,
        panda_robot,
    )


# ================================================================
# FIND GOAL
# ================================================================

def find_collision_free_goal(
    panda_robot,
    environment,
    collision_checker,
    physics_client,
):

    print(
        "Generating goal configurations..."
    )

    goal_generator = (
        GoalConfigurationGenerator(
            panda_robot=panda_robot,
            environment=environment,
            target_position=environment.target_position,
            target_half_extents=environment.target_half_extents,
            client_id=physics_client,
        )
    )

    solutions = (
        goal_generator.solve_multiple(
            num_solutions=20,
            num_rest_poses=20,
            seed=42,
        )
    )

    print(
        f"IK solutions generated: "
        f"{len(solutions)}"
    )

    for solution in solutions:

        configuration = np.asarray(
            solution["configuration"],
            dtype=float,
        )

        result = (
            collision_checker.check_configuration(
                configuration
            )
        )

        if not result["collision"]:

            print(
                "Collision-free goal found."
            )

            return configuration

    raise RuntimeError(
        "Could not find collision-free goal."
    )


# ================================================================
# PLAN
# ================================================================

def generate_trajectory():

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 1: PLANNING"
    )

    print(
        "=" * 70
    )

    physics_client = None

    try:

        physics_client, panda_robot = (
            create_simulation(
                p.DIRECT
            )
        )

        print()
        print(
            "PyBullet mode: DIRECT"
        )

        problem = load_problem()

        environment = (
            Environment.from_mpinets_problem(
                problem
            )
        )

        obstacle_ids = (
            environment.create_pybullet_obstacles(
                client_id=physics_client
            )
        )

        print()
        print(
            f"Obstacles created: "
            f"{len(obstacle_ids)}"
        )

        collision_checker = (
            CollisionChecker(
                panda_robot=panda_robot,
                environment=environment,
                client_id=physics_client,
            )
        )

        trajectory_generator = (
            TrajectoryGenerator(
                panda_robot
            )
        )

        start = np.asarray(
            problem.q0,
            dtype=float,
        )

        start_result = (
            collision_checker.check_configuration(
                start
            )
        )

        print()
        print(
            "Start collision: "
            f"{start_result['collision']}"
        )

        if start_result["collision"]:

            raise RuntimeError(
                "Start configuration is in collision."
            )

        goal = find_collision_free_goal(
            panda_robot=panda_robot,
            environment=environment,
            collision_checker=collision_checker,
            physics_client=physics_client,
        )

        planner = ObstacleAwarePlanner(
            panda_robot=panda_robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=PLANNER_SAMPLES,
            k_neighbors=PLANNER_K_NEIGHBORS,
            edge_points=PLANNER_EDGE_POINTS,
            seed=42,
        )

        print()
        print(
            "Generating obstacle-aware trajectory..."
        )

        result = planner.plan(
            start,
            goal,
        )

        if not result["success"]:

            raise RuntimeError(
                "Planner could not find a trajectory."
            )

        trajectory = np.asarray(
            result["trajectory"],
            dtype=float,
        )

        print()
        print(
            "Trajectory generated."
        )

        print(
            f"Trajectory shape: "
            f"{trajectory.shape}"
        )

        print(
            f"Path length: "
            f"{result['path_length']:.6f}"
        )

        validation = (
            collision_checker.check_trajectory(
                trajectory
            )
        )

        print()
        print(
            "Collision-free: "
            f"{validation['collision_free']}"
        )

        print(
            "Collision count: "
            f"{validation['collision_count']}"
        )

        print(
            "Obstacle collisions: "
            f"{validation['obstacle_collision_count']}"
        )

        print(
            "Self collisions: "
            f"{validation['self_collision_count']}"
        )

        print(
            "Joint-limit violations: "
            f"{validation['joint_limit_violation_count']}"
        )

        if not validation["collision_free"]:

            raise RuntimeError(
                "Generated trajectory is not valid."
            )

        return (
            trajectory,
            problem,
        )

    finally:

        if (
            physics_client is not None
            and p.isConnected(
                physics_client
            )
        ):

            p.disconnect(
                physics_client
            )

            print()
            print(
                "DIRECT planning server closed."
            )


# ================================================================
# GUI
# ================================================================

def run_gui(
    trajectory,
    problem,
):

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 2: 3D VISUALIZATION"
    )

    print(
        "=" * 70
    )

    physics_client = None

    try:

        # --------------------------------------------------------
        # Connect GUI
        # --------------------------------------------------------

        print()
        print(
            "Starting PyBullet GUI..."
        )

        physics_client = p.connect(
            p.GUI
        )

        if physics_client < 0:

            raise RuntimeError(
                "Failed to start PyBullet GUI."
            )

        print(
            f"GUI physics client: "
            f"{physics_client}"
        )

        time.sleep(
            1.0
        )

        if not p.isConnected(
            physics_client
        ):

            raise RuntimeError(
                "PyBullet GUI disconnected immediately "
                "after connection."
            )

        print(
            "GUI connection: ALIVE"
        )

        # --------------------------------------------------------
        # Search path
        # --------------------------------------------------------

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(
            0,
            0,
            -9.81,
        )

        # --------------------------------------------------------
        # Plane
        # --------------------------------------------------------

        p.loadURDF(
            "plane.urdf"
        )

        print(
            "Plane created."
        )

        # --------------------------------------------------------
        # Panda
        # --------------------------------------------------------

        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
        )

        panda_robot = PandaRobot(
            robot_id
        )

        print(
            "Panda robot created."
        )

        # --------------------------------------------------------
        # Environment
        # --------------------------------------------------------

        environment = (
            Environment.from_mpinets_problem(
                problem
            )
        )

        obstacle_ids = (
            environment.create_pybullet_obstacles(
                client_id=physics_client
            )
        )

        print(
            f"Obstacles created: "
            f"{len(obstacle_ids)}"
        )

        time.sleep(
            1.0
        )

        if not p.isConnected(
            physics_client
        ):

            raise RuntimeError(
                "GUI disconnected while creating "
                "the environment."
            )

        print(
            "GUI connection after environment: ALIVE"
        )

        # --------------------------------------------------------
        # Visualizer
        # --------------------------------------------------------

        visualizer = RobotVisualizer(
            panda_robot=panda_robot,
            environment=environment,
            client_id=physics_client,
            trajectory_dt=0.08,
            interpolation_steps=3,
        )

        # --------------------------------------------------------
        # Camera
        # --------------------------------------------------------

        visualizer.set_camera(
            distance=1.8,
            yaw=45.0,
            pitch=-35.0,
            target=environment.target_position,
        )

        print(
            "Camera configured."
        )

        # --------------------------------------------------------
        # Put robot at START
        # --------------------------------------------------------

        print()
        print(
            "Moving robot to START..."
        )

        visualizer.set_robot_configuration(
            trajectory[0]
        )

        p.stepSimulation()

        time.sleep(
            2.0
        )

        if not p.isConnected(
            physics_client
        ):

            raise RuntimeError(
                "GUI disconnected before animation."
            )

        print(
            "Robot is at START."
        )

        # --------------------------------------------------------
        # Draw trajectory
        # --------------------------------------------------------

        print(
            "Drawing trajectory..."
        )

        visualizer.mark_start_and_goal(
            trajectory
        )

        visualizer.draw_trajectory(
            trajectory
        )

        p.stepSimulation()

        time.sleep(
            2.0
        )

        if not p.isConnected(
            physics_client
        ):

            raise RuntimeError(
                "GUI disconnected after drawing trajectory."
            )

        print(
            "Trajectory displayed."
        )

        # --------------------------------------------------------
        # Animate
        # --------------------------------------------------------

        print()
        print(
            "Starting smooth animation..."
        )

        visualizer.animate_trajectory(
            trajectory,
            loop=False,
            hold_at_end=2.0,
            interpolation_steps=3,
        )

        print()
        print(
            "[PASS] Smooth animation completed."
        )

        # --------------------------------------------------------
        # Keep GUI alive
        # --------------------------------------------------------

        print()
        print(
            "Robot has reached GOAL."
        )

        print(
            "The window will remain open."
        )

        print(
            "Close the PyBullet window manually."
        )

        visualizer.wait_until_closed()

    finally:

        if (
            physics_client is not None
            and p.isConnected(
                physics_client
            )
        ):

            p.disconnect(
                physics_client
            )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TESTING ROBOT VISUALIZER"
    )

    print(
        "=" * 70
    )

    trajectory, problem = (
        generate_trajectory()
    )

    run_gui(
        trajectory,
        problem,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ROBOT VISUALIZER TEST PASSED"
    )

    print(
        "=" * 70
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()