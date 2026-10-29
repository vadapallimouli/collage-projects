"""
Test multiple obstacle-aware candidate trajectories.

This test:
    1. Loads a real MπNets planning problem.
    2. Creates the Panda robot.
    3. Creates the obstacle environment.
    4. Generates a valid goal configuration.
    5. Generates multiple obstacle-aware candidate paths.
    6. Validates every candidate.
    7. Checks path diversity.
"""

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


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "data/mpinets/both_solvable_problems.pkl"

NUM_CANDIDATES = 10
NUM_TRAJECTORY_POINTS = 200

NUM_SAMPLES = 700
K_NEIGHBORS = 15
EDGE_POINTS = 25

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("MULTIPLE CANDIDATE PATH TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print()
    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    print("Dataset loaded.")
    print("Top-level keys:", dataset.keys())

    # --------------------------------------------------------
    # SELECT PROBLEM
    # --------------------------------------------------------

    problem = dataset["tabletop"]["task_oriented"][0]

    print()
    print("Selected tabletop/task_oriented problem 0.")

    # --------------------------------------------------------
    # CONNECT TO PYBULLET
    # --------------------------------------------------------

    physics_client = p.connect(p.DIRECT)

    if physics_client < 0:
        raise RuntimeError(
            "Failed to connect to PyBullet."
        )

    try:

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.loadURDF("plane.urdf")

        # ----------------------------------------------------
        # LOAD PANDA
        # ----------------------------------------------------

        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True
        )

        robot = PandaRobot(robot_id)

        print()
        print("Panda loaded.")
        print("DOF:", robot.DOF)

        # ----------------------------------------------------
        # CREATE ENVIRONMENT
        # ----------------------------------------------------

        environment = Environment.from_mpinets_problem(
            problem
        )

        print()
        print("Environment created.")
        print(
            "Number of obstacles:",
            environment.num_obstacles
        )

        environment.create_pybullet_obstacles(
            physics_client
        )

        print("Obstacles loaded into PyBullet.")

        # ----------------------------------------------------
        # COLLISION CHECKER
        # ----------------------------------------------------

        collision_checker = CollisionChecker(
            panda_robot=robot,
            environment=environment,
            client_id=physics_client
        )

        # ----------------------------------------------------
        # TRAJECTORY GENERATOR
        # ----------------------------------------------------

        trajectory_generator = TrajectoryGenerator(
            robot
        )

        # ----------------------------------------------------
        # START CONFIGURATION
        # ----------------------------------------------------

        start = np.asarray(
            problem.q0,
            dtype=float
        )

        print()
        print("START CONFIGURATION:")
        print(start)

        start_result = collision_checker.check_configuration(
            start
        )

        print(
            "Start collision:",
            start_result["collision"]
        )

        if start_result["collision"]:
            raise RuntimeError(
                "Start configuration is in collision."
            )

        # ----------------------------------------------------
        # GENERATE GOAL CONFIGURATION
        # ----------------------------------------------------

        print()
        print("Generating goal configuration...")

        goal_generator = GoalConfigurationGenerator(
            panda_robot=robot,
            environment=environment,
            client_id=physics_client
        )

        goal_solutions = goal_generator.solve_multiple(
            num_solutions=1,
            num_rest_poses=50,
            seed=SEED
        )

        if len(goal_solutions) == 0:
            raise RuntimeError(
                "Could not generate a valid goal configuration."
            )

        goal = np.asarray(
            goal_solutions[0]["configuration"],
            dtype=float
        )

        print()
        print("GOAL CONFIGURATION:")
        print(goal)

        goal_result = collision_checker.check_configuration(
            goal
        )

        print(
            "Goal collision:",
            goal_result["collision"]
        )

        if goal_result["collision"]:
            raise RuntimeError(
                "Goal configuration is in collision."
            )

        # ----------------------------------------------------
        # CREATE OBSTACLE-AWARE PLANNER
        # ----------------------------------------------------

        planner = ObstacleAwarePlanner(
            panda_robot=robot,
            collision_checker=collision_checker,
            trajectory_generator=trajectory_generator,
            num_samples=NUM_SAMPLES,
            k_neighbors=K_NEIGHBORS,
            edge_points=EDGE_POINTS,
            seed=SEED
        )

        # ----------------------------------------------------
        # GENERATE MULTIPLE PATHS
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("GENERATING MULTIPLE CANDIDATE PATHS")
        print("=" * 60)

        print()
        print(
            "Requested candidates:",
            NUM_CANDIDATES
        )

        print(
            "Roadmap samples:",
            NUM_SAMPLES
        )

        print(
            "Nearest neighbors:",
            K_NEIGHBORS
        )

        print(
            "Edge validation points:",
            EDGE_POINTS
        )

        print()
        print("Building roadmap and searching for paths...")

        results = planner.plan_multiple(
            start_configuration=start,
            goal_configuration=goal,
            num_candidates=NUM_CANDIDATES,
            num_points=NUM_TRAJECTORY_POINTS
        )

        # ----------------------------------------------------
        # NUMBER OF RESULTS
        # ----------------------------------------------------

        print()
        print(
            "Generated candidates:",
            len(results)
        )

        if len(results) == 0:

            print()
            print("=" * 60)
            print("NO CANDIDATE PATHS FOUND")
            print("=" * 60)

            return

        # ----------------------------------------------------
        # ANALYZE CANDIDATES
        # ----------------------------------------------------

        successful_count = 0
        failed_count = 0

        successful_lengths = []
        failed_lengths = []

        print()
        print("=" * 60)
        print("CANDIDATE RESULTS")
        print("=" * 60)

        for i, result in enumerate(results):

            trajectory = result["trajectory"]

            validation = result["validation"]

            success = bool(
                result["success"]
            )

            path_length = float(
                result["path_length"]
            )

            graph_path = result["path"]

            num_waypoints = max(
                0,
                len(graph_path) - 2
            )

            if success:
                successful_count += 1
                successful_lengths.append(
                    path_length
                )
            else:
                failed_count += 1
                failed_lengths.append(
                    path_length
                )

            print()
            print(
                f"Candidate {i + 1}"
            )
            print("-" * 40)

            print(
                "Success:",
                success
            )

            print(
                "Trajectory shape:",
                trajectory.shape
            )

            print(
                "Graph path nodes:",
                len(graph_path)
            )

            print(
                "Intermediate waypoints:",
                num_waypoints
            )

            print(
                "Joint-space path length:",
                path_length
            )

            print(
                "Collision free:",
                validation["collision_free"]
            )

            print(
                "Collision count:",
                validation["collision_count"]
            )

            print(
                "Obstacle collisions:",
                validation[
                    "obstacle_collision_count"
                ]
            )

            print(
                "Self collisions:",
                validation[
                    "self_collision_count"
                ]
            )

            print(
                "Joint-limit violations:",
                validation[
                    "joint_limit_violation_count"
                ]
            )

            print(
                "Minimum clearance:",
                validation.get(
                    "minimum_clearance",
                    "N/A"
                )
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print()
        print(
            "Requested candidates:",
            NUM_CANDIDATES
        )

        print(
            "Generated candidates:",
            len(results)
        )

        print(
            "Successful candidates:",
            successful_count
        )

        print(
            "Failed candidates:",
            failed_count
        )

        # ----------------------------------------------------
        # SUCCESSFUL PATH LENGTHS
        # ----------------------------------------------------

        if len(successful_lengths) > 0:

            print()
            print("Successful path lengths:")

            for i, length in enumerate(
                successful_lengths
            ):
                print(
                    f"  {i + 1}: {length:.6f}"
                )

            print()
            print(
                "Shortest successful path:",
                f"{min(successful_lengths):.6f}"
            )

            print(
                "Longest successful path:",
                f"{max(successful_lengths):.6f}"
            )

        # ----------------------------------------------------
        # FAILED PATH LENGTHS
        # ----------------------------------------------------

        if len(failed_lengths) > 0:

            print()
            print("Failed path lengths:")

            for i, length in enumerate(
                failed_lengths
            ):
                print(
                    f"  {i + 1}: {length:.6f}"
                )

        # ----------------------------------------------------
        # PATH DIVERSITY
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("PATH DIVERSITY CHECK")
        print("=" * 60)

        graph_paths = [
            result["path"]
            for result in results
        ]

        if len(graph_paths) >= 2:

            similarities = []

            for i in range(
                len(graph_paths)
            ):

                for j in range(
                    i + 1,
                    len(graph_paths)
                ):

                    similarity = planner.path_similarity(
                        graph_paths[i],
                        graph_paths[j]
                    )

                    similarities.append(
                        similarity
                    )

                    print(
                        f"Path {i + 1} vs "
                        f"Path {j + 1}: "
                        f"{similarity:.3f}"
                    )

            if len(similarities) > 0:

                print()
                print(
                    "Average path similarity:",
                    f"{np.mean(similarities):.3f}"
                )

                print(
                    "Minimum path similarity:",
                    f"{np.min(similarities):.3f}"
                )

                print(
                    "Maximum path similarity:",
                    f"{np.max(similarities):.3f}"
                )

        else:

            print(
                "Not enough paths for diversity comparison."
            )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        print()
        print("=" * 60)

        if successful_count > 0:

            print(
                "MULTIPLE PATH TEST COMPLETED."
            )

            print(
                "At least one collision-free "
                "candidate was found."
            )

        else:

            print(
                "NO SUCCESSFUL PATHS WERE FOUND."
            )

        print("=" * 60)

    finally:

        # ----------------------------------------------------
        # DISCONNECT PYBULLET
        # ----------------------------------------------------

        if p.isConnected(physics_client):

            p.disconnect(
                physics_client
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()