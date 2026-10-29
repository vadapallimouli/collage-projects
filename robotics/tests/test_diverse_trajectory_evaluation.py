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

from analysis.collision_checker import (
    CollisionChecker
)

from analysis.clearance_analyzer import (
    ClearanceAnalyzer
)

from analysis.trajectory_evaluator import (
    TrajectoryEvaluator
)


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("DIVERSE TRAJECTORY EVALUATION TEST")
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

    panda_robot = PandaRobot(robot_id)

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

    q_start = np.asarray(
        problem.q0,
        dtype=float
    )

    q_goal = np.asarray(
        goals[0]["configuration"],
        dtype=float
    )

    # ---------------------------------------------------------
    # GENERATE DIVERSE TRAJECTORIES
    # ---------------------------------------------------------

    print(
        "\nGenerating diverse trajectories..."
    )

    trajectory_generator = (
        TrajectoryGenerator(
            panda_robot
        )
    )

    trajectories = (
        trajectory_generator
        .generate_diverse_trajectories(
            start_configuration=q_start,
            goal_configuration=q_goal,
            num_candidates=10,
            num_points=100,
            seed=42,
            perturbation_scale=0.8,
        )
    )

    print(
        f"Generated {len(trajectories)} trajectories."
    )

    # ---------------------------------------------------------
    # ANALYSIS MODULES
    # ---------------------------------------------------------

    print(
        "Creating CollisionChecker..."
    )

    collision_checker = CollisionChecker(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    print(
        "Creating ClearanceAnalyzer..."
    )

    clearance_analyzer = ClearanceAnalyzer(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    print(
        "Creating TrajectoryEvaluator..."
    )

    evaluator = TrajectoryEvaluator(
        panda_robot=panda_robot,
        environment=environment,
        collision_checker=collision_checker,
        clearance_analyzer=clearance_analyzer,
    )

    # ---------------------------------------------------------
    # EVALUATE ALL TRAJECTORIES
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("EVALUATING DIVERSE TRAJECTORIES")
    print("-" * 60)

    results = evaluator.evaluate_multiple(
        trajectories
    )

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    safe_count = 0
    collision_count = 0

    print()

    for result in results:

        index = (
            result["trajectory_index"]
            + 1
        )

        collision = result[
            "collision"
        ]

        clearance = result[
            "minimum_clearance"
        ]

        joint_length = result[
            "joint_space_length"
        ]

        valid = result[
            "valid"
        ]

        if valid:
            safe_count += 1

        if collision:
            collision_count += 1

        status = (
            "SAFE"
            if valid
            else "FAILED"
        )

        print(
            f"Trajectory {index:2d} | "
            f"{status:6s} | "
            f"collision={str(collision):5s} | "
            f"clearance={clearance: .6f} m | "
            f"joint_length={joint_length:.4f}"
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("RESULT SUMMARY")
    print("-" * 60)

    print(
        f"\nTotal trajectories: {len(results)}"
    )

    print(
        f"Collision-free trajectories: {safe_count}"
    )

    print(
        f"Colliding trajectories: {collision_count}"
    )

    print(
        f"Collision-free percentage: "
        f"{100.0 * safe_count / len(results):.1f}%"
    )

    # ---------------------------------------------------------
    # FEATURE RANGE
    # ---------------------------------------------------------

    clearances = np.array([
        result["minimum_clearance"]
        for result in results
    ])

    joint_lengths = np.array([
        result["joint_space_length"]
        for result in results
    ])

    max_steps = np.array([
        result["max_joint_step"]
        for result in results
    ])

    print("\nFeature ranges:")

    print(
        f"Minimum clearance: "
        f"{clearances.min():.6f} to "
        f"{clearances.max():.6f} m"
    )

    print(
        f"Joint-space length: "
        f"{joint_lengths.min():.6f} to "
        f"{joint_lengths.max():.6f}"
    )

    print(
        f"Maximum joint step: "
        f"{max_steps.min():.6f} to "
        f"{max_steps.max():.6f} rad"
    )

    # ---------------------------------------------------------
    # IMPORTANT DATASET CHECK
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("ML DATASET CHECK")
    print("-" * 60)

    if safe_count > 0 and collision_count > 0:

        print(
            "\nBoth successful and failed trajectories "
            "are present."
        )

        print(
            "This provides the two outcome classes "
            "needed for supervised failure prediction."
        )

    elif safe_count == 0:

        print(
            "\nAll trajectories failed."
        )

        print(
            "The current waypoint strategy does not "
            "yet produce enough successful trajectories."
        )

    else:

        print(
            "\nAll trajectories are collision-free."
        )

        print(
            "More challenging candidate trajectories "
            "will be needed to obtain failure examples."
        )

    # ---------------------------------------------------------
    # FINISH
    # ---------------------------------------------------------

    p.disconnect(
        client_id
    )

    print("\n" + "=" * 60)
    print(
        "DIVERSE TRAJECTORY EVALUATION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()