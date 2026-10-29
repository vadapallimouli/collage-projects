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
    print("TRAJECTORY EVALUATOR TEST")
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

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    print("Creating Environment...")

    environment = Environment.from_mpinets_problem(
        problem
    )

    print("Creating obstacles...")

    environment.create_pybullet_obstacles(
        client_id
    )

    # ---------------------------------------------------------
    # GOALS
    # ---------------------------------------------------------

    print("Generating goal configurations...")

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

    if len(goals) == 0:
        raise RuntimeError(
            "No valid goals generated."
        )

    # ---------------------------------------------------------
    # TRAJECTORIES
    # ---------------------------------------------------------

    print("Generating candidate trajectories...")

    trajectory_generator = TrajectoryGenerator(
        panda_robot
    )

    q_start = np.asarray(
        problem.q0,
        dtype=float
    )

    trajectories = (
        trajectory_generator.generate_candidate_trajectories(
            start_configuration=q_start,
            goal_configurations=goals,
            num_candidates=10,
            num_points=100,
        )
    )

    print(
        f"Generated {len(trajectories)} trajectories."
    )

    # ---------------------------------------------------------
    # ANALYSIS MODULES
    # ---------------------------------------------------------

    print("Creating CollisionChecker...")

    collision_checker = CollisionChecker(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    print("Creating ClearanceAnalyzer...")

    clearance_analyzer = ClearanceAnalyzer(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    print("Creating TrajectoryEvaluator...")

    evaluator = TrajectoryEvaluator(
        panda_robot=panda_robot,
        environment=environment,
        collision_checker=collision_checker,
        clearance_analyzer=clearance_analyzer,
    )

    # ---------------------------------------------------------
    # TEST 1
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 1: SINGLE TRAJECTORY EVALUATION")
    print("-" * 60)

    result = evaluator.evaluate_trajectory(
        trajectories[0]
    )

    print("\nEvaluation result:")

    for key, value in result.items():
        print(f"{key}: {value}")

    required_keys = [
        "collision",
        "self_collision",
        "obstacle_collision",
        "joint_limit_violation",
        "minimum_clearance",
        "joint_space_length",
        "max_joint_step",
        "num_points",
        "valid",
    ]

    for key in required_keys:
        assert key in result

    assert result["num_points"] == 100

    assert np.isfinite(
        result["minimum_clearance"]
    )

    assert np.isfinite(
        result["joint_space_length"]
    )

    assert np.isfinite(
        result["max_joint_step"]
    )

    print("\nTEST 1 PASSED")

    # ---------------------------------------------------------
    # TEST 2
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 2: MULTIPLE TRAJECTORY EVALUATION")
    print("-" * 60)

    results = evaluator.evaluate_multiple(
        trajectories
    )

    print(
        f"\nNumber of evaluation results: "
        f"{len(results)}"
    )

    assert len(results) == len(
        trajectories
    )

    for result in results:

        assert "trajectory_index" in result

        assert result["num_points"] == 100

        print(
            f"Trajectory "
            f"{result['trajectory_index'] + 1}: "
            f"collision={result['collision']}, "
            f"clearance="
            f"{result['minimum_clearance']:.6f}, "
            f"valid={result['valid']}"
        )

    print("\nTEST 2 PASSED")

    # ---------------------------------------------------------
    # TEST 3
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 3: COLLISION CONSISTENCY")
    print("-" * 60)

    for result in results:

        if result["collision"]:
            assert result["collision_count"] > 0

        else:
            assert result["collision_count"] == 0

    print(
        "\nCollision results are internally consistent."
    )

    print("TEST 3 PASSED")

    # ---------------------------------------------------------
    # TEST 4
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 4: CLEARANCE ANALYSIS")
    print("-" * 60)

    clearances = [
        result["minimum_clearance"]
        for result in results
    ]

    print("\nMinimum clearance across candidates:")

    for i, clearance in enumerate(
        clearances
    ):

        print(
            f"Trajectory {i + 1}: "
            f"{clearance:.6f} m"
        )

        assert np.isfinite(
            clearance
        )

    print(
        "\nClearance calculated for every trajectory."
    )

    print("TEST 4 PASSED")

    # ---------------------------------------------------------
    # TEST 5
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 5: FEATURE COLLECTION")
    print("-" * 60)

    for result in results:

        assert isinstance(
            result["joint_space_length"],
            (float, np.floating)
        )

        assert isinstance(
            result["max_joint_step"],
            (float, np.floating)
        )

        assert result[
            "joint_space_length"
        ] >= 0.0

        assert result[
            "max_joint_step"
        ] >= 0.0

    print(
        "\nBasic trajectory features successfully collected."
    )

    print("TEST 5 PASSED")

    # ---------------------------------------------------------
    # FINISH
    # ---------------------------------------------------------

    p.disconnect(
        client_id
    )

    print("\n" + "=" * 60)
    print("ALL TRAJECTORY EVALUATOR TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()