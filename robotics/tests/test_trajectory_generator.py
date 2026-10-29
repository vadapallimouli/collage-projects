import pickle
import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot
from environment.environment import Environment
from planning.goal_configuration import GoalConfigurationGenerator
from planning.trajectory_generator import TrajectoryGenerator


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("TRAJECTORY GENERATOR TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------

    print("\nConnecting to PyBullet...")

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

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

    environment = Environment.from_mpinets_problem(problem)

    print("Creating obstacles...")

    environment.create_pybullet_obstacles(client_id)

    print("Creating GoalConfigurationGenerator...")

    goal_generator = GoalConfigurationGenerator(
        panda_robot=panda_robot,
        environment=environment,
        client_id=client_id,
    )

    print("Generating goal configurations...")

    goals = goal_generator.solve_multiple(
        num_solutions=10,
        num_rest_poses=50,
        seed=42,
    )

    if len(goals) == 0:
        raise RuntimeError("No valid goal configurations were generated.")

    print(f"Generated {len(goals)} goal configurations.")

    print("\nCreating TrajectoryGenerator...")

    trajectory_generator = TrajectoryGenerator(
        panda_robot=panda_robot
    )

    q_start = np.asarray(problem.q0, dtype=float)

    # ---------------------------------------------------------
    # TEST 1
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 1: SINGLE TRAJECTORY")
    print("-" * 60)

    goal = goals[0]

    trajectory = trajectory_generator.generate_joint_trajectory(
        start_configuration=q_start,
        goal_configuration=goal["configuration"],
        num_points=100,
    )

    print("\nTrajectory shape:")
    print(trajectory.shape)

    print("\nFirst point:")
    print(trajectory[0])

    print("\nLast point:")
    print(trajectory[-1])

    assert trajectory.shape == (100, 7)

    assert np.allclose(
        trajectory[0],
        q_start,
        atol=1e-8
    )

    assert np.allclose(
        trajectory[-1],
        goal["configuration"],
        atol=1e-8
    )

    # Every point must respect joint limits.
    for q in trajectory:
        assert panda_robot.is_within_joint_limits(q)

    print("\nAll trajectory points satisfy joint limits.")

    print("TEST 1 PASSED")

    # ---------------------------------------------------------
    # TEST 2
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 2: MULTIPLE TRAJECTORIES")
    print("-" * 60)

    trajectories = trajectory_generator.generate_multiple_trajectories(
        start_configuration=q_start,
        goal_configurations=goals,
        num_points=100,
    )

    print(f"\nNumber of trajectories: {len(trajectories)}")

    assert len(trajectories) == len(goals)

    for i, traj in enumerate(trajectories):

        assert traj.shape == (100, 7)

        assert np.allclose(
            traj[0],
            q_start,
            atol=1e-8
        )

        assert np.allclose(
            traj[-1],
            goals[i]["configuration"],
            atol=1e-8
        )

        for q in traj:
            assert panda_robot.is_within_joint_limits(q)

        print(
            f"Trajectory {i + 1}: "
            f"shape={traj.shape}, "
            f"valid=True"
        )

    print("\nTEST 2 PASSED")

    # ---------------------------------------------------------
    # TEST 3
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 3: TRAJECTORY DIVERSITY")
    print("-" * 60)

    final_configurations = [
        trajectory[-1]
        for trajectory in trajectories
    ]

    distances = []

    for i in range(len(final_configurations)):
        for j in range(i + 1, len(final_configurations)):

            distance = np.linalg.norm(
                final_configurations[i]
                - final_configurations[j]
            )

            distances.append(distance)

    if distances:

        min_distance = min(distances)
        max_distance = max(distances)

        print(f"\nMinimum final joint-space distance: {min_distance:.6f}")
        print(f"Maximum final joint-space distance: {max_distance:.6f}")

        assert max_distance > 1e-3

        print("\nMultiple distinct trajectories confirmed.")

    print("TEST 3 PASSED")

    # ---------------------------------------------------------
    # TEST 4
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 4: SMOOTHNESS")
    print("-" * 60)

    trajectory = trajectories[0]

    # First differences approximate velocity.
    velocity = np.diff(trajectory, axis=0)

    # Ensure there are no sudden enormous jumps.
    max_step = np.max(np.abs(velocity))

    print(f"\nMaximum joint step: {max_step:.6f} rad")

    assert np.isfinite(max_step)
    assert max_step < 0.5

    # Cubic interpolation should have very small
    # velocity at both endpoints.
    start_velocity = np.linalg.norm(velocity[0])
    end_velocity = np.linalg.norm(velocity[-1])

    print(f"Start velocity approximation: {start_velocity:.6f}")
    print(f"End velocity approximation:   {end_velocity:.6f}")

    assert start_velocity < 0.01
    assert end_velocity < 0.01

    print("\nSmooth trajectory confirmed.")

    print("TEST 4 PASSED")

    # ---------------------------------------------------------
    # TEST 5
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TEST 5: CANDIDATE TRAJECTORY INTERFACE")
    print("-" * 60)

    candidates = trajectory_generator.generate_candidate_trajectories(
        start_configuration=q_start,
        goal_configurations=goals,
        num_candidates=5,
        num_points=100,
    )

    print(f"\nRequested candidates: 5")
    print(f"Generated candidates: {len(candidates)}")

    assert len(candidates) == 5

    for candidate in candidates:
        assert candidate.shape == (100, 7)
        assert np.all(np.isfinite(candidate))

    print("\nConfigurable candidate generation confirmed.")

    print("TEST 5 PASSED")

    # ---------------------------------------------------------
    # FINISH
    # ---------------------------------------------------------

    p.disconnect(client_id)

    print("\n" + "=" * 60)
    print("ALL TRAJECTORY GENERATOR TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()