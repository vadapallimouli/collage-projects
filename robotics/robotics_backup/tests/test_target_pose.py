import pickle

import numpy as np
import pybullet as p
import pybullet_data

from environment import Environment
from robot.panda_robot import PandaRobot


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def quaternion_distance(q1, q2):
    """
    Angular distance between two quaternions.

    PyBullet format:
        [x, y, z, w]
    """

    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.clip(
        abs(np.dot(q1, q2)),
        -1.0,
        1.0
    )

    return 2.0 * np.arccos(dot)


def main():

    print("=" * 60)
    print("MπNets TARGET POSE VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # PyBullet
    # ---------------------------------------------------------

    client_id = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id
    )

    panda = PandaRobot(robot_id)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("\nLoading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    environment = Environment.from_mpinets_problem(
        problem
    )

    # ---------------------------------------------------------
    # Print target
    # ---------------------------------------------------------

    print("\nMπNets target position:")
    print(environment.target_position)

    print("\nMπNets target quaternion [w,x,y,z]:")
    print(environment.target_quaternion)

    # Convert to PyBullet
    target_xyzw = np.array([
        environment.target_quaternion[1],
        environment.target_quaternion[2],
        environment.target_quaternion[3],
        environment.target_quaternion[0]
    ])

    print("\nTarget quaternion [x,y,z,w]:")
    print(target_xyzw)

    print("\nQuaternion norm:")
    print(np.linalg.norm(target_xyzw))

    # ---------------------------------------------------------
    # Start configuration
    # ---------------------------------------------------------

    q0 = environment.start_configuration

    print("\nMπNets start configuration:")
    print(q0)

    panda.set_joint_positions(q0)

    # ---------------------------------------------------------
    # Forward kinematics of q0
    # ---------------------------------------------------------

    state = p.getLinkState(
        robot_id,
        panda.END_EFFECTOR_LINK_INDEX,
        computeForwardKinematics=True,
        physicsClientId=client_id
    )

    current_position = np.asarray(
        state[4],
        dtype=float
    )

    current_orientation = np.asarray(
        state[5],
        dtype=float
    )

    print("\nPanda EE position at q0:")
    print(current_position)

    print("\nPanda EE orientation at q0 [x,y,z,w]:")
    print(current_orientation)

    # ---------------------------------------------------------
    # Distance from start EE to target
    # ---------------------------------------------------------

    position_distance = np.linalg.norm(
        current_position
        - environment.target_position
    )

    orientation_distance = quaternion_distance(
        current_orientation,
        target_xyzw
    )

    print("\nDistance from q0 EE to target:")
    print(
        f"Position:    {position_distance:.6f} m"
    )

    print(
        f"Orientation: {orientation_distance:.6f} rad"
    )

    # ---------------------------------------------------------
    # Try IK using ONLY position
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("POSITION-ONLY IK")
    print("-" * 60)

    limits = panda.get_joint_limits()

    lower = np.array(
        [x[0] for x in limits],
        dtype=float
    )

    upper = np.array(
        [x[1] for x in limits],
        dtype=float
    )

    ranges = upper - lower

    solution_position = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=panda.END_EFFECTOR_LINK_INDEX,
        targetPosition=environment.target_position.tolist(),
        lowerLimits=lower.tolist(),
        upperLimits=upper.tolist(),
        jointRanges=ranges.tolist(),
        restPoses=q0.tolist(),
        maxNumIterations=1000,
        residualThreshold=1e-7,
        physicsClientId=client_id
    )

    q_position = np.asarray(
        solution_position[:7],
        dtype=float
    )

    q_position = np.clip(
        q_position,
        lower + 1e-5,
        upper - 1e-5
    )

    panda.set_joint_positions(q_position)

    state = p.getLinkState(
        robot_id,
        panda.END_EFFECTOR_LINK_INDEX,
        computeForwardKinematics=True,
        physicsClientId=client_id
    )

    achieved_position = np.asarray(
        state[4],
        dtype=float
    )

    position_error = np.linalg.norm(
        achieved_position
        - environment.target_position
    )

    print("\nPosition-only IK solution:")
    print(q_position)

    print("\nAchieved position:")
    print(achieved_position)

    print("\nPosition error:")
    print(
        f"{position_error:.6f} m"
    )

    # ---------------------------------------------------------
    # Success
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TARGET POSE VERIFICATION COMPLETE")
    print("=" * 60)

    p.disconnect(
        physicsClientId=client_id
    )


if __name__ == "__main__":
    main()