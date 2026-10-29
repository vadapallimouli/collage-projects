import pickle
import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def quaternion_angle(q1, q2):
    """
    Calculate angular difference between two quaternions.

    PyBullet quaternion format:
        [x, y, z, w]
    """

    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # q and -q represent the same rotation
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)

    return 2.0 * np.arccos(dot)


def get_joint_limit_arrays(robot):
    """
    Convert PandaRobot joint-limit format:

        [
            (lower1, upper1),
            (lower2, upper2),
            ...
        ]

    into separate lower and upper arrays.
    """

    joint_limits = robot.get_joint_limits()

    lower = np.array(
        [limit[0] for limit in joint_limits],
        dtype=float
    )

    upper = np.array(
        [limit[1] for limit in joint_limits],
        dtype=float
    )

    return lower, upper


def main():

    print("=" * 70)
    print("PANDA TARGET REACHABILITY TEST")
    print("=" * 70)

    # =========================================================
    # Load MπNets problem
    # =========================================================

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    target = problem.target

    # =========================================================
    # Target position
    # =========================================================

    target_xyz = np.asarray(
        target._xyz,
        dtype=float
    )

    # =========================================================
    # Target orientation
    #
    # MπNets / PyQuaternion:
    # [w, x, y, z]
    #
    # PyBullet:
    # [x, y, z, w]
    # =========================================================

    quat = target._so3._quat

    target_quaternion = np.array(
        [
            quat.x,
            quat.y,
            quat.z,
            quat.w
        ],
        dtype=float
    )

    print("\nTarget position:")
    print(target_xyz)

    print("\nTarget quaternion [x,y,z,w]:")
    print(target_quaternion)

    # =========================================================
    # Start PyBullet
    # =========================================================

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

    robot = PandaRobot(robot_id)

    # =========================================================
    # Get Panda joint limits
    # =========================================================

    lower, upper = get_joint_limit_arrays(robot)

    print("\nPanda joint limits:")

    for i in range(7):
        print(
            f"Joint {i + 1}: "
            f"{lower[i]:.6f} "
            f"to "
            f"{upper[i]:.6f}"
        )

    # =========================================================
    # Rest configuration
    # =========================================================

    rest_pose = np.array(
        [
            0.0,
            -0.4,
            0.0,
            -2.0,
            0.0,
            1.6,
            0.8
        ],
        dtype=float
    )

    rest_pose = np.clip(
        rest_pose,
        lower + 1e-6,
        upper - 1e-6
    )

    # =========================================================
    # 1. POSITION-ONLY IK
    # =========================================================

    print("\n" + "-" * 70)
    print("1. POSITION ONLY IK")
    print("-" * 70)

    q_position = p.calculateInverseKinematics(
        robot_id,
        robot.END_EFFECTOR_LINK_INDEX,
        targetPosition=target_xyz.tolist(),
        lowerLimits=lower.tolist(),
        upperLimits=upper.tolist(),
        jointRanges=(upper - lower).tolist(),
        restPoses=rest_pose.tolist(),
        physicsClientId=client_id
    )

    q_position = np.array(
        q_position[:7],
        dtype=float
    )

    q_position = np.clip(
        q_position,
        lower + 1e-6,
        upper - 1e-6
    )

    robot.set_joint_positions(q_position)

    # ---------------------------------------------------------
    # Position-only achieved pose
    # ---------------------------------------------------------

    achieved_position = np.array(
        robot.get_end_effector_position(),
        dtype=float
    )

    position_only_position_error = np.linalg.norm(
        achieved_position - target_xyz
    )

    achieved_orientation = np.array(
        robot.get_end_effector_pose()[1],
        dtype=float
    )

    position_only_orientation_error = quaternion_angle(
        achieved_orientation,
        target_quaternion
    )

    print("\nJoint configuration:")
    print(q_position)

    print("\nAchieved position:")
    print(achieved_position)

    print("\nTarget position:")
    print(target_xyz)

    print("\nPosition error:")
    print(
        f"{position_only_position_error:.8f} m"
    )

    print("\nAchieved orientation [x,y,z,w]:")
    print(achieved_orientation)

    print("\nTarget orientation [x,y,z,w]:")
    print(target_quaternion)

    print("\nOrientation error:")
    print(
        f"{position_only_orientation_error:.8f} rad"
    )

    # =========================================================
    # 2. POSITION + ORIENTATION IK
    # =========================================================

    print("\n" + "-" * 70)
    print("2. POSITION + ORIENTATION IK")
    print("-" * 70)

    q_full = p.calculateInverseKinematics(
        robot_id,
        robot.END_EFFECTOR_LINK_INDEX,
        targetPosition=target_xyz.tolist(),
        targetOrientation=target_quaternion.tolist(),
        lowerLimits=lower.tolist(),
        upperLimits=upper.tolist(),
        jointRanges=(upper - lower).tolist(),
        restPoses=rest_pose.tolist(),
        physicsClientId=client_id
    )

    q_full = np.array(
        q_full[:7],
        dtype=float
    )

    q_full = np.clip(
        q_full,
        lower + 1e-6,
        upper - 1e-6
    )

    robot.set_joint_positions(q_full)

    # ---------------------------------------------------------
    # Full-pose achieved pose
    # ---------------------------------------------------------

    achieved_position = np.array(
        robot.get_end_effector_position(),
        dtype=float
    )

    full_pose_position_error = np.linalg.norm(
        achieved_position - target_xyz
    )

    achieved_orientation = np.array(
        robot.get_end_effector_pose()[1],
        dtype=float
    )

    full_pose_orientation_error = quaternion_angle(
        achieved_orientation,
        target_quaternion
    )

    print("\nJoint configuration:")
    print(q_full)

    print("\nAchieved position:")
    print(achieved_position)

    print("\nTarget position:")
    print(target_xyz)

    print("\nPosition error:")
    print(
        f"{full_pose_position_error:.8f} m"
    )

    print("\nAchieved orientation [x,y,z,w]:")
    print(achieved_orientation)

    print("\nTarget orientation [x,y,z,w]:")
    print(target_quaternion)

    print("\nOrientation error:")
    print(
        f"{full_pose_orientation_error:.8f} rad"
    )

    # =========================================================
    # FINAL COMPARISON
    # =========================================================

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\nPosition-only IK:")
    print(
        f"  Position error    = "
        f"{position_only_position_error:.8f} m"
    )
    print(
        f"  Orientation error = "
        f"{position_only_orientation_error:.8f} rad"
    )

    print("\nFull-pose IK:")
    print(
        f"  Position error    = "
        f"{full_pose_position_error:.8f} m"
    )
    print(
        f"  Orientation error = "
        f"{full_pose_orientation_error:.8f} rad"
    )

    # =========================================================
    # Interpretation
    # =========================================================

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if position_only_position_error < 0.01:
        print(
            "\nPosition-only IK successfully reaches "
            "the target position."
        )
    else:
        print(
            "\nPosition-only IK did not reach the target "
            "within 1 cm."
        )

    if full_pose_orientation_error < 0.2:
        print(
            "Full-pose IK achieves a relatively close "
            "orientation."
        )
    else:
        print(
            "Full-pose IK has significant orientation error."
        )

    if full_pose_position_error > 0.05:
        print(
            "However, full-pose IK produces a large "
            "position error."
        )

    # =========================================================
    # Cleanup
    # =========================================================

    p.disconnect(client_id)

    print("\n" + "=" * 70)
    print("REACHABILITY TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()