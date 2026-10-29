import time

import numpy as np
import pybullet as p
import pybullet_data

from robot.panda_robot import PandaRobot


def main():

    # --------------------------------------------------------------
    # Start PyBullet
    # --------------------------------------------------------------

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True,
    )

    # --------------------------------------------------------------
    # Create our PandaRobot abstraction
    # --------------------------------------------------------------

    robot = PandaRobot(robot_id)

    print("\n" + "=" * 60)
    print("PANDA ROBOT CLASS TEST")
    print("=" * 60)

    # --------------------------------------------------------------
    # Test DOF
    # --------------------------------------------------------------

    print("\nDOF:")
    print(robot.get_dof())

    assert robot.get_dof() == 7

    # --------------------------------------------------------------
    # Test joint limits
    # --------------------------------------------------------------

    print("\nJoint limits:")

    limits = robot.get_joint_limits()

    for i, (lower, upper) in enumerate(
        limits,
        start=1,
    ):
        print(
            f"Joint {i}: "
            f"{lower:.4f} to {upper:.4f} rad"
        )

    assert limits.shape == (7, 2)

    # --------------------------------------------------------------
    # Test a valid configuration
    # --------------------------------------------------------------

    q = np.array([
        0.0,
        -0.4,
        0.0,
        -2.0,
        0.0,
        1.6,
        0.8,
    ])

    print("\nTesting configuration:")
    print(q)

    print(
        "Within limits:",
        robot.is_within_joint_limits(q),
    )

    assert robot.is_within_joint_limits(q)

    # --------------------------------------------------------------
    # Move Panda
    # --------------------------------------------------------------

    robot.set_joint_positions(q)

    current_q = robot.get_joint_positions()

    print("\nCurrent joint configuration:")
    print(np.round(current_q, 4))

    # --------------------------------------------------------------
    # Test forward kinematics
    # --------------------------------------------------------------

    position, orientation = (
        robot.get_end_effector_pose()
    )

    print("\nEnd-effector position:")
    print(np.round(position, 4))

    print("\nEnd-effector orientation:")
    print(np.round(orientation, 4))

    assert position.shape == (3,)
    assert orientation.shape == (4,)

    print("\n" + "=" * 60)
    print("ALL PANDA ROBOT TESTS PASSED")
    print("=" * 60)

    # Keep GUI open briefly.
    for _ in range(600):
        p.stepSimulation()
        time.sleep(1 / 240)

    p.disconnect()


if __name__ == "__main__":
    main()