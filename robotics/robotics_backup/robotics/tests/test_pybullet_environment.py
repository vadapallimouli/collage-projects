import pickle

import pybullet as p
import pybullet_data

from environment import Environment
from robot.panda_robot import PandaRobot


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("PYBULLET ENVIRONMENT TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Connect to PyBullet
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

    # ---------------------------------------------------------
    # Load Panda URDF
    # ---------------------------------------------------------

    print("Loading Panda URDF...")

    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id
    )

    print(
        f"Panda URDF loaded. "
        f"PyBullet ID = {robot_id}"
    )

    # ---------------------------------------------------------
    # Create PandaRobot
    # ---------------------------------------------------------

    print("Creating PandaRobot...")

    panda = PandaRobot(robot_id)

    # ---------------------------------------------------------
    # Load MπNets dataset
    # ---------------------------------------------------------

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    # ---------------------------------------------------------
    # Create Environment
    # ---------------------------------------------------------

    print("Creating Environment...")

    environment = Environment.from_mpinets_problem(
        problem
    )

    print("\nEnvironment created successfully.")

    # ---------------------------------------------------------
    # Create obstacles in PyBullet
    # ---------------------------------------------------------

    print("\nCreating PyBullet obstacles...")

    obstacle_ids = (
        environment.create_pybullet_obstacles(
            client_id
        )
    )

    print(
        f"Created {len(obstacle_ids)} "
        f"PyBullet obstacles."
    )

    # ---------------------------------------------------------
    # Verify obstacle count
    # ---------------------------------------------------------

    assert len(obstacle_ids) == 19

    assert environment.num_obstacles == 19

    # ---------------------------------------------------------
    # Verify every obstacle
    # ---------------------------------------------------------

    print("\nObstacle IDs:")

    for i, obstacle in enumerate(
        environment.obstacles,
        start=1
    ):

        assert "pybullet_id" in obstacle

        obstacle_id = obstacle["pybullet_id"]

        assert obstacle_id >= 0

        print(
            f"  {i:02d}: "
            f"{obstacle['type']:8s} "
            f"ID = {obstacle_id}"
        )

    # ---------------------------------------------------------
    # Verify geometry counts
    # ---------------------------------------------------------

    cuboid_count = sum(
        obstacle["type"] == "Cuboid"
        for obstacle in environment.obstacles
    )

    cylinder_count = sum(
        obstacle["type"] == "Cylinder"
        for obstacle in environment.obstacles
    )

    print("\nGeometry counts:")

    print(
        f"Cuboids:   {cuboid_count}"
    )

    print(
        f"Cylinders: {cylinder_count}"
    )

    assert cuboid_count == 14
    assert cylinder_count == 5

    # ---------------------------------------------------------
    # Verify Panda
    # ---------------------------------------------------------

    assert panda.robot_id == robot_id

    print(
        f"\nPanda PyBullet ID: "
        f"{panda.robot_id}"
    )

    # ---------------------------------------------------------
    # Verify Panda has 7 arm joints
    # ---------------------------------------------------------

    assert panda.get_dof() == 7

    print(
        f"Panda DOF: "
        f"{panda.get_dof()}"
    )

    # ---------------------------------------------------------
    # Count bodies
    # ---------------------------------------------------------

    num_bodies = p.getNumBodies(
        physicsClientId=client_id
    )

    print(
        f"Total bodies in PyBullet: "
        f"{num_bodies}"
    )

    # Panda + 19 obstacles
    assert num_bodies == 20

    # ---------------------------------------------------------
    # Success
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL PYBULLET ENVIRONMENT TESTS PASSED")
    print("=" * 60)

    # ---------------------------------------------------------
    # Disconnect
    # ---------------------------------------------------------

    p.disconnect(
        physicsClientId=client_id
    )


if __name__ == "__main__":
    main()