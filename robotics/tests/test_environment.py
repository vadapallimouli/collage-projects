import pickle
import numpy as np

from environment import Environment


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    print("Creating Environment...")

    environment = Environment.from_mpinets_problem(
        problem
    )

    print("\nEnvironment created successfully.")

    environment.summary()

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert environment.dof == 7

    assert environment.start_configuration.shape == (7,)

    assert environment.target_position.shape == (3,)

    assert environment.target_orientation.shape == (4,)

    assert environment.target_half_extents.shape == (3,)

    assert environment.target_quaternion.shape == (4,)

    assert environment.num_obstacles == 19

    assert len(environment.get_obstacle_centers()) == 19

    assert environment.get_obstacle_centers().shape == (19, 3)


    assert np.all(
        np.isfinite(
            environment.start_configuration
        )
    )

    assert np.all(
        np.isfinite(
            environment.target_position
        )
    )

    print("\n" + "=" * 60)
    print("ALL ENVIRONMENT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()