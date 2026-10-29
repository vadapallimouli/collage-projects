import pickle
import numpy as np


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def inspect_object(name, obj):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("TYPE:")
    print(type(obj))

    print("\nATTRIBUTES:")

    for attribute in dir(obj):

        if not attribute.startswith("__"):
            print(" ", attribute)

    print("\n__dict__:")

    try:

        data = object.__getattribute__(
            obj,
            "__dict__"
        )

        print("Keys:")
        print(list(data.keys()))

        for key in data.keys():

            print(f"\nFIELD: {key}")

            try:

                value = object.__getattribute__(
                    obj,
                    key
                )

                print("Type:", type(value))

                if isinstance(value, np.ndarray):

                    print("Shape:", value.shape)
                    print("Value:", value)

                elif isinstance(
                    value,
                    (int, float, str, bool)
                ):

                    print("Value:", value)

                else:

                    nested_data = getattr(
                        value,
                        "__dict__",
                        {}
                    )

                    if nested_data:

                        print(
                            "Nested keys:",
                            list(nested_data.keys())
                        )

                    print(
                        "Nested attributes:"
                    )

                    for attribute in dir(value):

                        if not attribute.startswith("__"):

                            print(
                                " ",
                                attribute
                            )

            except Exception as e:

                print(
                    "Could not inspect field:",
                    e
                )

    except Exception as e:

        print(
            "Could not access __dict__:",
            e
        )


def main():

    print("Loading MπNets dataset...")

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    # ----------------------------------------------------------
    # Find the first non-Cuboid obstacle
    # ----------------------------------------------------------

    for index, obstacle in enumerate(
        problem.obstacles
    ):

        obstacle_type = type(obstacle).__name__

        if obstacle_type != "Cuboid":

            print(
                f"\nFound non-Cuboid obstacle at index "
                f"{index}: {obstacle_type}"
            )

            inspect_object(
                f"OBSTACLE {index}",
                obstacle
            )

            break

    else:

        print(
            "\nAll obstacles in this problem are Cuboids."
        )


if __name__ == "__main__":
    main()