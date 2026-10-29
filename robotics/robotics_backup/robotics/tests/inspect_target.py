import pickle
import numpy as np


DATASET_PATH = r"C:\Users\sharm\Downloads\both_solvable_problems.pkl"


def main():

    print("=" * 60)
    print("MπNets TARGET OBJECT INSPECTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    problem = data["tabletop"]["task_oriented"][0]

    target = problem.target
    target_volume = problem.target_volume

    # ---------------------------------------------------------
    # Basic target information
    # ---------------------------------------------------------

    print("\nTarget object type:")
    print(type(target))

    print("\nTarget class:")
    print(type(target).__name__)

    # ---------------------------------------------------------
    # Inspect SE3 internals
    # ---------------------------------------------------------

    target_xyz = object.__getattribute__(
        target,
        "_xyz"
    )

    target_so3 = object.__getattribute__(
        target,
        "_so3"
    )

    print("\nTarget _xyz:")
    print(target_xyz)

    print("\nTarget _so3 type:")
    print(type(target_so3))

    print("\nTarget _so3 __dict__:")
    print(target_so3.__dict__)

    # ---------------------------------------------------------
    # Quaternion
    # ---------------------------------------------------------

    quaternion = object.__getattribute__(
        target_so3,
        "_quat"
    )

    print("\nQuaternion object:")
    print(type(quaternion))

    print("\nQuaternion components:")

    print("w =", quaternion.w)
    print("x =", quaternion.x)
    print("y =", quaternion.y)
    print("z =", quaternion.z)

    # ---------------------------------------------------------
    # Verify quaternion norm
    # ---------------------------------------------------------

    q = np.array(
        [
            quaternion.w,
            quaternion.x,
            quaternion.y,
            quaternion.z
        ],
        dtype=float
    )

    print("\nQuaternion [w,x,y,z]:")
    print(q)

    print("\nQuaternion norm:")
    print(np.linalg.norm(q))

    # ---------------------------------------------------------
    # Target volume
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TARGET VOLUME")
    print("-" * 60)

    print("\nTarget volume type:")
    print(type(target_volume))

    print("\nTarget volume class:")
    print(target_volume.__class__.__name__)

    print("\nTarget volume attributes:")

    for key, value in target_volume.__dict__.items():

        print(f"\n{key}:")
        print(f"  type = {type(value)}")

        # -----------------------------------------------------
        # SE3 pose
        # -----------------------------------------------------

        if key == "_pose":

            pose = value

            pose_xyz = object.__getattribute__(
                pose,
                "_xyz"
            )

            pose_so3 = object.__getattribute__(
                pose,
                "_so3"
            )

            pose_quat = object.__getattribute__(
                pose_so3,
                "_quat"
            )

            print("  pose position:")
            print("   ", pose_xyz)

            print("  pose quaternion [w,x,y,z]:")
            print(
                "   ",
                [
                    pose_quat.w,
                    pose_quat.x,
                    pose_quat.y,
                    pose_quat.z
                ]
            )

        # -----------------------------------------------------
        # Numpy arrays
        # -----------------------------------------------------

        elif isinstance(value, np.ndarray):

            print("  value:")
            print("   ", value)

        # -----------------------------------------------------
        # Other values
        # -----------------------------------------------------

        else:

            print("  value:")
            print("   ", value)

    # ---------------------------------------------------------
    # Problem information
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("PROBLEM INFORMATION")
    print("-" * 60)

    print("\nProblem type:")
    print(type(problem))

    print("\nProblem attributes:")

    for key in problem.__dict__.keys():
        print(" ", key)

    # ---------------------------------------------------------
    # Extra useful information
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TARGET VOLUME DIMENSIONS")
    print("-" * 60)

    if hasattr(target_volume, "_dims"):

        dims = object.__getattribute__(
            target_volume,
            "_dims"
        )

        print("\nFull dimensions:")
        print(dims)

        print("\nHalf extents:")
        print(dims / 2.0)

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TARGET INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()