"""
Find visually challenging MπNets problems for the final robot-planning demo.

Run from the robotics project root:

    python -u tests/find_demo_problem.py

The script scans tabletop/task_oriented problems and ranks them using:
    1. start-to-goal Cartesian separation
    2. how close obstacles are to the direct start-goal segment
    3. whether the direct segment passes through an expanded obstacle region
    4. obstacle count

IMPORTANT:
This is a DEMO-PROBLEM SELECTION TOOL.
It does NOT change the dataset or planning algorithm.
It only identifies problems worth testing with the existing planner.
"""

from pathlib import Path
import pickle
import math

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "mpinets"
    / "both_solvable_problems.pkl"
)

ENVIRONMENT = "tabletop"
PROBLEM_SET = "task_oriented"

TOP_K = 15

# How close an obstacle must be to the direct Cartesian
# start-goal segment to count as "strongly interacting".
NEAR_THRESHOLD = 0.12

# Extra expansion used for deciding whether the direct
# Cartesian segment is blocked by an obstacle.
BLOCK_MARGIN = 0.04


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def normalize_quaternion_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)

    if q.shape != (4,):
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    return q / norm


def quaternion_to_rotation_matrix_wxyz(q):
    """
    Convert [w,x,y,z] quaternion to 3x3 rotation matrix.
    """
    w, x, y, z = normalize_quaternion_wxyz(q)

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def get_pose_components(pose):
    """
    Extract translation and quaternion from geometrout SE3
    using its current internal representation.
    """
    xyz = np.asarray(
        getattr(pose, "_xyz"),
        dtype=float,
    )

    so3 = getattr(pose, "_so3")

    quat_obj = getattr(so3, "_quat")

    q = np.array(
        [
            float(quat_obj.w),
            float(quat_obj.x),
            float(quat_obj.y),
            float(quat_obj.z),
        ],
        dtype=float,
    )

    return xyz, normalize_quaternion_wxyz(q)


def obstacle_geometry(obstacle):
    """
    Convert an MπNets obstacle into a simple geometry record.
    """

    name = type(obstacle).__name__.lower()

    pose = getattr(
        obstacle,
        "_pose",
        None,
    )

    if pose is None:
        return None

    center, quaternion = get_pose_components(
        pose
    )

    if "cuboid" in name:

        dims = np.asarray(
            getattr(obstacle, "_dims"),
            dtype=float,
        )

        half_extents = dims / 2.0

        return {
            "type": "cuboid",
            "center": center,
            "half_extents": half_extents,
            "quaternion": quaternion,
        }

    if "cylinder" in name:

        radius = float(
            getattr(obstacle, "radius")
        )

        height = float(
            getattr(obstacle, "height")
        )

        return {
            "type": "cylinder",
            "center": center,
            "radius": radius,
            "height": height,
            "quaternion": quaternion,
        }

    return None


def point_to_segment_distance(point, start, end):
    """
    Euclidean distance from point to 3D line segment.
    """
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length_sq = float(
        np.dot(direction, direction)
    )

    if length_sq < 1e-12:
        return float(
            np.linalg.norm(point - start)
        )

    t = float(
        np.dot(point - start, direction)
        / length_sq
    )

    t = np.clip(t, 0.0, 1.0)

    closest = start + t * direction

    return float(
        np.linalg.norm(point - closest)
    )


def segment_samples(start, end, count=151):
    t = np.linspace(
        0.0,
        1.0,
        count,
    )

    return (
        (1.0 - t[:, None]) * start[None, :]
        + t[:, None] * end[None, :]
    )


def obstacle_effective_radius(geometry):
    """
    Conservative spherical radius used only for ranking demo
    problems. It is NOT used by the planner or collision checker.
    """
    if geometry["type"] == "cuboid":
        return float(
            np.linalg.norm(
                geometry["half_extents"]
            )
        )

    return float(
        math.sqrt(
            geometry["radius"] ** 2
            + (geometry["height"] / 2.0) ** 2
        )
    )


def segment_obstacle_interaction(
    start,
    goal,
    geometry,
):
    """
    Approximate direct-path interaction score.

    Returns:
        minimum clearance-like distance
        blocked flag
        near fraction

    This is ONLY a ranking heuristic for selecting a demo problem.
    """

    samples = segment_samples(
        start,
        goal,
        count=151,
    )

    center = geometry["center"]

    distances = np.linalg.norm(
        samples - center[None, :],
        axis=1,
    )

    effective_radius = (
        obstacle_effective_radius(
            geometry
        )
    )

    # Conservative expanded obstacle radius.
    expanded_radius = (
        effective_radius
        + BLOCK_MARGIN
    )

    min_distance = float(
        np.min(distances)
    )

    blocked = bool(
        min_distance <= expanded_radius
    )

    near_fraction = float(
        np.mean(
            distances
            <= effective_radius
            + NEAR_THRESHOLD
        )
    )

    return (
        min_distance,
        blocked,
        near_fraction,
    )


# ============================================================
# PYBULLET START EE
# ============================================================

def get_start_ee_position(q0):
    """
    Use the real Panda URDF to obtain the Cartesian EE position
    corresponding to the MπNets start configuration.
    """

    client = p.connect(p.DIRECT)

    if client < 0:
        raise RuntimeError(
            "Could not start PyBullet."
        )

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=client,
        )

        q0 = np.asarray(
            q0,
            dtype=float,
        )

        for joint_index in range(7):
            p.resetJointState(
                robot_id,
                joint_index,
                float(q0[joint_index]),
                physicsClientId=client,
            )

        p.stepSimulation(
            physicsClientId=client
        )

        state = p.getLinkState(
            robot_id,
            11,
            physicsClientId=client,
        )

        return np.asarray(
            state[4],
            dtype=float,
        )

    finally:
        p.disconnect(
            physicsClientId=client
        )


# ============================================================
# SCORE ONE PROBLEM
# ============================================================

def score_problem(problem, problem_index):
    q0 = np.asarray(
        problem.q0,
        dtype=float,
    )

    start = get_start_ee_position(q0)

    target = np.asarray(
        problem.target._xyz,
        dtype=float,
    )

    goal_distance = float(
        np.linalg.norm(
            target - start
        )
    )

    geometries = []

    for obstacle in problem.obstacles:
        geometry = obstacle_geometry(
            obstacle
        )

        if geometry is not None:
            geometries.append(
                geometry
            )

    interactions = []

    for geometry in geometries:
        (
            minimum_distance,
            blocked,
            near_fraction,
        ) = segment_obstacle_interaction(
            start,
            target,
            geometry,
        )

        interactions.append(
            {
                "distance": minimum_distance,
                "blocked": blocked,
                "near_fraction": near_fraction,
                "type": geometry["type"],
                "center": geometry["center"],
            }
        )

    if interactions:

        minimum_obstacle_distance = min(
            item["distance"]
            for item in interactions
        )

        blocked_count = sum(
            int(item["blocked"])
            for item in interactions
        )

        near_obstacle_count = sum(
            int(
                item["distance"]
                <= NEAR_THRESHOLD
                + obstacle_effective_radius(
                    geometries[i]
                )
            )
            for i, item in enumerate(
                interactions
            )
        )

        total_near_fraction = sum(
            item["near_fraction"]
            for item in interactions
        )

    else:

        minimum_obstacle_distance = 999.0
        blocked_count = 0
        near_obstacle_count = 0
        total_near_fraction = 0.0

    # --------------------------------------------------------
    # Demo score
    #
    # We want:
    #   - meaningful start-goal distance
    #   - obstacles close to the direct path
    #   - direct path likely obstructed
    #   - several obstacles participating
    #
    # This score is NOT a planning/risk score.
    # --------------------------------------------------------

    separation_score = np.clip(
        goal_distance / 0.75,
        0.0,
        1.5,
    )

    blockage_score = min(
        blocked_count / 3.0,
        1.0,
    )

    proximity_score = np.clip(
        (0.20 - minimum_obstacle_distance)
        / 0.20,
        0.0,
        1.0,
    )

    obstacle_density_score = min(
        len(geometries) / 20.0,
        1.0,
    )

    score = (
        2.0 * separation_score
        + 4.0 * blockage_score
        + 2.0 * proximity_score
        + 1.0 * obstacle_density_score
        + 0.5 * min(
            total_near_fraction,
            2.0,
        )
    )

    return {
        "problem_index": problem_index,
        "start": start,
        "goal": target,
        "goal_distance": goal_distance,
        "obstacle_count": len(geometries),
        "minimum_obstacle_distance": (
            minimum_obstacle_distance
        ),
        "blocked_count": blocked_count,
        "near_obstacle_count": (
            near_obstacle_count
        ),
        "score": float(score),
        "interactions": interactions,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("MπNets DEMO PROBLEM SELECTOR")
    print("=" * 72)

    print()
    print(
        f"Dataset : {DATASET_PATH}"
    )
    print(
        f"Environment : {ENVIRONMENT}"
    )
    print(
        f"Problem set : {PROBLEM_SET}"
    )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "Dataset not found:\n"
            f"{DATASET_PATH}"
        )

    print()
    print("Loading dataset...")

    with open(
        DATASET_PATH,
        "rb",
    ) as f:
        dataset = pickle.load(f)

    problems = dataset[
        ENVIRONMENT
    ][
        PROBLEM_SET
    ]

    print(
        f"Problems available: "
        f"{len(problems)}"
    )

    results = []

    print()
    print(
        "Scanning problems..."
    )

    for index, problem in enumerate(
        problems
    ):

        try:

            result = score_problem(
                problem,
                index,
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"  Skipping problem "
                f"{index}: {exc}"
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    print()
    print("=" * 72)
    print(
        f"TOP {min(TOP_K, len(results))} "
        "DEMO CANDIDATES"
    )
    print("=" * 72)

    print()

    header = (
        f"{'Rank':<6}"
        f"{'Problem':<10}"
        f"{'Start→Goal':<13}"
        f"{'Min Obs Dist':<15}"
        f"{'Blocked':<10}"
        f"{'Obs':<7}"
        f"{'Score':<8}"
    )

    print(header)
    print("-" * len(header))

    for rank, result in enumerate(
        results[:TOP_K],
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{result['problem_index']:<10}"
            f"{result['goal_distance']:<13.3f}"
            f"{result['minimum_obstacle_distance']:<15.3f}"
            f"{result['blocked_count']:<10}"
            f"{result['obstacle_count']:<7}"
            f"{result['score']:<8.2f}"
        )

    # --------------------------------------------------------
    # Detailed top 5
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DETAILED TOP 5")
    print("=" * 72)

    for rank, result in enumerate(
        results[:5],
        start=1,
    ):

        print()
        print(
            f"[{rank}] Problem "
            f"{result['problem_index']}"
        )

        print(
            "    Start EE : "
            f"{np.round(result['start'], 4)}"
        )

        print(
            "    Goal     : "
            f"{np.round(result['goal'], 4)}"
        )

        print(
            "    Distance : "
            f"{result['goal_distance']:.4f} m"
        )

        print(
            "    Obstacles: "
            f"{result['obstacle_count']}"
        )

        print(
            "    Direct-path obstacle "
            f"intersections (approx.): "
            f"{result['blocked_count']}"
        )

        print(
            "    Minimum obstacle proximity: "
            f"{result['minimum_obstacle_distance']:.4f} m"
        )

        print(
            "    Demo score: "
            f"{result['score']:.3f}"
        )

        print(
            "    Obstacle types near direct path:"
        )

        nearby = sorted(
            result["interactions"],
            key=lambda x: x["distance"],
        )[:5]

        for obstacle in nearby:

            print(
                "       "
                f"{obstacle['type']:<9} "
                f"distance="
                f"{obstacle['distance']:.4f} m "
                f"blocked="
                f"{obstacle['blocked']}"
            )

    # --------------------------------------------------------
    # Recommended first candidate
    # --------------------------------------------------------

    if results:

        best = results[0]

        print()
        print("=" * 72)
        print("RECOMMENDED DEMO PROBLEM")
        print("=" * 72)

        print()
        print(
            f"Problem index: "
            f"{best['problem_index']}"
        )

        print(
            "Use this temporarily in "
            "test_final_end_to_end.py:"
        )

        print()
        print(
            f"    PROBLEM_INDEX = "
            f"{best['problem_index']}"
        )

        print()
        print(
            "NOTE:"
        )
        print(
            "This recommendation is based only on "
            "geometric demo difficulty."
        )
        print(
            "The existing planner, ML model, "
            "collision checker, and dataset are "
            "not modified."
        )

        print()
        print(
            "Next step: run the final end-to-end "
            "demo with this problem and verify "
            "the generated candidates physically."
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
