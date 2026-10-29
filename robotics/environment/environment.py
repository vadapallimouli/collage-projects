import numpy as np
import pybullet as p


class Environment:
    """
    Environment representation for Panda robot motion planning.

    The environment is constructed from an MπNets PlanningProblem
    and contains:

    - Panda start configuration
    - Target position
    - Target orientation
    - Target volume
    - Cuboid obstacles
    - Cylinder obstacles

    The obstacle geometry can also be instantiated in PyBullet
    for collision checking and visualization.
    """

    def __init__(
        self,
        start_configuration,
        target_position,
        target_orientation,
        target_half_extents,
        target_quaternion,
        obstacles
    ):
        # ---------------------------------------------------------
        # Robot / start state
        # ---------------------------------------------------------

        self.start_configuration = np.asarray(
            start_configuration,
            dtype=float
        )

        self.dof = 7

        # ---------------------------------------------------------
        # Target
        # ---------------------------------------------------------

        self.target_position = np.asarray(
            target_position,
            dtype=float
        )

        self.target_orientation = np.asarray(
            target_orientation,
            dtype=float
        )

        self.target_half_extents = np.asarray(
            target_half_extents,
            dtype=float
        )

        self.target_quaternion = np.asarray(
            target_quaternion,
            dtype=float
        )

        # ---------------------------------------------------------
        # Obstacles
        # ---------------------------------------------------------

        self.obstacles = obstacles

        self.num_obstacles = len(self.obstacles)

        # PyBullet IDs are added later when the environment
        # is instantiated inside a PyBullet simulation.
        #
        # Example:
        # obstacle["pybullet_id"] = 5

        self.validate()

    # =============================================================
    # VALIDATION
    # =============================================================

    def validate(self):
        """
        Validate the environment data.
        """

        # Start configuration
        if self.start_configuration.shape != (7,):
            raise ValueError(
                "Start configuration must have shape (7,)"
            )

        # Target position
        if self.target_position.shape != (3,):
            raise ValueError(
                "Target position must have shape (3,)"
            )

        # Target orientation
        if self.target_orientation.shape != (4,):
            raise ValueError(
                "Target orientation must have shape (4,)"
            )

        # Target half-extents
        if self.target_half_extents.shape != (3,):
            raise ValueError(
                "Target half-extents must have shape (3,)"
            )

        # Target quaternion
        if self.target_quaternion.shape != (4,):
            raise ValueError(
                "Target quaternion must have shape (4,)"
            )

        # Obstacles
        for i, obstacle in enumerate(self.obstacles):

            if "type" not in obstacle:
                raise ValueError(
                    f"Obstacle {i + 1} has no 'type'"
                )

            if "position" not in obstacle:
                raise ValueError(
                    f"Obstacle {i + 1} has no 'position'"
                )

            if "quaternion" not in obstacle:
                raise ValueError(
                    f"Obstacle {i + 1} has no 'quaternion'"
                )

            if obstacle["type"] == "Cuboid":

                if "half_extents" not in obstacle:
                    raise ValueError(
                        f"Cuboid obstacle {i + 1} has no "
                        "'half_extents'"
                    )

                if np.asarray(
                    obstacle["half_extents"]
                ).shape != (3,):
                    raise ValueError(
                        f"Cuboid obstacle {i + 1} half-extents "
                        "must have shape (3,)"
                    )

            elif obstacle["type"] == "Cylinder":

                if "radius" not in obstacle:
                    raise ValueError(
                        f"Cylinder obstacle {i + 1} has no radius"
                    )

                if "height" not in obstacle:
                    raise ValueError(
                        f"Cylinder obstacle {i + 1} has no height"
                    )

                if obstacle["radius"] <= 0:
                    raise ValueError(
                        f"Cylinder obstacle {i + 1} radius "
                        "must be positive"
                    )

                if obstacle["height"] <= 0:
                    raise ValueError(
                        f"Cylinder obstacle {i + 1} height "
                        "must be positive"
                    )

            else:
                raise ValueError(
                    f"Unsupported obstacle type: "
                    f"{obstacle['type']}"
                )

    # =============================================================
    # MπNets CONVERSION
    # =============================================================

    @classmethod
    def from_mpinets_problem(cls, problem):
        """
        Convert an MπNets PlanningProblem into our Environment
        representation.

        The installed geometrout version exposes some of the
        required information through internal attributes, so we
        intentionally access those attributes directly rather
        than relying on broken convenience properties.
        """

        # ---------------------------------------------------------
        # Start configuration
        # ---------------------------------------------------------

        start_configuration = np.asarray(
            problem.q0,
            dtype=float
        )

        # ---------------------------------------------------------
        # Target pose
        # ---------------------------------------------------------

        target = problem.target

        target_position = cls._get_se3_position(target)

        target_quaternion = cls._get_se3_quaternion(target)

        # ---------------------------------------------------------
        # Target volume
        # ---------------------------------------------------------

        target_volume = problem.target_volume

        target_half_extents = (
            cls._get_cuboid_half_extents(target_volume)
        )

        # ---------------------------------------------------------
        # Obstacles
        # ---------------------------------------------------------

        obstacles = []

        for obstacle in problem.obstacles:

            obstacle_type = type(obstacle).__name__

            # -----------------------------------------------------
            # Cuboid
            # -----------------------------------------------------

            if obstacle_type == "Cuboid":

                position = cls._get_se3_position(
                    obstacle._pose
                )

                quaternion = cls._get_se3_quaternion(
                    obstacle._pose
                )

                # geometrout Cuboid stores FULL dimensions.
                # We need HALF extents for PyBullet.
                half_extents = (
                    np.asarray(
                        obstacle._dims,
                        dtype=float
                    ) / 2.0
                )

                obstacles.append(
                    {
                        "type": "Cuboid",
                        "position": position,
                        "half_extents": half_extents,
                        "quaternion": quaternion
                    }
                )

            # -----------------------------------------------------
            # Cylinder
            # -----------------------------------------------------

            elif obstacle_type == "Cylinder":

                position = cls._get_se3_position(
                    obstacle._pose
                )

                quaternion = cls._get_se3_quaternion(
                    obstacle._pose
                )

                radius = float(obstacle.radius)
                height = float(obstacle.height)

                obstacles.append(
                    {
                        "type": "Cylinder",
                        "position": position,
                        "radius": radius,
                        "height": height,
                        "quaternion": quaternion
                    }
                )

            else:
                raise ValueError(
                    f"Unsupported MπNets obstacle type: "
                    f"{obstacle_type}"
                )

        return cls(
            start_configuration=start_configuration,
            target_position=target_position,
            target_orientation=target_quaternion,
            target_half_extents=target_half_extents,
            target_quaternion=target_quaternion,
            obstacles=obstacles
        )

    # =============================================================
    # GEOMETROUT HELPERS
    # =============================================================

    @staticmethod
    def _get_se3_position(se3):
        """
        Extract XYZ position from a geometrout SE3 object.
        """

        xyz = object.__getattribute__(
            se3,
            "_xyz"
        )

        return np.asarray(
            xyz,
            dtype=float
        ).copy()

    @staticmethod
    def _get_se3_quaternion(se3):
        """
        Extract quaternion from a geometrout SE3 object.

        Returned format:

            [w, x, y, z]
        """

        so3 = object.__getattribute__(
            se3,
            "_so3"
        )

        quaternion = object.__getattribute__(
            so3,
            "_quat"
        )

        return np.array(
            [
                float(quaternion.w),
                float(quaternion.x),
                float(quaternion.y),
                float(quaternion.z)
            ],
            dtype=float
        )

    @staticmethod
    def _get_cuboid_half_extents(cuboid):
        """
        Extract half-extents from a geometrout Cuboid.
        """

        dimensions = object.__getattribute__(
            cuboid,
            "_dims"
        )

        dimensions = np.asarray(
            dimensions,
            dtype=float
        )

        return dimensions / 2.0

    # =============================================================
    # INFORMATION METHODS
    # =============================================================

    def get_obstacle_centers(self):
        """
        Return the centers of all obstacles.

        Returns:
            numpy array of shape (N, 3)
        """

        centers = [
            obstacle["position"]
            for obstacle in self.obstacles
        ]

        return np.asarray(
            centers,
            dtype=float
        )

    def get_obstacle_types(self):
        """
        Return obstacle types in their original order.
        """

        return [
            obstacle["type"]
            for obstacle in self.obstacles
        ]

    # =============================================================
    # PYBULLET OBSTACLE CREATION
    # =============================================================

    def create_pybullet_obstacles(self, client_id):
        """
        Create all environment obstacles inside PyBullet.

        Each obstacle receives a 'pybullet_id'.

        This allows the CollisionChecker to test:

            Panda <-> obstacle

        collisions directly through PyBullet.
        """

        obstacle_ids = []

        for obstacle in self.obstacles:

            obstacle_type = obstacle["type"]

            position = np.asarray(
                obstacle["position"],
                dtype=float
            )

            quaternion_wxyz = np.asarray(
                obstacle["quaternion"],
                dtype=float
            )

            # -----------------------------------------------------
            # Quaternion conversion
            # -----------------------------------------------------
            #
            # Our environment uses:
            #
            #     [w, x, y, z]
            #
            # PyBullet expects:
            #
            #     [x, y, z, w]
            #
            # -----------------------------------------------------

            quaternion_xyzw = [
                float(quaternion_wxyz[1]),
                float(quaternion_wxyz[2]),
                float(quaternion_wxyz[3]),
                float(quaternion_wxyz[0])
            ]

            # -----------------------------------------------------
            # CUBOID
            # -----------------------------------------------------

            if obstacle_type == "Cuboid":

                half_extents = np.asarray(
                    obstacle["half_extents"],
                    dtype=float
                )

                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=half_extents.tolist(),
                    physicsClientId=client_id
                )

                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=half_extents.tolist(),
                    rgbaColor=[
                        0.7,
                        0.7,
                        0.7,
                        1.0
                    ],
                    physicsClientId=client_id
                )

            # -----------------------------------------------------
            # CYLINDER
            # -----------------------------------------------------

            elif obstacle_type == "Cylinder":

                radius = float(
                    obstacle["radius"]
                )

                height = float(
                    obstacle["height"]
                )

                collision_shape = p.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=radius,
                    height=height,
                    physicsClientId=client_id
                )

                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=[
                        0.7,
                        0.7,
                        0.7,
                        1.0
                    ],
                    physicsClientId=client_id
                )

            else:

                raise ValueError(
                    f"Unsupported obstacle type: "
                    f"{obstacle_type}"
                )

            # -----------------------------------------------------
            # Create static body
            # -----------------------------------------------------

            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position.tolist(),
                baseOrientation=quaternion_xyzw,
                physicsClientId=client_id
            )

            # Store PyBullet ID inside the obstacle dictionary.
            obstacle["pybullet_id"] = obstacle_id

            obstacle_ids.append(
                obstacle_id
            )

        return obstacle_ids

    # =============================================================
    # SUMMARY
    # =============================================================

    def summary(self):
        """
        Print a human-readable environment summary.
        """

        print("=" * 60)
        print("ENVIRONMENT SUMMARY")
        print("=" * 60)

        print("\nRobot DOF:")
        print(self.dof)

        print("\nStart configuration:")
        print(self.start_configuration)

        print("\nTarget position:")
        print(self.target_position)

        print("\nTarget orientation [w,x,y,z]:")
        print(self.target_orientation)

        print("\nTarget half-extents:")
        print(self.target_half_extents)

        print("\nNumber of obstacles:")
        print(self.num_obstacles)

        print("\nObstacle types:")

        for i, obstacle in enumerate(
            self.obstacles,
            start=1
        ):
            print(
                f"  {i}: {obstacle['type']}"
            )

        print("=" * 60)