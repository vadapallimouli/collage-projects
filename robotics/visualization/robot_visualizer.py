# visualization/robot_visualizer.py

import time
import numpy as np
import pybullet as p
import pybullet_data


class RobotVisualizer:
    """
    Presentation-oriented PyBullet visualizer.

    Visual story:
        1. Show environment + obstacles + START/GOAL.
        2. Reveal candidate trajectories one at a time.
        3. Show a timestamp for each candidate in a fixed information area.
        4. Highlight the selected candidate in yellow.
        5. Execute only the selected trajectory.

    The planning/ML modules are not modified by this class.
    """

    # Presentation colors
    START_COLOR = (0.10, 1.00, 0.25)
    GOAL_COLOR = (1.00, 0.15, 0.15)
    CANDIDATE_COLOR = (0.15, 0.55, 1.00)
    SELECTED_COLOR = (1.00, 0.75, 0.05)
    EE_COLOR = (0.85, 0.15, 1.00)

    CUBOID_COLOR = (0.10, 0.38, 0.68, 1.0)
    CYLINDER_COLOR = (0.92, 0.18, 0.10, 1.0)
    FLOOR_COLOR = (0.16, 0.18, 0.22, 1.0)

    def __init__(
        self,
        panda_robot,
        environment,
        client_id,
        trajectory_dt=0.08,
        interpolation_steps=3,
        max_visual_points=60,
        max_path_points=40,
        candidate_reveal_time=1.2,
        candidate_pause_time=0.35,
        environment_pause_time=2.0,
        selection_pause_time=2.0,
    ):
        self.panda_robot = panda_robot
        self.environment = environment
        self.client_id = client_id

        self.trajectory_dt = float(trajectory_dt)
        self.interpolation_steps = int(interpolation_steps)
        self.max_visual_points = int(max_visual_points)
        self.max_path_points = int(max_path_points)

        self.candidate_reveal_time = float(candidate_reveal_time)
        self.candidate_pause_time = float(candidate_pause_time)
        self.environment_pause_time = float(environment_pause_time)
        self.selection_pause_time = float(selection_pause_time)

        self.robot_id = panda_robot.robot_id

        self.path_debug_ids = []
        self.marker_ids = []
        self.obstacle_ids = []
        self.obstacle_outline_ids = []

        self.text_ids = []
        self.dynamic_text_ids = []
        self.hud_ids = []

        self.moving_marker_id = None
        self.ee_label_id = None

        self.start_position = None
        self.goal_position = None

        self.scene_center = np.array([0.0, 0.0, 0.35])
        self.scene_span = 1.0

        self.candidate_timestamps = []
        self.timeline_text_id = None
        self.request_hud_id = None
        self.stage_hud_id = None
        self.analysis_hud_id = None
        self.ml_hud_id = None
        self.recoverability_hud_id = None

    # ============================================================
    # CONNECTION / GUI
    # ============================================================

    def is_connected(self):
        try:
            return bool(p.isConnected(self.client_id))
        except Exception:
            return False

    def configure_clean_gui(self):
        if not self.is_connected():
            return False

        try:
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI,
                0,
                physicsClientId=self.client_id,
            )

            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS,
                1,
                physicsClientId=self.client_id,
            )

            p.configureDebugVisualizer(
                p.COV_ENABLE_MOUSE_PICKING,
                0,
                physicsClientId=self.client_id,
            )

            return True
        except Exception:
            return False

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    def _get_obstacle_position(self, obstacle):
        try:
            return np.asarray(
                obstacle["position"],
                dtype=float,
            )
        except Exception:
            return np.zeros(3)

    def _calculate_scene_bounds(self):
        points = []

        for obstacle in getattr(
            self.environment,
            "obstacles",
            [],
        ):
            position = self._get_obstacle_position(
                obstacle
            )
            points.append(position)

        if self.start_position is not None:
            points.append(
                np.asarray(
                    self.start_position,
                    dtype=float,
                )
            )

        if self.goal_position is not None:
            points.append(
                np.asarray(
                    self.goal_position,
                    dtype=float,
                )
            )

        if not points:
            self.scene_center = np.array(
                [0.0, 0.0, 0.35]
            )
            self.scene_span = 1.0
            return

        points = np.asarray(points)

        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)

        self.scene_center = (
            minimum + maximum
        ) / 2.0

        self.scene_center[2] = max(
            0.30,
            self.scene_center[2],
        )

        self.scene_span = max(
            0.8,
            float(
                np.max(
                    maximum - minimum
                )
            ),
        )

    def _create_floor(self):
        """Create a clean solid floor instead of the distracting grid."""
        try:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    2.5,
                    2.5,
                    0.025,
                ],
                physicsClientId=self.client_id,
            )

            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    2.5,
                    2.5,
                    0.025,
                ],
                rgbaColor=list(
                    self.FLOOR_COLOR
                ),
                physicsClientId=self.client_id,
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[
                    0.0,
                    0.0,
                    -0.025,
                ],
                physicsClientId=self.client_id,
            )
        except Exception:
            # If the custom floor fails, the standard plane remains usable.
            try:
                p.loadURDF(
                    "plane.urdf",
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

    def _create_obstacle_outline(
        self,
        obstacle,
        body_id,
    ):
        """
        Add a thin outline around each obstacle.
        This makes obstacles readable even when paths overlap them.
        """
        if not self.is_connected():
            return

        position = self._get_obstacle_position(
            obstacle
        )

        obstacle_type = str(
            obstacle.get("type", "")
        ).lower()

        try:
            if obstacle_type == "cuboid":
                half = np.asarray(
                    obstacle["half_extents"],
                    dtype=float,
                )

                corners = []

                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            corners.append(
                                position
                                + np.array(
                                    [
                                        sx * half[0],
                                        sy * half[1],
                                        sz * half[2],
                                    ]
                                )
                            )

                # Corner indexing by signs.
                def c(sx, sy, sz):
                    return position + np.array(
                        [
                            sx * half[0],
                            sy * half[1],
                            sz * half[2],
                        ]
                    )

                edges = [
                    (c(-1, -1, -1), c(1, -1, -1)),
                    (c(-1, 1, -1), c(1, 1, -1)),
                    (c(-1, -1, 1), c(1, -1, 1)),
                    (c(-1, 1, 1), c(1, 1, 1)),
                    (c(-1, -1, -1), c(-1, 1, -1)),
                    (c(1, -1, -1), c(1, 1, -1)),
                    (c(-1, -1, 1), c(-1, 1, 1)),
                    (c(1, -1, 1), c(1, 1, 1)),
                    (c(-1, -1, -1), c(-1, -1, 1)),
                    (c(1, -1, -1), c(1, -1, 1)),
                    (c(-1, 1, -1), c(-1, 1, 1)),
                    (c(1, 1, -1), c(1, 1, 1)),
                ]

            elif obstacle_type == "cylinder":
                radius = float(
                    obstacle["radius"]
                )
                height = float(
                    obstacle["height"]
                )

                # A simple 12-sided vertical outline.
                edges = []
                segments = 12

                for i in range(segments):
                    a1 = (
                        2.0
                        * np.pi
                        * i
                        / segments
                    )
                    a2 = (
                        2.0
                        * np.pi
                        * (i + 1)
                        / segments
                    )

                    p1_bottom = position + np.array(
                        [
                            radius * np.cos(a1),
                            radius * np.sin(a1),
                            -height / 2.0,
                        ]
                    )
                    p2_bottom = position + np.array(
                        [
                            radius * np.cos(a2),
                            radius * np.sin(a2),
                            -height / 2.0,
                        ]
                    )
                    p1_top = position + np.array(
                        [
                            radius * np.cos(a1),
                            radius * np.sin(a1),
                            height / 2.0,
                        ]
                    )
                    p2_top = position + np.array(
                        [
                            radius * np.cos(a2),
                            radius * np.sin(a2),
                            height / 2.0,
                        ]
                    )

                    edges.extend(
                        [
                            (p1_bottom, p2_bottom),
                            (p1_top, p2_top),
                        ]
                    )

                    if i % 3 == 0:
                        edges.append(
                            (p1_bottom, p1_top)
                        )
            else:
                return

            for a, b in edges:
                debug_id = p.addUserDebugLine(
                    a.tolist(),
                    b.tolist(),
                    lineColorRGB=[
                        0.03,
                        0.03,
                        0.03,
                    ],
                    lineWidth=2.0,
                    lifeTime=0,
                    physicsClientId=self.client_id,
                )

                self.obstacle_outline_ids.append(
                    debug_id
                )

        except Exception:
            pass

    def create_environment(self):
        if not self.is_connected():
            return False

        try:
            self.configure_clean_gui()

            p.setAdditionalSearchPath(
                pybullet_data.getDataPath()
            )

            self._create_floor()

            # IMPORTANT: the end-to-end demo already creates the Panda
            # in the GUI connection before constructing RobotVisualizer.
            # Do NOT load another Panda here.  Loading a second URDF was
            # the reason two robot arms appeared in the visualization.
            existing_robot_is_valid = False
            try:
                if self.robot_id is not None and self.robot_id >= 0:
                    if p.isConnected(self.client_id):
                        joint_count = p.getNumJoints(
                            self.robot_id,
                            physicsClientId=self.client_id,
                        )
                        existing_robot_is_valid = joint_count >= 7
            except Exception:
                existing_robot_is_valid = False

            if not existing_robot_is_valid:
                self.robot_id = p.loadURDF(
                    "franka_panda/panda.urdf",
                    useFixedBase=True,
                    physicsClientId=self.client_id,
                )

            # Keep the PandaRobot wrapper synchronized with the one and
            # only robot body in this GUI connection.
            self.panda_robot.robot_id = self.robot_id

            self.obstacle_ids = []

            for obstacle in getattr(
                self.environment,
                "obstacles",
                [],
            ):
                obstacle_type = str(
                    obstacle.get("type", "")
                ).lower()

                position = self._get_obstacle_position(
                    obstacle
                )

                quaternion = np.asarray(
                    obstacle.get(
                        "quaternion",
                        [1, 0, 0, 0],
                    ),
                    dtype=float,
                )

                quat_xyzw = [
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                    float(quaternion[0]),
                ]

                if obstacle_type == "cuboid":
                    half_extents = np.asarray(
                        obstacle[
                            "half_extents"
                        ],
                        dtype=float,
                    )

                    collision_shape = (
                        p.createCollisionShape(
                            p.GEOM_BOX,
                            halfExtents=(
                                half_extents.tolist()
                            ),
                            physicsClientId=self.client_id,
                        )
                    )

                    visual_shape = (
                        p.createVisualShape(
                            p.GEOM_BOX,
                            halfExtents=(
                                half_extents.tolist()
                            ),
                            rgbaColor=list(
                                self.CUBOID_COLOR
                            ),
                            specularColor=[
                                0.25,
                                0.25,
                                0.25,
                            ],
                            physicsClientId=self.client_id,
                        )
                    )

                elif obstacle_type == "cylinder":
                    radius = float(
                        obstacle["radius"]
                    )
                    height = float(
                        obstacle["height"]
                    )

                    collision_shape = (
                        p.createCollisionShape(
                            p.GEOM_CYLINDER,
                            radius=radius,
                            height=height,
                            physicsClientId=self.client_id,
                        )
                    )

                    visual_shape = (
                        p.createVisualShape(
                            p.GEOM_CYLINDER,
                            radius=radius,
                            length=height,
                            rgbaColor=list(
                                self.CYLINDER_COLOR
                            ),
                            specularColor=[
                                0.35,
                                0.35,
                                0.35,
                            ],
                            physicsClientId=self.client_id,
                        )
                    )

                else:
                    continue

                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=(
                        collision_shape
                    ),
                    baseVisualShapeIndex=(
                        visual_shape
                    ),
                    basePosition=position.tolist(),
                    baseOrientation=quat_xyzw,
                    physicsClientId=self.client_id,
                )

                self.obstacle_ids.append(
                    body_id
                )

                self._create_obstacle_outline(
                    obstacle,
                    body_id,
                )

            print(
                f"Obstacles created: "
                f"{len(self.obstacle_ids)}"
            )

            self._calculate_scene_bounds()

            return True

        except Exception as exc:
            print(
                "ERROR creating visualization "
                f"environment: {exc}"
            )
            return False

    # ============================================================
    # CAMERA
    # ============================================================

    def set_camera(
        self,
        distance=1.25,
        yaw=42,
        pitch=-35,
        target_position=None,
        target=None,
        **kwargs,
    ):
        if target_position is None:
            target_position = target

        if target_position is None:
            target_position = self.scene_center

        if not self.is_connected():
            return False

        try:
            target_position = np.asarray(
                target_position,
                dtype=float,
            )

            if target_position.shape != (3,):
                raise ValueError(
                    "Camera target must have "
                    "shape (3,)."
                )

            distance = float(
                np.clip(
                    distance,
                    0.80,
                    1.35,
                )
            )

            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=float(yaw),
                cameraPitch=float(pitch),
                cameraTargetPosition=(
                    target_position.tolist()
                ),
                physicsClientId=self.client_id,
            )

            return True

        except Exception:
            return False

    def fit_camera(
        self,
        start_position=None,
        goal_position=None,
        distance=1.15,
        yaw=42,
        pitch=-35,
    ):
        if not self.is_connected():
            return False

        points = []

        for obstacle in getattr(
            self.environment,
            "obstacles",
            [],
        ):
            points.append(
                self._get_obstacle_position(
                    obstacle
                )
            )

        if start_position is not None:
            points.append(
                np.asarray(
                    start_position,
                    dtype=float,
                )
            )

        if goal_position is not None:
            points.append(
                np.asarray(
                    goal_position,
                    dtype=float,
                )
            )

        if points:
            points = np.asarray(points)
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            center = (
                minimum + maximum
            ) / 2.0
            center[2] = max(
                center[2],
                0.32,
            )
        else:
            center = self.scene_center

        return self.set_camera(
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            target_position=center,
        )

    # ============================================================
    # ROBOT
    # ============================================================

    def set_robot_configuration(
        self,
        configuration,
    ):
        if not self.is_connected():
            return False

        configuration = np.asarray(
            configuration,
            dtype=float,
        )

        if configuration.shape != (7,):
            raise ValueError(
                "Robot configuration must "
                "have shape (7,)."
            )

        try:
            for i, joint_index in enumerate(
                self.panda_robot.ARM_JOINT_INDICES
            ):
                p.resetJointState(
                    self.robot_id,
                    joint_index,
                    float(configuration[i]),
                    physicsClientId=self.client_id,
                )

            p.stepSimulation(
                physicsClientId=self.client_id
            )

            return True

        except Exception:
            return False

    def get_end_effector_position(self):
        if not self.is_connected():
            return None

        try:
            state = p.getLinkState(
                self.robot_id,
                self.panda_robot.END_EFFECTOR_LINK_INDEX,
                physicsClientId=self.client_id,
            )

            return np.asarray(
                state[4],
                dtype=float,
            )

        except Exception:
            return None

    # ============================================================
    # TEXT / HUD
    # ============================================================

    def add_text(
        self,
        text,
        position,
        color=(1, 1, 1),
        text_size=0.8,
    ):
        if not self.is_connected():
            return None

        try:
            text_id = p.addUserDebugText(
                str(text),
                np.asarray(
                    position,
                    dtype=float,
                ).tolist(),
                textColorRGB=list(color),
                textSize=float(text_size),
                lifeTime=0,
                physicsClientId=self.client_id,
            )

            self.text_ids.append(
                text_id
            )

            return text_id

        except Exception:
            return None

    def add_hud_text(
        self,
        text,
        position,
        color=(1, 1, 1),
        text_size=0.8,
        existing_id=None,
    ):
        if not self.is_connected():
            return existing_id

        try:
            kwargs = dict(
                text=str(text),
                textPosition=np.asarray(
                    position,
                    dtype=float,
                ).tolist(),
                textColorRGB=list(color),
                textSize=float(text_size),
                lifeTime=0,
                physicsClientId=self.client_id,
            )

            if existing_id is not None:
                kwargs[
                    "replaceItemUniqueId"
                ] = int(existing_id)

                p.addUserDebugText(
                    **kwargs
                )

                return existing_id

            text_id = p.addUserDebugText(
                **kwargs
            )

            self.hud_ids.append(
                text_id
            )

            return text_id

        except Exception:
            return existing_id

    def clear_text(self):
        if not self.is_connected():
            return

        for text_id in (
            self.text_ids
            + self.dynamic_text_ids
            + self.hud_ids
        ):
            try:
                p.removeUserDebugItem(
                    text_id,
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

        self.text_ids = []
        self.dynamic_text_ids = []
        self.hud_ids = []

    def _hud_origin(self):
        """
        World-space location deliberately far from the robot,
        in the upper-left side of the camera composition.
        """
        c = np.asarray(
            self.scene_center,
            dtype=float,
        )

        # Keep information panels closer to the camera's scene center so they
        # remain in view instead of floating far off to the side.
        return c + np.array([-0.20, -0.30, 0.95])

    def show_static_hud(
        self,
        candidate_count=None,
    ):
        self.clear_text()

        origin = self._hud_origin()

        title = (
            "INTELLIGENT MOTION PLANNER"
        )

        self.add_text(
            title,
            origin,
            color=(0.95, 0.95, 1.0),
            text_size=1.0,
        )

        self.add_text(
            "ENVIRONMENT READY",
            origin + np.array(
                [0.0, 0.0, -0.075]
            ),
            color=(0.35, 1.0, 0.45),
            text_size=0.72,
        )

        self.add_text(
            "OBSTACLES IN SCENE: "
            f"{len(self.obstacle_ids)}",
            origin + np.array(
                [0.0, 0.0, -0.145]
            ),
            color=(1.0, 0.65, 0.20),
            text_size=0.72,
        )

        if candidate_count is not None:
            self.add_text(
                "CANDIDATES: "
                f"{candidate_count}",
                origin + np.array(
                    [0.0, 0.0, -0.215]
                ),
                color=(0.25, 0.70, 1.0),
                text_size=0.72,
            )

        # Legend is deliberately placed away from the robot.
        legend = origin + np.array([0.78, 0.0, 0.0])
        self.add_text(
            "BLUE  = CANDIDATE PATHS",
            legend,
            color=self.CANDIDATE_COLOR,
            text_size=0.62,
        )
        self.add_text(
            "YELLOW = SELECTED PATH",
            legend + np.array([0.0, 0.0, -0.07]),
            color=self.SELECTED_COLOR,
            text_size=0.62,
        )
        self.add_text(
            "RED = OBSTACLES / GOAL",
            legend + np.array([0.0, 0.0, -0.14]),
            color=(1.0, 0.25, 0.18),
            text_size=0.62,
        )

    def _update_stage_hud(
        self,
        candidate_index,
        total_candidates,
        timestamp,
        paths_visible,
        status,
        selected=False,
        existing_id=None,
    ):
        origin = self._hud_origin()

        if selected:
            title_color = (
                1.0,
                0.78,
                0.10,
            )
        else:
            title_color = (
                0.25,
                0.75,
                1.0,
            )

        text = (
            "INTELLIGENT MOTION PLANNER\n"
            f"Candidate: "
            f"{candidate_index}/{total_candidates}\n"
            f"Timestamp: "
            f"{timestamp:05.2f} s\n"
            f"Paths visible: "
            f"{paths_visible}/{total_candidates}\n"
            f"Status: {status}"
        )

        return self.add_hud_text(
            text,
            origin,
            color=title_color,
            text_size=0.72,
            existing_id=existing_id,
        )

    # ============================================================
    # START / GOAL
    # ============================================================

    def clear_markers(self):
        if not self.is_connected():
            return

        for marker_id in self.marker_ids:
            try:
                p.removeBody(
                    marker_id,
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

        self.marker_ids = []
        self.moving_marker_id = None

    def add_marker(
        self,
        position,
        color,
        size=0.05,
    ):
        if not self.is_connected():
            return None

        try:
            position = np.asarray(
                position,
                dtype=float,
            )

            if position.shape != (3,):
                return None

            visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=float(size),
                rgbaColor=[
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                    1.0,
                ],
                physicsClientId=self.client_id,
            )

            body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual,
                basePosition=position.tolist(),
                physicsClientId=self.client_id,
            )

            self.marker_ids.append(body)

            return body

        except Exception:
            return None

    def mark_start_and_goal(
        self,
        trajectory,
        target_position=None,
        target=None,
        **kwargs,
    ):
        if target_position is None:
            target_position = target

        if not self.is_connected():
            return False

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 7
            or len(trajectory) == 0
        ):
            return False

        self.clear_markers()

        self.set_robot_configuration(
            trajectory[0]
        )

        start = (
            self.get_end_effector_position()
        )

        if start is not None:
            self.start_position = np.asarray(
                start,
                dtype=float,
            )

            self.add_marker(
                self.start_position,
                self.START_COLOR,
                size=0.075,
            )

            self.add_text(
                "START",
                self.start_position + np.array([0.0, 0.0, 0.10]),
                color=self.START_COLOR,
                text_size=0.62,
            )


        if target_position is not None:
            target_position = np.asarray(
                target_position,
                dtype=float,
            )

            if target_position.shape != (3,):
                raise ValueError(
                    "target_position must be "
                    "XYZ coordinates."
                )

            self.goal_position = (
                target_position.copy()
            )

            self.add_marker(
                self.goal_position,
                self.GOAL_COLOR,
                size=0.085,
            )

            self.add_text(
                "GOAL",
                self.goal_position + np.array([0.0, 0.0, 0.10]),
                color=self.GOAL_COLOR,
                text_size=0.62,
            )


        self._calculate_scene_bounds()

        return True

    # ============================================================
    # TRAJECTORY PROCESSING
    # ============================================================

    def downsample_trajectory(
        self,
        trajectory,
        max_points=None,
    ):
        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if len(trajectory) <= 1:
            return trajectory

        if max_points is None:
            max_points = self.max_visual_points

        max_points = max(
            2,
            int(max_points),
        )

        if len(trajectory) <= max_points:
            return trajectory

        indices = np.linspace(
            0,
            len(trajectory) - 1,
            max_points,
            dtype=int,
        )

        return trajectory[
            np.unique(indices)
        ]

    def interpolate_trajectory(
        self,
        trajectory,
        steps=None,
    ):
        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if len(trajectory) < 2:
            return trajectory

        if steps is None:
            steps = self.interpolation_steps

        steps = max(
            1,
            int(steps),
        )

        result = []

        for i in range(
            len(trajectory) - 1
        ):
            q0 = trajectory[i]
            q1 = trajectory[i + 1]

            for j in range(steps):
                t = j / float(steps)

                # Smoothstep.
                s = (
                    3.0 * t * t
                    - 2.0 * t * t * t
                )

                result.append(
                    (1.0 - s) * q0
                    + s * q1
                )

        result.append(
            trajectory[-1]
        )

        return np.asarray(
            result,
            dtype=float,
        )

    # ============================================================
    # PATH GEOMETRY
    # ============================================================

    def _trajectory_to_ee_positions(
        self,
        trajectory,
        max_points=None,
    ):
        trajectory = self.downsample_trajectory(
            trajectory,
            max_points,
        )

        original = None

        try:
            original = np.array(
                [
                    p.getJointState(
                        self.robot_id,
                        joint_index,
                        physicsClientId=self.client_id,
                    )[0]
                    for joint_index
                    in self.panda_robot.ARM_JOINT_INDICES
                ],
                dtype=float,
            )

            positions = []

            for q in trajectory:
                if not self.is_connected():
                    break

                self.set_robot_configuration(
                    q
                )

                ee = (
                    self.get_end_effector_position()
                )

                if ee is not None:
                    positions.append(
                        ee.tolist()
                    )

            return np.asarray(
                positions,
                dtype=float,
            )

        finally:
            if (
                original is not None
                and self.is_connected()
            ):
                self.set_robot_configuration(
                    original
                )

    def _draw_ee_polyline(
        self,
        positions,
        color,
        width,
    ):
        if len(positions) < 2:
            return []

        ids = []

        for i in range(
            len(positions) - 1
        ):
            if not self.is_connected():
                break

            try:
                debug_id = p.addUserDebugLine(
                    positions[i].tolist(),
                    positions[i + 1].tolist(),
                    lineColorRGB=list(color),
                    lineWidth=float(width),
                    lifeTime=0,
                    physicsClientId=self.client_id,
                )

                ids.append(debug_id)

            except Exception:
                break

        self.path_debug_ids.extend(ids)

        return ids

    def clear_path(self):
        if not self.is_connected():
            self.path_debug_ids = []
            return

        for debug_id in self.path_debug_ids:
            try:
                p.removeUserDebugItem(
                    debug_id,
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

        self.path_debug_ids = []

    # ============================================================
    # STAGED CANDIDATE REVEAL
    # ============================================================

    def _reveal_single_candidate(
        self,
        trajectory,
        candidate_index,
        total_candidates,
        timestamp,
    ):
        positions = (
            self._trajectory_to_ee_positions(
                trajectory,
                max_points=self.max_path_points,
            )
        )

        if len(positions) < 2:
            return

        # Draw gradually, so the audience sees
        # the candidate being constructed.
        segment_count = len(
            positions
        )

        if segment_count <= 1:
            return

        segment_delay = (
            self.candidate_reveal_time
            / float(segment_count - 1)
        )

        for i in range(
            segment_count - 1
        ):
            if not self.is_connected():
                return

            try:
                debug_id = p.addUserDebugLine(
                    positions[i].tolist(),
                    positions[i + 1].tolist(),
                    lineColorRGB=list(
                        self.CANDIDATE_COLOR
                    ),
                    # Thin blue alternatives keep the scene readable.
                    lineWidth=1.0,
                    lifeTime=0,
                    physicsClientId=self.client_id,
                )
                self.path_debug_ids.append(
                    debug_id
                )
            except Exception:
                pass

            time.sleep(
                segment_delay
            )

        self.stage_hud_id = self._update_stage_hud(
            candidate_index,
            total_candidates,
            timestamp,
            candidate_index,
            "CANDIDATE READY",
            existing_id=self.stage_hud_id,
        )

    def draw_multiple_trajectories(
        self,
        trajectories,
        selected_index=None,
        normal_color=None,
        selected_color=None,
        normal_width=2.0,
        selected_width=5.5,
        staged=True,
        highlight_after=False,
        **kwargs,
    ):
        """
        Main presentation method.

        If staged=True, candidates are revealed one by one.
        This is intentionally slower for a project demonstration.
        """

        if not self.is_connected():
            return 0

        valid = []

        for trajectory in trajectories or []:
            arr = np.asarray(
                trajectory,
                dtype=float,
            )

            if (
                arr.ndim == 2
                and arr.shape[1] == 7
                and len(arr) >= 2
            ):
                valid.append(arr)

        if not valid:
            return 0

        if normal_color is not None:
            self.CANDIDATE_COLOR = tuple(
                normal_color
            )

        if selected_color is not None:
            self.SELECTED_COLOR = tuple(
                selected_color
            )

        self.clear_path()

        total = len(valid)

        # Keep the USER REQUEST panel visible while paths are generated.
        # Do not clear it here: the audience should first see the
        # problem statement, then watch candidate paths appear.
        self._calculate_scene_bounds()

        time.sleep(
            self.environment_pause_time
        )

        start_time = time.time()
        self.candidate_timestamps = []

        # Reveal one candidate at a time.
        for index, trajectory in enumerate(
            valid
        ):

            if not self.is_connected():
                return index

            timestamp = (
                time.time()
                - start_time
            )

            self.candidate_timestamps.append(
                timestamp
            )

            self.stage_hud_id = (
                self._update_stage_hud(
                    index + 1,
                    total,
                    timestamp,
                    index,
                    "GENERATING PATH...",
                    existing_id=self.stage_hud_id,
                )
            )

            self._reveal_single_candidate(
                trajectory,
                index + 1,
                total,
                timestamp,
            )

            if index < total - 1:
                time.sleep(
                    self.candidate_pause_time
                )

        # Selection is intentionally separated from path generation.
        # This lets the presentation show ML explanation and
        # recoverability before the best path turns yellow.
        selected_index = (
            selected_index
            if selected_index is not None
            else 0
        )

        selected_index = int(
            np.clip(
                selected_index,
                0,
                total - 1,
            )
        )

        selection_time = (
            time.time()
            - start_time
        )

        if highlight_after:
            self._update_stage_hud(
                selected_index + 1,
                total,
                selection_time,
                total,
                "BEST PATH SELECTED",
                selected=True,
                existing_id=self.timeline_text_id,
            )

            selected_positions = (
                self._trajectory_to_ee_positions(
                    valid[selected_index],
                    max_points=self.max_path_points,
                )
            )

            self._draw_ee_polyline(
                selected_positions,
                self.SELECTED_COLOR,
                selected_width,
            )

            time.sleep(
                self.selection_pause_time
            )

        # Compact timestamp summary at the upper-right
        # side of the same information area.
        origin = self._hud_origin()

        timeline_lines = [
            "PATH TIMELINE"
        ]

        for i, timestamp in enumerate(
            self.candidate_timestamps
        ):
            timeline_lines.append(
                f"P{i + 1}: "
                f"{timestamp:04.1f}s"
            )

        # Keep it compact by putting 5 paths per line.
        timeline = "PATH TIMELINE\n"

        for start in range(
            0,
            len(
                self.candidate_timestamps
            ),
            5,
        ):
            items = []

            for j in range(
                start,
                min(
                    start + 5,
                    len(
                        self.candidate_timestamps
                    ),
                ),
            ):
                items.append(
                    f"P{j + 1} "
                    f"{self.candidate_timestamps[j]:.1f}s"
                )

            timeline += (
                " | ".join(items)
                + "\n"
            )

        # Keep the timestamp timeline separate from the main
        # information panel so the robot/paths stay unobstructed.
        timeline_origin = origin + np.array([0.95, 0.10, 0.02])
        self.timeline_text_id = (
            self.add_hud_text(
                timeline.rstrip(),
                timeline_origin,
                color=(0.90, 0.92, 0.98),
                text_size=0.66,
                existing_id=self.timeline_text_id,
            )
        )

        print(
            f"Candidate trajectories "
            f"displayed: {total}"
        )
        if highlight_after:
            print(
                "Selected trajectory highlighted: "
                f"Candidate {selected_index + 1}"
            )
        else:
            print(
                "Candidate paths displayed; selection deferred to analysis stage."
            )

        return total

    def draw_trajectory(
        self,
        trajectory,
        color=None,
        line_width=1.0,
        max_points=None,
        **kwargs,
    ):
        if color is None:
            color = self.CANDIDATE_COLOR

        if max_points is None:
            max_points = self.max_path_points

        positions = (
            self._trajectory_to_ee_positions(
                trajectory,
                max_points=max_points,
            )
        )

        return self._draw_ee_polyline(
            positions,
            color,
            line_width,
        )

    # ============================================================
    # SELECTED PATH EMPHASIS
    # ============================================================

    def highlight_selected_trajectory(
        self,
        trajectory,
        color=None,
        line_width=6.0,
    ):
        if color is None:
            color = self.SELECTED_COLOR

        positions = (
            self._trajectory_to_ee_positions(
                trajectory,
                max_points=self.max_path_points,
            )
        )

        return self._draw_ee_polyline(
            positions,
            color,
            line_width,
        )

    # ============================================================
    # PRESENTATION STAGES
    # ============================================================

    def _replace_panel(self, text, position, color, existing_id=None, size=0.66):
        return self.add_hud_text(
            text,
            position,
            color=color,
            text_size=size,
            existing_id=existing_id,
        )

    def _remove_hud_item(self, item_id):
        if item_id is None or not self.is_connected():
            return
        try:
            p.removeUserDebugItem(
                int(item_id),
                physicsClientId=self.client_id,
            )
        except Exception:
            pass

    def show_user_input_panel(
        self,
        start_position,
        goal_position,
        obstacle_count,
        candidate_count,
        hold_time=3.0,
    ):
        """Show the problem presented to the planner before path generation."""
        self.clear_text()
        origin = self._hud_origin()
        start = np.asarray(start_position, dtype=float)
        goal = np.asarray(goal_position, dtype=float)

        text = (
            "USER REQUEST\n"
            f"START   ({start[0]:+.2f}, {start[1]:+.2f}, {start[2]:+.2f})\n"
            f"GOAL    ({goal[0]:+.2f}, {goal[1]:+.2f}, {goal[2]:+.2f})\n"
            f"OBSTACLES   {int(obstacle_count)}\n"
            f"CANDIDATES  {int(candidate_count)}\n"
            "STATUS   READY TO PLAN"
        )
        self.request_hud_id = self._replace_panel(
            text,
            origin,
            (0.92, 0.95, 1.0),
            existing_id=None,
            size=0.72,
        )
        time.sleep(float(hold_time))
        return self.request_hud_id

    def show_ml_explanation_panel(
        self,
        predictions,
        selected_index,
        hold_time=4.0,
    ):
        """Show a compact, human-readable XGBoost explanation."""
        origin = self._hud_origin()
        selected = None
        failed = None

        for item in predictions or []:
            idx = int(item.get("trajectory_index", -1))
            if idx == int(selected_index):
                selected = item
            if failed is None and str(item.get("predicted_class", "")).lower() == "failure":
                failed = item

        if selected is None and predictions:
            selected = predictions[min(int(selected_index), len(predictions)-1)]
        if failed is None and predictions:
            failed = max(predictions, key=lambda x: float(x.get("failure_probability", 0.0)))

        def top_factors(item):
            if not item:
                return []
            c = item.get("local_contributions", item.get("feature_contributions", {}))
            if isinstance(c, dict):
                return sorted(c.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:2]
            if isinstance(c, list):
                out=[]
                for x in c[:5]:
                    if isinstance(x, dict):
                        out.append((x.get("feature", "feature"), float(x.get("contribution", 0.0))))
                return sorted(out, key=lambda kv: abs(kv[1]), reverse=True)[:2]
            return []

        # Hide the temporary path-generation status while the analysis panel is shown.
        self._remove_hud_item(self.stage_hud_id)
        self.stage_hud_id = None

        lines = ["ML FAILURE PREDICTION", ""]
        if failed is not None:
            idx = int(failed.get("trajectory_index", 0)) + 1
            prob = float(failed.get("failure_probability", 0.0)) * 100.0
            lines.append(f"P{idx}  FAILURE RISK   {prob:5.1f}%")
            for name, value in top_factors(failed):
                direction = "failure" if value > 0 else "success"
                lines.append(f"  {name[:27]}  {value:+.2f}  → {direction}")

        if selected is not None:
            idx = int(selected.get("trajectory_index", selected_index)) + 1
            prob = float(selected.get("success_probability", 0.0)) * 100.0
            lines.append("")
            lines.append(f"P{idx}  SELECTED CANDIDATE   {prob:5.1f}% SUCCESS")
            for name, value in top_factors(selected):
                direction = "failure" if value > 0 else "success"
                lines.append(f"  {name[:27]}  {value:+.2f}  → {direction}")

        lines.append("")
        lines.append("Positive contribution → failure")
        lines.append("Negative contribution → success")
        lines.append("")
        lines.append("ML prediction is checked by physical collision validation.")

        self.ml_hud_id = self._replace_panel(
            "\n".join(lines),
            origin + np.array([0.0, 0.0, -0.10]),
            (1.0, 0.88, 0.20),
            existing_id=self.ml_hud_id,
            size=0.82,
        )
        time.sleep(float(hold_time))
        return self.ml_hud_id

    def show_recoverability_panel(
        self,
        recoverability,
        feasible_count,
        total_count,
        hold_time=3.0,
    ):
        """Show the recovery capability without inventing a recovery event."""
        self._remove_hud_item(self.stage_hud_id)
        self.stage_hud_id = None

        origin = self._hud_origin()
        result = recoverability if isinstance(recoverability, dict) else {}
        recoverable = result.get("recoverable_count", 0)
        unrecoverable = result.get("unrecoverable_count", 0)
        failed = total_count - feasible_count

        if failed == 0:
            text = (
                "RECOVERABILITY ANALYSIS\n"
                f"CANDIDATES          {int(total_count)}\n"
                f"FEASIBLE            {int(feasible_count)}/{int(total_count)}\n"
                "FAILED              0\n"
                "RECOVERY TRIGGERED  NO\n"
                "ALTERNATIVES        READY\n"
                "STATUS              NO RECOVERY REQUIRED"
            )
        else:
            text = (
                "RECOVERABILITY ANALYSIS\n"
                f"CANDIDATES          {int(total_count)}\n"
                f"FEASIBLE            {int(feasible_count)}/{int(total_count)}\n"
                f"FAILED              {int(failed)}\n"
                f"RECOVERABLE         {int(recoverable)}\n"
                f"UNRECOVERABLE       {int(unrecoverable)}\n"
                "STATUS              RECOVERY AVAILABLE"
            )

        self.recoverability_hud_id = self._replace_panel(
            text,
            origin + np.array([0.0, 0.0, -0.68]),
            (0.45, 0.95, 1.0),
            existing_id=self.recoverability_hud_id,
            size=0.82,
        )
        time.sleep(float(hold_time))
        return self.recoverability_hud_id

    def highlight_selected_path(
        self,
        trajectory,
        selected_index,
        total_candidates,
        hold_time=3.0,
        selected_prediction=None,
        feasible_count=None,
    ):
        """Turn the chosen candidate yellow and explicitly explain why it was selected."""
        positions = self._trajectory_to_ee_positions(
            trajectory,
            max_points=self.max_path_points,
        )
        self._draw_ee_polyline(
            positions,
            self.SELECTED_COLOR,
            6.0,
        )

        origin = self._hud_origin()

        if selected_prediction is not None:
            success = float(selected_prediction.get("success_probability", 0.0)) * 100.0
            failure = float(selected_prediction.get("failure_probability", 0.0)) * 100.0
        else:
            success = None
            failure = None

        lines = [
            "FINAL SELECTION",
            f"CANDIDATE {int(selected_index) + 1}/{int(total_candidates)}",
        ]
        if success is not None:
            lines.append(f"ML SUCCESS PROBABILITY   {success:5.2f}%")
            lines.append(f"ML FAILURE PROBABILITY   {failure:5.2f}%")
        if feasible_count is not None:
            lines.append(f"PHYSICALLY FEASIBLE       {int(feasible_count)}/{int(total_candidates)}")
        lines.extend([
            "",
            "SELECTION BASIS",
            "Highest ML success probability",
            "among physically feasible candidates.",
            "",
            "YELLOW = SELECTED FEASIBLE PATH",
            "READY FOR ROBOT EXECUTION",
        ])

        self.analysis_hud_id = self._replace_panel(
            "\n".join(lines),
            origin + np.array([0.0, 0.0, -1.05]),
            (1.0, 0.78, 0.08),
            existing_id=self.analysis_hud_id,
            size=0.78,
        )
        time.sleep(float(hold_time))
        return True

    # ============================================================
    # ANIMATION
    # ============================================================

    def animate_trajectory(
        self,
        trajectory,
        frame_delay=None,
        loop=False,
        num_loops=1,
        hold_at_end=False,
        hold_time=3.0,
        candidate_index=None,
        total_candidates=None,
        **kwargs,
    ):
        if not self.is_connected():
            return False

        if frame_delay is None:
            frame_delay = self.trajectory_dt

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 7
            or len(trajectory) == 0
        ):
            raise ValueError(
                "Trajectory must have shape (N,7)."
            )

        visual = self.downsample_trajectory(
            trajectory,
            self.max_visual_points,
        )

        frames = self.interpolate_trajectory(
            visual,
            self.interpolation_steps,
        )

        total_frames = len(frames)

        estimated = (
            max(0, total_frames - 1)
            * float(frame_delay)
        )

        print()
        print(
            "Trajectory animation:"
        )
        print(
            f"  Original configurations : "
            f"{len(trajectory)}"
        )
        print(
            f"  Visual configurations   : "
            f"{len(visual)}"
        )
        print(
            f"  Animation frames        : "
            f"{total_frames}"
        )
        print(
            f"  Frame interval          : "
            f"{float(frame_delay):.3f} s"
        )
        print(
            f"  Estimated duration      : "
            f"{estimated:.2f} s"
        )

        try:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.045,
                rgbaColor=[
                    *self.EE_COLOR,
                    1.0,
                ],
                physicsClientId=self.client_id,
            )

            self.moving_marker_id = (
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[
                        0,
                        0,
                        0,
                    ],
                    physicsClientId=self.client_id,
                )
            )

            self.marker_ids.append(
                self.moving_marker_id
            )

            # The label follows the moving end effector.
            self.ee_label_id = p.addUserDebugText(
                "END EFFECTOR",
                [0, 0, 0.08],
                textColorRGB=list(self.EE_COLOR),
                textSize=0.65,
                lifeTime=0,
                physicsClientId=self.client_id,
            )

        except Exception:
            self.moving_marker_id = None
            self.ee_label_id = None

        origin = self._hud_origin()
        hud_id = self.analysis_hud_id

        start_time = time.time()

        for frame_index, configuration in enumerate(
            frames
        ):

            if not self.is_connected():
                return False

            self.set_robot_configuration(
                configuration
            )

            ee = (
                self.get_end_effector_position()
            )

            if (
                ee is not None
                and self.moving_marker_id is not None
            ):
                try:
                    p.resetBasePositionAndOrientation(
                        self.moving_marker_id,
                        ee.tolist(),
                        [
                            0,
                            0,
                            0,
                            1,
                        ],
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    pass

                if self.ee_label_id is not None:
                    try:
                        label_position = (
                            np.asarray(ee, dtype=float)
                            + np.array([0.0, 0.0, 0.085])
                        )

                        p.addUserDebugText(
                            "END EFFECTOR",
                            label_position.tolist(),
                            textColorRGB=list(self.EE_COLOR),
                            textSize=0.65,
                            lifeTime=0,
                            replaceItemUniqueId=int(
                                self.ee_label_id
                            ),
                            physicsClientId=self.client_id,
                        )
                    except Exception:
                        pass

            if total_frames > 1:
                progress = (
                    frame_index
                    / float(
                        total_frames - 1
                    )
                )
            else:
                progress = 1.0

            elapsed = (
                frame_index
                * float(frame_delay)
            )

            candidate_text = (
                f"Candidate "
                f"{candidate_index + 1}"
                f"/{total_candidates}"
                if candidate_index is not None
                and total_candidates is not None
                else "Selected trajectory"
            )

            text = (
                "EXECUTING SELECTED PATH\n"
                f"{candidate_text}\n"
                f"Time: {elapsed:05.2f} s\n"
                f"Progress: "
                f"{100.0 * progress:05.1f}%\n"
                "Status: EXECUTING"
            )

            hud_id = self.add_hud_text(
                text,
                origin,
                color=(
                    1.0,
                    0.78,
                    0.10,
                ),
                text_size=0.72,
                existing_id=hud_id,
            )

            if (
                frame_index == 0
                or frame_index % 20 == 0
                or frame_index == total_frames - 1
            ):
                print(
                    "\r"
                    f"  Time: {elapsed:6.2f}s | "
                    f"Progress: "
                    f"{100.0 * progress:5.1f}%",
                    end="",
                    flush=True,
                )

            time.sleep(
                float(frame_delay)
            )

        print()

        if self.is_connected():
            final_elapsed = (
                max(
                    0,
                    total_frames - 1,
                )
                * float(frame_delay)
            )

            self.add_hud_text(
                "EXECUTION COMPLETE\n"
                f"Candidate "
                f"{candidate_index + 1}"
                f"/{total_candidates}\n"
                f"Time: "
                f"{final_elapsed:05.2f} s\n"
                "Progress: 100.0%\n"
                "Status: COMPLETED",
                origin,
                color=(
                    0.3,
                    1.0,
                    0.4,
                ),
                text_size=0.72,
                existing_id=hud_id,
            )

        if hold_at_end and self.is_connected():
            self.set_robot_configuration(
                trajectory[-1]
            )

            print(
                "Holding at final "
                "configuration for "
                f"{float(hold_time):.2f} seconds..."
            )

            start_hold = time.time()

            while (
                self.is_connected()
                and (
                    time.time()
                    - start_hold
                )
                < float(hold_time)
            ):
                time.sleep(0.05)

        return True

    # ============================================================
    # CLEANUP
    # ============================================================

    def clear_all(self):
        self.clear_path()
        self.clear_markers()
        self.clear_text()

        for debug_id in (
            self.obstacle_outline_ids
        ):
            try:
                p.removeUserDebugItem(
                    debug_id,
                    physicsClientId=self.client_id,
                )
            except Exception:
                pass

        self.obstacle_outline_ids = []

    def show_trajectory(
        self,
        trajectory,
        animate=True,
        **kwargs,
    ):
        self.clear_path()
        self.draw_trajectory(
            trajectory
        )

        if animate:
            return self.animate_trajectory(
                trajectory,
                **kwargs,
            )

        return True

    def wait_until_closed(
        self,
        poll_interval=0.1,
    ):
        print()
        print(
            "Waiting for PyBullet window..."
        )

        while self.is_connected():
            time.sleep(
                poll_interval
            )

        print(
            "PyBullet window closed."
        )
