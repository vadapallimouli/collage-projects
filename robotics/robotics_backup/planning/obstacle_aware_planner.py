"""
Obstacle-aware motion planner for the Franka Panda robot.

PRM-style planner with support for multiple diverse candidate paths.

Pipeline:

    Random joint sampling
            ↓
    Collision-free configurations
            ↓
    Collision-free roadmap
            ↓
    K-shortest / diverse graph paths
            ↓
    Smooth joint trajectories
            ↓
    Final dense collision validation
"""

import heapq
import numpy as np


class ObstacleAwarePlanner:
    """
    PRM-style obstacle-aware planner for a 7-DOF Panda robot.
    """

    def __init__(
        self,
        panda_robot,
        collision_checker,
        trajectory_generator,
        num_samples=300,
        k_neighbors=10,
        edge_points=20,
        seed=42,
    ):

        if panda_robot is None:
            raise ValueError(
                "panda_robot cannot be None."
            )

        if collision_checker is None:
            raise ValueError(
                "collision_checker cannot be None."
            )

        if trajectory_generator is None:
            raise ValueError(
                "trajectory_generator cannot be None."
            )

        self.panda_robot = panda_robot
        self.collision_checker = collision_checker
        self.trajectory_generator = trajectory_generator

        self.num_samples = int(num_samples)
        self.k_neighbors = int(k_neighbors)
        self.edge_points = int(edge_points)
        self.seed = int(seed)

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than zero."
            )

        if self.k_neighbors <= 0:
            raise ValueError(
                "k_neighbors must be greater than zero."
            )

        if self.edge_points < 2:
            raise ValueError(
                "edge_points must be at least 2."
            )

        self.dof = int(
            panda_robot.DOF
        )

        # PandaRobot stores:
        #
        # [[lower, upper],
        #  [lower, upper],
        #  ...]
        #
        self.joint_limits = np.asarray(
            panda_robot.joint_limits,
            dtype=float
        )

        if self.joint_limits.shape != (
            self.dof,
            2
        ):
            raise ValueError(
                "Expected panda_robot.joint_limits "
                f"to have shape ({self.dof}, 2), "
                f"got {self.joint_limits.shape}"
            )

        self.lower_limits = (
            self.joint_limits[:, 0]
        )

        self.upper_limits = (
            self.joint_limits[:, 1]
        )

    # =========================================================
    # RANDOM CONFIGURATION
    # =========================================================

    def _sample_random_configuration(
        self,
        rng
    ):

        return rng.uniform(
            low=self.lower_limits,
            high=self.upper_limits
        )

    # =========================================================
    # COLLISION-FREE CONFIGURATION
    # =========================================================

    def _is_collision_free(
        self,
        configuration
    ):

        result = (
            self.collision_checker
            .check_configuration(
                configuration
            )
        )

        return not bool(
            result["collision"]
        )

    # =========================================================
    # SAMPLE FREE CONFIGURATIONS
    # =========================================================

    def sample_free_configurations(
        self,
        num_samples=None,
        seed=None
    ):

        if num_samples is None:
            num_samples = self.num_samples

        if seed is None:
            seed = self.seed

        num_samples = int(
            num_samples
        )

        if num_samples <= 0:
            return []

        rng = np.random.default_rng(
            seed
        )

        free_configurations = []

        attempts = 0

        max_attempts = max(
            num_samples * 20,
            1000
        )

        while (
            len(free_configurations)
            < num_samples
            and
            attempts < max_attempts
        ):

            q = (
                self._sample_random_configuration(
                    rng
                )
            )

            attempts += 1

            if self._is_collision_free(q):

                free_configurations.append(
                    q
                )

        return free_configurations

    # =========================================================
    # CONFIGURATION DISTANCE
    # =========================================================

    @staticmethod
    def configuration_distance(
        q1,
        q2
    ):

        q1 = np.asarray(
            q1,
            dtype=float
        )

        q2 = np.asarray(
            q2,
            dtype=float
        )

        return float(
            np.linalg.norm(
                q1 - q2
            )
        )

    # =========================================================
    # EDGE INTERPOLATION
    # =========================================================

    def _interpolate_edge(
        self,
        q1,
        q2
    ):

        q1 = np.asarray(
            q1,
            dtype=float
        )

        q2 = np.asarray(
            q2,
            dtype=float
        )

        return np.linspace(
            q1,
            q2,
            self.edge_points
        )

    # =========================================================
    # EDGE COLLISION CHECK
    # =========================================================

    def is_edge_collision_free(
        self,
        q1,
        q2
    ):

        edge = self._interpolate_edge(
            q1,
            q2
        )

        result = (
            self.collision_checker
            .check_trajectory(
                edge
            )
        )

        return bool(
            result["collision_free"]
        )

    # =========================================================
    # BUILD PRM GRAPH
    # =========================================================

    def build_graph(
        self,
        configurations
    ):

        configurations = list(
            configurations
        )

        num_nodes = len(
            configurations
        )

        graph = {
            i: []
            for i in range(num_nodes)
        }

        if num_nodes <= 1:
            return graph

        distance_matrix = np.zeros(
            (
                num_nodes,
                num_nodes
            ),
            dtype=float
        )

        # -----------------------------------------------------
        # Pairwise distances
        # -----------------------------------------------------

        for i in range(
            num_nodes
        ):

            for j in range(
                i + 1,
                num_nodes
            ):

                distance = (
                    self.configuration_distance(
                        configurations[i],
                        configurations[j]
                    )
                )

                distance_matrix[i, j] = (
                    distance
                )

                distance_matrix[j, i] = (
                    distance
                )

        # -----------------------------------------------------
        # Connect nearby nodes
        # -----------------------------------------------------

        for i in range(
            num_nodes
        ):

            neighbor_indices = np.argsort(
                distance_matrix[i]
            )

            neighbors_checked = 0

            for j in neighbor_indices:

                if j == i:
                    continue

                if neighbors_checked >= (
                    self.k_neighbors
                ):
                    break

                neighbors_checked += 1

                already_connected = any(
                    neighbor == j
                    for neighbor, _ in graph[i]
                )

                if already_connected:
                    continue

                if self.is_edge_collision_free(
                    configurations[i],
                    configurations[j]
                ):

                    distance = (
                        distance_matrix[i, j]
                    )

                    graph[i].append(
                        (
                            int(j),
                            float(distance)
                        )
                    )

                    graph[j].append(
                        (
                            int(i),
                            float(distance)
                        )
                    )

        return graph

    # =========================================================
    # CONNECT START / GOAL
    # =========================================================

    def _connect_node(
        self,
        node_index,
        configurations,
        graph,
        k=None
    ):

        if k is None:
            k = self.k_neighbors

        num_nodes = len(
            configurations
        )

        distances = []

        for i in range(
            num_nodes
        ):

            if i == node_index:
                continue

            distance = (
                self.configuration_distance(
                    configurations[node_index],
                    configurations[i]
                )
            )

            distances.append(
                (
                    distance,
                    i
                )
            )

        distances.sort(
            key=lambda x: x[0]
        )

        connections = 0

        for distance, i in distances:

            if connections >= k:
                break

            already_connected = any(
                neighbor == i
                for neighbor, _ in graph[node_index]
            )

            if already_connected:
                continue

            if self.is_edge_collision_free(
                configurations[node_index],
                configurations[i]
            ):

                graph[node_index].append(
                    (
                        int(i),
                        float(distance)
                    )
                )

                graph[i].append(
                    (
                        int(node_index),
                        float(distance)
                    )
                )

                connections += 1

        return connections

    # =========================================================
    # DIJKSTRA
    # =========================================================

    def shortest_path(
        self,
        graph,
        start_index,
        goal_index,
        banned_nodes=None,
        banned_edges=None
    ):
        """
        Dijkstra shortest path with optional temporary
        node/edge exclusions.
        """

        if banned_nodes is None:
            banned_nodes = set()

        if banned_edges is None:
            banned_edges = set()

        if start_index in banned_nodes:
            return None

        if goal_index in banned_nodes:
            return None

        distances = {
            node: float("inf")
            for node in graph
        }

        previous = {
            node: None
            for node in graph
        }

        distances[start_index] = 0.0

        queue = [
            (
                0.0,
                start_index
            )
        ]

        visited = set()

        while queue:

            current_distance, current = (
                heapq.heappop(
                    queue
                )
            )

            if current in visited:
                continue

            if current in banned_nodes:
                continue

            visited.add(current)

            if current == goal_index:
                break

            for neighbor, weight in (
                graph[current]
            ):

                if neighbor in banned_nodes:
                    continue

                edge = tuple(
                    sorted(
                        (
                            current,
                            neighbor
                        )
                    )
                )

                if edge in banned_edges:
                    continue

                new_distance = (
                    current_distance
                    + weight
                )

                if new_distance < (
                    distances[neighbor]
                ):

                    distances[neighbor] = (
                        new_distance
                    )

                    previous[neighbor] = (
                        current
                    )

                    heapq.heappush(
                        queue,
                        (
                            new_distance,
                            neighbor
                        )
                    )

        if np.isinf(
            distances[goal_index]
        ):
            return None

        path = []

        current = goal_index

        while current is not None:

            path.append(
                current
            )

            if current == start_index:
                break

            current = previous[current]

        path.reverse()

        if len(path) == 0:
            return None

        if path[0] != start_index:
            return None

        return path

    # =========================================================
    # PATH LENGTH
    # =========================================================

    def configuration_path_length(
        self,
        path,
        configurations
    ):

        if path is None:
            return float("inf")

        if len(path) < 2:
            return 0.0

        total = 0.0

        for i in range(
            len(path) - 1
        ):

            total += (
                self.configuration_distance(
                    configurations[path[i]],
                    configurations[path[i + 1]]
                )
            )

        return float(total)

    # =========================================================
    # PATH SIMILARITY
    # =========================================================

    def path_similarity(
        self,
        path_a,
        path_b
    ):

        if not path_a or not path_b:
            return 0.0

        set_a = set(path_a)
        set_b = set(path_b)

        intersection = len(
            set_a.intersection(
                set_b
            )
        )

        union = len(
            set_a.union(
                set_b
            )
        )

        if union == 0:
            return 0.0

        return (
            intersection
            /
            union
        )

    # =========================================================
    # PATH EDGE SET
    # =========================================================

    @staticmethod
    def _path_edges(
        path
    ):
        """
        Convert a node path into an undirected edge set.
        """

        edges = set()

        for i in range(
            len(path) - 1
        ):

            edge = tuple(
                sorted(
                    (
                        path[i],
                        path[i + 1]
                    )
                )
            )

            edges.add(edge)

        return edges

    # =========================================================
    # DIVERSE K-SHORTEST PATHS
    # =========================================================

    def _find_multiple_graph_paths(
        self,
        graph,
        configurations,
        start_index,
        goal_index,
        num_candidates
    ):
        """
        Generate multiple distinct graph paths.

        This uses a Yen-style K-shortest-path strategy.

        The first path is the shortest path.

        Subsequent paths are generated by taking previously
        discovered paths and creating "spur" searches where
        portions of previous paths are temporarily removed.

        This produces alternatives such as:

            Path 1:
                START -> GOAL

            Path 2:
                START -> A -> GOAL

            Path 3:
                START -> B -> C -> GOAL

            Path 4:
                START -> D -> E -> F -> GOAL

        rather than repeatedly returning the same route.
        """

        if num_candidates <= 0:
            return []

        # -----------------------------------------------------
        # First shortest path
        # -----------------------------------------------------

        first_path = self.shortest_path(
            graph,
            start_index,
            goal_index
        )

        if first_path is None:
            return []

        accepted_paths = [
            first_path
        ]

        if num_candidates == 1:
            return accepted_paths

        # -----------------------------------------------------
        # Candidate pool.
        #
        # Each item:
        #
        # (path_cost, unique_counter, path)
        #
        # The counter prevents heap comparison errors when
        # two paths have identical costs.
        # -----------------------------------------------------

        candidate_heap = []

        candidate_seen = set()

        counter = 0

        # -----------------------------------------------------
        # Generate paths 2 ... K
        # -----------------------------------------------------

        while len(accepted_paths) < (
            num_candidates
        ):

            previous_path = (
                accepted_paths[-1]
            )

            # -------------------------------------------------
            # Every node except the goal can become a spur node.
            # -------------------------------------------------

            for spur_index in range(
                len(previous_path) - 1
            ):

                root_path = (
                    previous_path[
                        :spur_index + 1
                    ]
                )

                spur_node = (
                    root_path[-1]
                )

                # -------------------------------------------------
                # Temporarily remove edges belonging to accepted
                # paths that share the same root path.
                # -------------------------------------------------

                removed_edges = set()

                for accepted_path in (
                    accepted_paths
                ):

                    if len(accepted_path) <= (
                        spur_index
                    ):
                        continue

                    same_root = (
                        accepted_path[
                            :spur_index + 1
                        ]
                        ==
                        root_path
                    )

                    if same_root:

                        edge = tuple(
                            sorted(
                                (
                                    accepted_path[
                                        spur_index
                                    ],
                                    accepted_path[
                                        spur_index + 1
                                    ]
                                )
                            )
                        )

                        removed_edges.add(
                            edge
                        )

                # -------------------------------------------------
                # Prevent loops by banning nodes in the root path
                # except the spur node.
                # -------------------------------------------------

                removed_nodes = set(
                    root_path[:-1]
                )

                # -------------------------------------------------
                # Find spur path.
                # -------------------------------------------------

                spur_path = self.shortest_path(
                    graph,
                    spur_node,
                    goal_index,
                    banned_nodes=removed_nodes,
                    banned_edges=removed_edges
                )

                if spur_path is None:
                    continue

                # -------------------------------------------------
                # Combine root + spur.
                # -------------------------------------------------

                total_path = (
                    root_path[:-1]
                    +
                    spur_path
                )

                # -------------------------------------------------
                # Must actually reach goal.
                # -------------------------------------------------

                if total_path[-1] != (
                    goal_index
                ):
                    continue

                # -------------------------------------------------
                # Exact duplicate check.
                # -------------------------------------------------

                path_key = tuple(
                    total_path
                )

                if path_key in candidate_seen:
                    continue

                if any(
                    total_path == path
                    for path in accepted_paths
                ):
                    continue

                candidate_seen.add(
                    path_key
                )

                path_cost = (
                    self.configuration_path_length(
                        total_path,
                        configurations
                    )
                )

                heapq.heappush(
                    candidate_heap,
                    (
                        path_cost,
                        counter,
                        total_path
                    )
                )

                counter += 1

            # -------------------------------------------------
            # If no alternatives were generated, stop.
            # -------------------------------------------------

            if len(candidate_heap) == 0:
                break

            # -------------------------------------------------
            # Choose the shortest sufficiently different
            # candidate.
            # -------------------------------------------------

            selected_candidate = None

            deferred_candidates = []

            while candidate_heap:

                (
                    path_cost,
                    candidate_counter,
                    candidate_path
                ) = heapq.heappop(
                    candidate_heap
                )

                # ---------------------------------------------
                # Compare against accepted paths.
                # ---------------------------------------------

                too_similar = False

                for accepted_path in (
                    accepted_paths
                ):

                    similarity = (
                        self.path_similarity(
                            candidate_path,
                            accepted_path
                        )
                    )

                    if similarity >= 0.80:

                        too_similar = True
                        break

                if too_similar:

                    # Keep it available for a future iteration.
                    deferred_candidates.append(
                        (
                            path_cost,
                            candidate_counter,
                            candidate_path
                        )
                    )

                    continue

                selected_candidate = (
                    candidate_path
                )

                break

            # -------------------------------------------------
            # Put deferred candidates back.
            # -------------------------------------------------

            for item in deferred_candidates:
                heapq.heappush(
                    candidate_heap,
                    item
                )

            # -------------------------------------------------
            # If no sufficiently different path exists, use
            # the shortest remaining path.
            # -------------------------------------------------

            if selected_candidate is None:

                if len(candidate_heap) == 0:
                    break

                (
                    _,
                    _,
                    selected_candidate
                ) = heapq.heappop(
                    candidate_heap
                )

            accepted_paths.append(
                selected_candidate
            )

        return accepted_paths

    # =========================================================
    # GRAPH PATH -> SMOOTH TRAJECTORY
    # =========================================================

    def _graph_path_to_trajectory(
        self,
        path,
        configurations,
        num_points=200
    ):

        if path is None:
            return None

        if len(path) < 2:
            raise ValueError(
                "A valid path must contain at least "
                "start and goal."
            )

        path_configurations = [
            configurations[index]
            for index in path
        ]

        start_configuration = (
            path_configurations[0]
        )

        goal_configuration = (
            path_configurations[-1]
        )

        waypoints = (
            path_configurations[1:-1]
        )

        trajectory = (
            self.trajectory_generator
            .generate_waypoint_trajectory(
                start_configuration,
                waypoints,
                goal_configuration,
                num_points=num_points
            )
        )

        return trajectory

    # =========================================================
    # FINAL TRAJECTORY VALIDATION
    # =========================================================

    def _validate_final_trajectory(
        self,
        trajectory
    ):

        return (
            self.collision_checker
            .check_trajectory(
                trajectory
            )
        )

    # =========================================================
    # SINGLE PATH PLANNING
    # =========================================================

    def plan(
        self,
        start_configuration,
        goal_configuration,
        num_points=200
    ):

        q_start = (
            self.trajectory_generator
            ._validate_configuration(
                start_configuration,
                "start_configuration"
            )
        )

        q_goal = (
            self.trajectory_generator
            ._validate_configuration(
                goal_configuration,
                "goal_configuration"
            )
        )

        # -----------------------------------------------------
        # Validate start
        # -----------------------------------------------------

        start_result = (
            self.collision_checker
            .check_configuration(
                q_start
            )
        )

        if start_result["collision"]:

            raise ValueError(
                "Start configuration is in collision."
            )

        # -----------------------------------------------------
        # Validate goal
        # -----------------------------------------------------

        goal_result = (
            self.collision_checker
            .check_configuration(
                q_goal
            )
        )

        if goal_result["collision"]:

            raise ValueError(
                "Goal configuration is in collision."
            )

        # -----------------------------------------------------
        # Sample free space
        # -----------------------------------------------------

        sampled_configurations = (
            self.sample_free_configurations()
        )

        configurations = [
            q_start
        ]

        configurations.extend(
            sampled_configurations
        )

        configurations.append(
            q_goal
        )

        start_index = 0

        goal_index = (
            len(configurations) - 1
        )

        # -----------------------------------------------------
        # Build roadmap
        # -----------------------------------------------------

        graph = self.build_graph(
            configurations
        )

        # -----------------------------------------------------
        # Reconnect start and goal
        # -----------------------------------------------------

        graph[start_index] = []
        graph[goal_index] = []

        self._connect_node(
            start_index,
            configurations,
            graph
        )

        self._connect_node(
            goal_index,
            configurations,
            graph
        )

        # -----------------------------------------------------
        # Find path
        # -----------------------------------------------------

        path = self.shortest_path(
            graph,
            start_index,
            goal_index
        )

        if path is None:

            return {
                "success": False,
                "trajectory": None,
                "path": None,
                "configurations": configurations,
                "graph": graph,
                "path_length": float("inf"),
                "num_waypoints": 0,
                "validation": None,
            }

        # -----------------------------------------------------
        # Convert to trajectory
        # -----------------------------------------------------

        trajectory = (
            self._graph_path_to_trajectory(
                path,
                configurations,
                num_points=num_points
            )
        )

        # -----------------------------------------------------
        # Dense final validation
        # -----------------------------------------------------

        validation = (
            self._validate_final_trajectory(
                trajectory
            )
        )

        return {
            "success": bool(
                validation[
                    "collision_free"
                ]
            ),

            "trajectory": trajectory,

            "path": path,

            "configurations": configurations,

            "graph": graph,

            "path_length": (
                self.configuration_path_length(
                    path,
                    configurations
                )
            ),

            "num_waypoints": max(
                0,
                len(path) - 2
            ),

            "validation": validation,
        }

    # =========================================================
    # MULTIPLE PATH PLANNING
    # =========================================================

    def plan_multiple(
        self,
        start_configuration,
        goal_configuration,
        num_candidates=10,
        num_points=200
    ):
        """
        Generate multiple diverse obstacle-aware candidate
        trajectories.
        """

        if not isinstance(
            num_candidates,
            (int, np.integer)
        ):
            raise TypeError(
                "num_candidates must be an integer."
            )

        if num_candidates <= 0:
            raise ValueError(
                "num_candidates must be greater than zero."
            )

        # -----------------------------------------------------
        # Validate configurations
        # -----------------------------------------------------

        q_start = (
            self.trajectory_generator
            ._validate_configuration(
                start_configuration,
                "start_configuration"
            )
        )

        q_goal = (
            self.trajectory_generator
            ._validate_configuration(
                goal_configuration,
                "goal_configuration"
            )
        )

        # -----------------------------------------------------
        # Check start
        # -----------------------------------------------------

        start_result = (
            self.collision_checker
            .check_configuration(
                q_start
            )
        )

        if start_result["collision"]:

            raise ValueError(
                "Start configuration is in collision."
            )

        # -----------------------------------------------------
        # Check goal
        # -----------------------------------------------------

        goal_result = (
            self.collision_checker
            .check_configuration(
                q_goal
            )
        )

        if goal_result["collision"]:

            raise ValueError(
                "Goal configuration is in collision."
            )

        # -----------------------------------------------------
        # Larger roadmap for multiple candidates.
        # -----------------------------------------------------

        sample_count = max(
            self.num_samples,
            self.num_samples
            + 50 * (
                num_candidates - 1
            )
        )

        # -----------------------------------------------------
        # Sample free configurations.
        # -----------------------------------------------------

        sampled_configurations = (
            self.sample_free_configurations(
                num_samples=sample_count,
                seed=self.seed
            )
        )

        # -----------------------------------------------------
        # Add start and goal.
        # -----------------------------------------------------

        configurations = [
            q_start
        ]

        configurations.extend(
            sampled_configurations
        )

        configurations.append(
            q_goal
        )

        start_index = 0

        goal_index = (
            len(configurations) - 1
        )

        # -----------------------------------------------------
        # Build roadmap.
        # -----------------------------------------------------

        graph = self.build_graph(
            configurations
        )

        # -----------------------------------------------------
        # Explicitly connect endpoints.
        # -----------------------------------------------------

        graph[start_index] = []
        graph[goal_index] = []

        start_connections = (
            self._connect_node(
                start_index,
                configurations,
                graph
            )
        )

        goal_connections = (
            self._connect_node(
                goal_index,
                configurations,
                graph
            )
        )

        if (
            start_connections == 0
            or
            goal_connections == 0
        ):

            return []

        # -----------------------------------------------------
        # Find K diverse graph paths.
        # -----------------------------------------------------

        graph_paths = (
            self._find_multiple_graph_paths(
                graph,
                configurations,
                start_index,
                goal_index,
                num_candidates
            )
        )

        results = []

        # -----------------------------------------------------
        # Convert each path into a trajectory.
        # -----------------------------------------------------

        for candidate_index, path in enumerate(
            graph_paths
        ):

            trajectory = (
                self._graph_path_to_trajectory(
                    path,
                    configurations,
                    num_points=num_points
                )
            )

            # -------------------------------------------------
            # Dense validation.
            # -------------------------------------------------

            validation = (
                self._validate_final_trajectory(
                    trajectory
                )
            )

            path_length = (
                self.configuration_path_length(
                    path,
                    configurations
                )
            )

            result = {
                "candidate_index": (
                    candidate_index
                ),

                "success": bool(
                    validation[
                        "collision_free"
                    ]
                ),

                "trajectory": trajectory,

                "path": path,

                "path_length": (
                    path_length
                ),

                "validation": validation,

                "num_graph_nodes": (
                    len(path)
                ),

                "num_waypoints": max(
                    0,
                    len(path) - 2
                ),
            }

            results.append(
                result
            )

        # -----------------------------------------------------
        # Organize output.
        #
        # This is NOT an ML risk score.
        # It only puts validated successful paths first.
        # -----------------------------------------------------

        results.sort(
            key=lambda result: (
                not result["success"],
                result["path_length"]
            )
        )

        for i, result in enumerate(
            results
        ):

            result["candidate_index"] = i

        return results