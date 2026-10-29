"""
analysis/collision_checker.py

Collision checking for the Panda robot.

Uses PyBullet as the physical ground truth for:
- obstacle collisions
- self collisions
- joint-limit violations
- trajectory validation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pybullet as p


class CollisionChecker:
    """
    Check Panda robot configurations and trajectories
    for physical feasibility.
    """

    def __init__(
        self,
        panda_robot,
        environment,
        client_id: Optional[int] = None,
    ):
        self.panda_robot = panda_robot
        self.environment = environment
        self.client_id = (
            client_id
            if client_id is not None
            else p.getConnectionInfo()["clientIndex"]
        )

        self.robot_id = panda_robot.robot_id

        self.arm_joint_indices = tuple(
            panda_robot.ARM_JOINT_INDICES
        )

        self.joint_limits = np.asarray(
            panda_robot.joint_limits,
            dtype=float,
        )

        if self.joint_limits.shape != (7, 2):
            raise ValueError(
                "Expected Panda joint limits with shape "
                f"(7, 2), got {self.joint_limits.shape}"
            )

        self.lower_limits = (
            self.joint_limits[:, 0]
        )

        self.upper_limits = (
            self.joint_limits[:, 1]
        )

        self._last_self_collision = False
        self._last_obstacle_collision = False

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_configuration(
        self,
        configuration,
    ):

        q = np.asarray(
            configuration,
            dtype=float,
        )

        if q.shape != (7,):
            raise ValueError(
                "Configuration must have shape "
                f"(7,), got {q.shape}"
            )

        if not np.all(
            np.isfinite(q)
        ):
            raise ValueError(
                "Configuration contains "
                "non-finite values."
            )

        return q

    # ============================================================
    # SET ROBOT STATE
    # ============================================================

    def _set_configuration(
        self,
        configuration,
    ):

        q = self._validate_configuration(
            configuration
        )

        for joint_index, joint_value in zip(
            self.arm_joint_indices,
            q,
        ):

            p.resetJointState(
                self.robot_id,
                joint_index,
                float(joint_value),
                physicsClientId=self.client_id,
            )

        p.performCollisionDetection(
            physicsClientId=self.client_id
        )

    # ============================================================
    # JOINT LIMIT CHECK
    # ============================================================

    def _check_joint_limits(
        self,
        configuration,
    ):

        q = np.asarray(
            configuration,
            dtype=float,
        )

        below_lower = (
            q < self.lower_limits
        )

        above_upper = (
            q > self.upper_limits
        )

        violations = (
            below_lower
            | above_upper
        )

        return bool(
            np.any(violations)
        )

    # ============================================================
    # SELF COLLISION
    # ============================================================

    def _check_self_collision(
        self,
    ):
        """
        Check Panda self-collision.

        PyBullet may return None in some edge cases,
        so contact results are normalized to an empty
        list before processing.
        """

        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.robot_id,
            physicsClientId=self.client_id,
        )

        # Robustness:
        # PyBullet normally returns a tuple/list,
        # but can return None.
        if contacts is None:
            contacts = []

        # Ignore contacts that do not involve
        # actual penetrating/contacting robot links.
        for contact in contacts:

            if contact is None:
                continue

            # Contact distance is field 8.
            try:
                distance = float(
                    contact[8]
                )
            except (
                IndexError,
                TypeError,
                ValueError,
            ):
                continue

            if distance <= 0.0:
                return True

        return False

    # ============================================================
    # OBSTACLE IDs
    # ============================================================

    def _get_obstacle_ids(self):

        obstacle_ids = []

        try:
            num_bodies = p.getNumBodies(
                physicsClientId=self.client_id
            )
        except Exception:
            return obstacle_ids

        for body_id in range(
            num_bodies
        ):

            # Skip Panda.
            if body_id == self.robot_id:
                continue

            # Skip the plane if it exists.
            try:

                body_info = p.getBodyInfo(
                    body_id,
                    physicsClientId=self.client_id,
                )

                body_name = (
                    body_info[1]
                    if body_info
                    and len(body_info) > 1
                    else b""
                )

                if isinstance(
                    body_name,
                    bytes,
                ):

                    body_name = (
                        body_name
                        .decode(
                            errors="ignore"
                        )
                    )

                if str(body_name).lower() == "plane":
                    continue

            except Exception:
                pass

            obstacle_ids.append(
                body_id
            )

        return obstacle_ids

    # ============================================================
    # OBSTACLE COLLISION
    # ============================================================

    def _check_obstacle_collision(
        self,
    ):
        """
        Check collision between the Panda and
        all environment obstacles.

        This method is deliberately defensive because
        PyBullet contact queries may return None.
        """

        obstacle_ids = (
            self._get_obstacle_ids()
        )

        for obstacle_id in obstacle_ids:

            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_id,
                physicsClientId=self.client_id,
            )

            # ----------------------------------------------------
            # IMPORTANT FIX
            # ----------------------------------------------------
            #
            # PyBullet may return None.
            # Never call len(None).
            #
            if contacts is None:
                contacts = []

            if len(contacts) == 0:
                continue

            for contact in contacts:

                if contact is None:
                    continue

                try:
                    distance = float(
                        contact[8]
                    )
                except (
                    IndexError,
                    TypeError,
                    ValueError,
                ):
                    continue

                # A contact with distance <= 0 means
                # touching or penetration.
                if distance <= 0.0:
                    return True

        return False

    # ============================================================
    # CONFIGURATION CHECK
    # ============================================================

    def check_configuration(
        self,
        configuration,
    ):
        """
        Check a single Panda configuration.

        Returns
        -------
        dict
            {
                "collision": bool,
                "self_collision": bool,
                "obstacle_collision": bool,
                "joint_limit_violation": bool,
            }
        """

        q = self._validate_configuration(
            configuration
        )

        joint_limit_violation = (
            self._check_joint_limits(
                q
            )
        )

        # Put robot into requested state even
        # when it violates the limits. This allows
        # the physical collision check to remain
        # consistent for testing.
        self._set_configuration(
            q
        )

        self_collision = (
            self._check_self_collision()
        )

        obstacle_collision = (
            self._check_obstacle_collision()
        )

        collision = (
            joint_limit_violation
            or self_collision
            or obstacle_collision
        )

        self._last_self_collision = (
            self_collision
        )

        self._last_obstacle_collision = (
            obstacle_collision
        )

        return {
            "collision": bool(
                collision
            ),
            "self_collision": bool(
                self_collision
            ),
            "obstacle_collision": bool(
                obstacle_collision
            ),
            "joint_limit_violation": bool(
                joint_limit_violation
            ),
        }

    # ============================================================
    # TRAJECTORY CHECK
    # ============================================================

    def check_trajectory(
        self,
        trajectory,
    ):
        """
        Check every configuration in a trajectory.

        Parameters
        ----------
        trajectory:
            Array with shape (N, 7).

        Returns
        -------
        dict
        """

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "Trajectory must have shape "
                "(N, 7)."
            )

        if trajectory.shape[1] != 7:
            raise ValueError(
                "Trajectory must have 7 joints."
            )

        if len(trajectory) == 0:
            raise ValueError(
                "Trajectory cannot be empty."
            )

        configuration_results = []

        collision_count = 0
        self_collision_count = 0
        obstacle_collision_count = 0
        joint_limit_violation_count = 0

        for q in trajectory:

            result = (
                self.check_configuration(
                    q
                )
            )

            configuration_results.append(
                result
            )

            if result["collision"]:

                collision_count += 1

            if result[
                "self_collision"
            ]:

                self_collision_count += 1

            if result[
                "obstacle_collision"
            ]:

                obstacle_collision_count += 1

            if result[
                "joint_limit_violation"
            ]:

                joint_limit_violation_count += 1

        collision_free = (
            collision_count == 0
        )

        return {
            "collision_free": bool(
                collision_free
            ),
            "collision_count": int(
                collision_count
            ),
            "self_collision_count": int(
                self_collision_count
            ),
            "obstacle_collision_count": int(
                obstacle_collision_count
            ),
            "joint_limit_violation_count": int(
                joint_limit_violation_count
            ),
            "num_configurations": int(
                len(trajectory)
            ),
            "configuration_results":
                configuration_results,
        }