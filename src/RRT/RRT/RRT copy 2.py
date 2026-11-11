import rclpy

from rclpy.node import Node


from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 

from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree,cKDTree
import numpy as np
import os
from shapely import STRtree, LineString
import time


angle_limits = np.array([
    [-np.pi/2, np.pi/2],
    [-3*np.pi/4, 3*np.pi/4],
    [-3*np.pi/4, 3*np.pi/4]
])


angle_path="/home/cotrobot/mushroom_harvesting_scara_ws/resources/angle_cache/angles_pos.pkl"
if not os.path.exists(angle_path):
    # print("buliding kinematics tree")
    # random_angles=np.array([[np.random.uniform (*angle_limits[i]) for i in range(0,3)] for _ in tqdm(range(1_000_000))])
    # random_position=forward_kinematics_batch(random_angles)
    # angle_dict={
    #     "angles": random_angles,
    #     "positions": random_position,
    # }

    # with open(angle_path,"wb") as file:
    #     pickle.dump(angle_dict,file)
    print("file not found")
else:
    print("loading pickle")
    with open(angle_path,"rb") as file:
        angle_dict=pickle.load(file)
    random_angles=angle_dict["angles"]
    random_position=angle_dict["positions"]
    lengths=angle_dict["length"]
    kd_tree_kineamtics=KDTree(random_position[:,-1])


import numpy as np
from shapely import LineString,STRtree,Polygon
from matplotlib import pyplot as plt


angle_limits = np.array([
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2]
])

def forward_kinematics(angles, length=lengths):
    cumulative_angles = np.cumsum(angles)
    x = np.sum(length * np.cos(cumulative_angles))
    y = np.sum(length * np.sin(cumulative_angles))
    return np.array([x, y])

def jacobian(angles, length=lengths):
    dof = len(angles)
    J = np.zeros((2, dof))
    cumulative_angles = np.cumsum(angles)
    sin_angles = np.sin(cumulative_angles)
    cos_angles = np.cos(cumulative_angles)
    for i in range(dof):
        J[0, i] = -np.sum(length[i:] * sin_angles[i:])
        J[1, i] = np.sum(length[i:] * cos_angles[i:])
    return J

def inverse_kinematics(target, initial_angles=None, length=lengths,
                       angle_limits=angle_limits, lr=0.05, max_iters=1000, tol=1e-4):
    if initial_angles is None:
        angles = np.zeros(len(length))
    else:
        angles = initial_angles.copy()

    for _ in range(max_iters):
        pos = forward_kinematics(angles, length)
        error = target - pos
        if np.linalg.norm(error) < tol:
             return angles,error
        J = jacobian(angles, length)
        J_pinv = np.linalg.pinv(J)
        angles += lr * (J_pinv @ error)
        angles = np.clip(angles, angle_limits[:,0], angle_limits[:,1])
    return None,None



def forward_kinematics_batch(angles, length=lengths):
    n = angles.shape[0]
    agent_locations = np.zeros((n, 4, 2), dtype=np.float32)
    cumulative_angles = np.cumsum(angles, axis=1)
    cos_angles = np.cos(cumulative_angles)
    sin_angles = np.sin(cumulative_angles)
    dx = length * cos_angles
    dy = length * sin_angles
    agent_locations[:, 1, 0] = dx[:, 0]
    agent_locations[:, 1, 1] = dy[:, 0]
    agent_locations[:, 2, 0] = dx[:, 0] + dx[:, 1]
    agent_locations[:, 2, 1] = dy[:, 0] + dy[:, 1]
    agent_locations[:, 3, 0] = dx[:, 0] + dx[:, 1] + dx[:, 2]
    agent_locations[:, 3, 1] = dy[:, 0] + dy[:, 1] + dy[:, 2]
    return agent_locations



def check_movement_possible(start,end,length,obstacle_tree,n=10):
    fks=forward_kinematics_batch(np.linspace(start,end,n),length)
    robot_line=np.array([LineString(i) for i in fks])
    query,_ = obstacle_tree.query(robot_line,"intersects")
    if len(query)>0:
        return False
    return True



def check_direct_movemnt_possible(target,initial_angles,robot_link_lengths,obstcales,n=1000):
    target=np.array(target)+1e-6
    angles_sol, final_pos = inverse_kinematics(target, initial_angles)
    if angles_sol is None:
        print("failed to improve path")
        return False,None
    if check_movement_possible(initial_angles,angles_sol,robot_link_lengths,obstcales,n):
        return True,angles_sol
    return False,None


def most_similar(potential_angles, start_node):
    potential_angles = np.asarray(potential_angles)
    start_node = np.asarray(start_node)

    # Relative angles for all candidates
    candidate_del1 = potential_angles[:, 1] - potential_angles[:, 0]
    candidate_del2 = potential_angles[:, 2] - potential_angles[:, 1]

    # Relative angles for start configuration
    start_del1 = start_node[1] - start_node[0]
    start_del2 = start_node[2] - start_node[1]

    # Difference in relative-link angles
    error1 = np.abs(candidate_del1 - start_del1)
    error2 = np.abs(candidate_del2 - start_del2)

    total_error = error1 + error2

    best_index = np.argmin(total_error)

    return potential_angles[best_index]

from tqdm import tqdm
import random

class RRTAngleBatch:

    def __init__(
        self,
        start_pos,
        obstacles,
        tolerance=0.075,
        step_size=0.2,
        batch_size=10,
        limits=angle_limits,
        obstcale_buffer=0.075
    ):
        # ---------------------------------------------------------
        # Basic parameters
        # ---------------------------------------------------------
        self.limits = np.asarray(limits, dtype=np.float32)
        self.length = lengths
        self.workspace_size = sum(self.length)

        self.start_node = self._as_config(start_pos)

        self.tolerance = float(tolerance)
        self.step_size = float(step_size)
        self.batch_size = int(batch_size)
        self.discriminant = float(tolerance)

        self.goal_node = None
        self.error = None
        self.pos = None

        # ---------------------------------------------------------
        # Obstacles
        # ---------------------------------------------------------
        self.obstcale_buffer = float(obstcale_buffer)

        self._raw_obstacles = list(obstacles)

        # Buffered versions only for visualization
        self.obstacles = [
            obstacle.buffer(self.obstcale_buffer)
            for obstacle in self._raw_obstacles
        ]

        self.obstacle_tree = (
            STRtree(self._raw_obstacles)
            if len(self._raw_obstacles) > 0
            else None
        )

        # ---------------------------------------------------------
        # RRT tree
        # ---------------------------------------------------------
        start_tuple = self._tree_tuple(self.start_node)

        self.tree = {
            start_tuple: None
        }

        self.tree_nodes = [
            self.start_node.copy()
        ]

        self.kd_tree = cKDTree(
            np.asarray(self.tree_nodes)
        )

        # =========================================================
        # CACHES
        # =========================================================

        # Per configuration:
        #
        # key -> {
        #     "q": configuration,
        #     "fk": forward kinematics,
        #     "ee": end-effector position,
        #     "line": LineString,
        #     "collision": bool
        # }
        self.config_cache = {}

        # Per workspace goal:
        #
        # goal -> {
        #     potential_angles,
        #     most_similar_angle,
        #     goal_config_tree
        # }
        self.goal_cache = {}

        # Optional continuous-edge collision cache
        self.edge_cache = {}

        # IK results
        self.ik_cache = {}

        # Keep compatibility with your old code
        self.tree_member_properties = {}

        # ---------------------------------------------------------
        # Compute start properties ONCE
        # ---------------------------------------------------------
        records = self._ensure_properties(
            self.start_node[None, :],
            need_fk=True,
            need_line=True,
            need_collision=True
        )

        rec = records[0]

        self.tree_member_properties[start_tuple] = {
            "fk": rec["fk"],
            "line": rec["line"]
        }

    # =============================================================
    # CONFIGURATION REPRESENTATION
    # =============================================================

    @staticmethod
    def _as_config(q):
        """
        Every configuration inside the planner becomes float32.

        This is important because it gives us a stable cache key.
        """
        return np.ascontiguousarray(
            q,
            dtype=np.float32
        ).reshape(-1)

    @staticmethod
    def _as_batch(q):
        q = np.asarray(
            q,
            dtype=np.float32
        )

        if q.ndim == 1:
            q = q[None, :]

        return np.ascontiguousarray(q)

    @staticmethod
    def _config_key(q):
        """
        Exact binary representation of the float32 configuration.

        Same configuration -> same key.
        """
        return np.ascontiguousarray(
            q,
            dtype=np.float32
        ).reshape(-1).tobytes()

    @staticmethod
    def _tree_tuple(q):
        return tuple(
            np.asarray(
                q,
                dtype=np.float32
            ).reshape(-1).tolist()
        )

    # =============================================================
    # CONFIGURATION CACHE
    # =============================================================

    def _get_record(self, q):

        q = self._as_config(q)

        key = self._config_key(q)

        rec = self.config_cache.get(key)

        if rec is None:

            rec = {
                "q": q.copy(),
                "fk": None,
                "ee": None,
                "line": None,
                "collision": None
            }

            self.config_cache[key] = rec

        return key, rec

    # =============================================================
    # MAIN CACHE ENGINE
    # =============================================================

    def _ensure_properties(
        self,
        configs,
        need_fk=False,
        need_line=False,
        need_collision=False
    ):
        """
        Core function.

        Expensive calculations happen ONLY when the result
        does not already exist in the cache.

        FK:
            batched only for configs whose FK is missing.

        LineString:
            created only if missing.

        Collision:
            checked only if missing.
        """

        configs = self._as_batch(configs)

        # ---------------------------------------------------------
        # Deduplicate configs in this request
        # ---------------------------------------------------------
        records_by_key = {}

        for q in configs:

            key, rec = self._get_record(q)

            records_by_key[key] = rec

        # ---------------------------------------------------------
        # FK
        # ---------------------------------------------------------

        require_fk = (
            need_fk
            or need_line
            or need_collision
        )

        if require_fk:

            missing_fk = [
                rec
                for rec in records_by_key.values()
                if rec["fk"] is None
            ]

            if missing_fk:

                missing_configs = np.stack(
                    [
                        rec["q"]
                        for rec in missing_fk
                    ],
                    axis=0
                )

                # *************************************
                # FK CALCULATED ONLY FOR CACHE MISSES
                # *************************************
                fks = forward_kinematics_batch(
                    missing_configs,
                    length=self.length
                )

                for rec, fk in zip(
                    missing_fk,
                    fks
                ):
                    rec["fk"] = fk

                    rec["ee"] = np.asarray(
                        fk[-1],
                        dtype=np.float32
                    )

        # ---------------------------------------------------------
        # LineStrings
        # ---------------------------------------------------------

        if need_line or need_collision:

            for rec in records_by_key.values():

                if rec["line"] is None:

                    # *************************************
                    # LineString CREATED ONLY ONCE
                    # *************************************
                    rec["line"] = LineString(
                        rec["fk"]
                    )

        # ---------------------------------------------------------
        # Collision
        # ---------------------------------------------------------

        if need_collision:

            missing_collision = [
                rec
                for rec in records_by_key.values()
                if rec["collision"] is None
            ]

            if missing_collision:

                # No obstacles
                if self.obstacle_tree is None:

                    for rec in missing_collision:
                        rec["collision"] = False

                else:

                    lines = np.asarray(
                        [
                            rec["line"]
                            for rec in missing_collision
                        ],
                        dtype=object
                    )

                    collisions = np.zeros(
                        len(lines),
                        dtype=bool
                    )

                    # *************************************
                    # Collision checked ONLY for misses
                    # *************************************
                    result = self.obstacle_tree.query(
                        lines,
                        predicate="dwithin",
                        distance=self.obstcale_buffer
                    )

                    result = np.asarray(result)

                    if result.size > 0:

                        input_indices = result[0]

                        collisions[
                            np.unique(input_indices)
                        ] = True

                    for rec, collision in zip(
                        missing_collision,
                        collisions
                    ):
                        rec["collision"] = bool(
                            collision
                        )

        # Return in the same order as input
        return [
            self.config_cache[
                self._config_key(q)
            ]
            for q in configs
        ]

    # =============================================================
    # OLD FUNCTION COMPATIBILITY
    # =============================================================

    def add_tree_member_properties(
        self,
        member,
        fk=None,
        line=None
    ):

        q = self._as_config(member)

        key, rec = self._get_record(q)

        if fk is not None:

            if rec["fk"] is None:

                rec["fk"] = fk

                rec["ee"] = np.asarray(
                    fk[-1],
                    dtype=np.float32
                )

        if line is not None:

            if rec["line"] is None:
                rec["line"] = line

        self._ensure_properties(
            q,
            need_fk=True,
            need_line=True
        )

        rec = self.config_cache[key]

        self.tree_member_properties[
            self._tree_tuple(q)
        ] = {
            "fk": rec["fk"],
            "line": rec["line"]
        }

    # =============================================================
    # FK
    # =============================================================

    def forward_kinematics(self, angles):

        q = self._as_config(angles)

        rec = self._ensure_properties(
            q,
            need_fk=True
        )[0]

        # preserve your old output shape
        return rec["fk"].T

    # =============================================================
    # WORKSPACE DISTANCE
    # =============================================================

    def distance_in_workspace_batch(
        self,
        angles_batch,
        goal_ws
    ):

        configs = self._as_batch(
            angles_batch
        )

        goal_ws = np.asarray(
            goal_ws,
            dtype=np.float32
        )

        records = self._ensure_properties(
            configs,
            need_fk=True
        )

        ee_positions = np.stack(
            [
                rec["ee"]
                for rec in records
            ]
        )

        return np.linalg.norm(
            ee_positions - goal_ws,
            axis=1
        )

    # =============================================================
    # COLLISION CHECK
    # =============================================================

    def is_in_obstacle_batch(
        self,
        angles_batch
    ):

        configs = self._as_batch(
            angles_batch
        )

        records = self._ensure_properties(
            configs,
            need_fk=True,
            need_line=True,
            need_collision=True
        )

        return np.asarray(
            [
                rec["collision"]
                for rec in records
            ],
            dtype=bool
        )

    # =============================================================
    # GOAL CACHE
    # =============================================================

    def _goal_key(self, goal_ws):

        goal_ws = np.ascontiguousarray(
            goal_ws,
            dtype=np.float32
        )

        return (
            goal_ws.tobytes(),
            float(self.discriminant)
        )

    def _prepare_goal(self, goal_ws):

        goal_ws = np.asarray(
            goal_ws,
            dtype=np.float32
        )

        key = self._goal_key(goal_ws)

        # ---------------------------------------------------------
        # Goal already prepared
        # ---------------------------------------------------------

        if key in self.goal_cache:
            return self.goal_cache[key]

        # ---------------------------------------------------------
        # Find configurations whose end-effector is near goal
        # ---------------------------------------------------------

        indices = kd_tree_kineamtics.query_ball_point(
            goal_ws,
            self.discriminant
        )

        potential_angles = np.asarray(
            random_angles[indices],
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # No possible configurations
        # ---------------------------------------------------------

        if len(potential_angles) == 0:

            result = {
                "potential_angles": potential_angles,
                "most_similar_angle": None,
                "goal_config_tree": None
            }

            self.goal_cache[key] = result

            return result

        # =========================================================
        # IMPORTANT:
        # REMOVE COLLIDING GOAL CONFIGURATIONS
        # =========================================================

        collisions = self.is_in_obstacle_batch(
            potential_angles
        )

        free_mask = ~collisions

        potential_angles = potential_angles[
            free_mask
        ]

        # ---------------------------------------------------------
        # All possible IK/goal configurations are in collision
        # ---------------------------------------------------------

        if len(potential_angles) == 0:

            result = {
                "potential_angles": potential_angles,
                "most_similar_angle": None,
                "goal_config_tree": None
            }

            self.goal_cache[key] = result

            print(
                "All potential goal configurations are in collision"
            )

            return result

        # =========================================================
        # NOW choose preferred angle ONLY among collision-free ones
        # =========================================================

        preferred = most_similar(
            potential_angles,
            self.start_node
        )

        preferred = self._as_config(
            preferred
        )

        # ---------------------------------------------------------
        # Safety assertion
        #
        # This should already be False because potential_angles
        # was filtered above.
        # ---------------------------------------------------------

        if self.is_in_obstacle_batch(
            preferred
        )[0]:

            raise RuntimeError(
                "Preferred goal configuration is unexpectedly "
                "in collision after filtering."
            )

        # =========================================================
        # KD-tree contains ONLY collision-free goal configurations
        # =========================================================

        goal_config_tree = cKDTree(
            potential_angles
        )

        result = {
            "potential_angles": potential_angles,
            "most_similar_angle": preferred,
            "goal_config_tree": goal_config_tree
        }

        self.goal_cache[key] = result

        return result
    # =============================================================
    # SAMPLING
    # =============================================================

    def sample_positions_batch(
        self,
        n,
        goal_ws=None,
        goal_bias=0.15
    ):
        """
        IMPORTANT:

        No FK here.
        No LineString here.
        No collision here.

        Samples are just targets for steering.
        """

        n = int(n)

        samples = np.empty(
            (
                n,
                len(self.start_node)
            ),
            dtype=np.float32
        )

        goal_mask = (
            np.random.random(n)
            < goal_bias
        )

        random_indices = np.flatnonzero(
            ~goal_mask
        )

        goal_indices = np.flatnonzero(
            goal_mask
        )

        # ---------------------------------------------------------
        # Random configuration sampling
        # ---------------------------------------------------------

        if len(random_indices) > 0:

            lows = self.limits[:, 0]
            highs = self.limits[:, 1]

            samples[random_indices] = (
                np.random.uniform(
                    lows,
                    highs,
                    size=(
                        len(random_indices),
                        len(lows)
                    )
                )
            ).astype(np.float32)

        # ---------------------------------------------------------
        # Goal biased samples
        # ---------------------------------------------------------

        if len(goal_indices) > 0:

            # First goal sample = most similar
            samples[
                goal_indices[0]
            ] = self.most_similar_angle

            # Others randomly selected from possible goal configs
            if len(goal_indices) > 1:

                selected = np.random.randint(
                    0,
                    len(self.potential_angles),
                    size=len(goal_indices) - 1
                )

                samples[
                    goal_indices[1:]
                ] = self.potential_angles[
                    selected
                ]

        return samples

    # =============================================================
    # NEAREST NODE
    # =============================================================

    def nearest_nodes_batch(
        self,
        samples
    ):

        _, idx = self.kd_tree.query(
            samples,
            k=1
        )

        return np.asarray(
            self.kd_tree.data[idx],
            dtype=np.float32
        )

    # =============================================================
    # STEERING
    # =============================================================

    def steer_batch(
        self,
        from_nodes,
        to_nodes
    ):

        from_nodes = self._as_batch(
            from_nodes
        )

        to_nodes = self._as_batch(
            to_nodes
        )

        directions = (
            to_nodes
            - from_nodes
        )

        distances = np.linalg.norm(
            directions,
            axis=1,
            keepdims=True
        )

        new_nodes = (
            from_nodes
            +
            (
                directions
                /
                (distances + 1e-10)
            )
            * self.step_size
        )

        return new_nodes.astype(
            np.float32
        )

    # =============================================================
    # DUPLICATE FILTER
    # =============================================================

    def _filter_duplicates(
        self,
        new_nodes,
        parent_nodes
    ):
        """
        Remove duplicates BEFORE doing FK or collision.
        """

        keep = []

        seen = set()

        for i, node in enumerate(
            new_nodes
        ):

            key = self._tree_tuple(
                node
            )

            # Already exists in RRT
            if key in self.tree:
                continue

            # Duplicate inside current batch
            if key in seen:
                continue

            seen.add(key)

            keep.append(i)

        if len(keep) == 0:

            dof = new_nodes.shape[1]

            return (
                np.empty(
                    (0, dof),
                    dtype=np.float32
                ),
                np.empty(
                    (0, dof),
                    dtype=np.float32
                )
            )

        keep = np.asarray(
            keep,
            dtype=np.int64
        )

        return (
            new_nodes[keep],
            parent_nodes[keep]
        )

    # =============================================================
    # OPTIONAL EDGE CACHE
    # =============================================================

    def check_edge_cached(
        self,
        start,
        end,
        check_function
    ):
        """
        If you later enable check_movement_possible(),
        use this wrapper.

        The exact same edge will never be checked twice.
        """

        start = self._as_config(
            start
        )

        end = self._as_config(
            end
        )

        edge_key = (
            self._config_key(start),
            self._config_key(end)
        )

        cached = self.edge_cache.get(
            edge_key
        )

        if cached is not None:
            return cached

        valid = bool(
            check_function(
                start,
                end
            )
        )

        self.edge_cache[
            edge_key
        ] = valid

        return valid

    # =============================================================
    # BUILD TREE
    # =============================================================

    def build_tree(
        self,
        goal_ws,
        max_iterations=10_000,
        goal_bias=0.2
    ):

        self.error = None
        self.goal_node = None

        goal_ws = np.asarray(
            goal_ws,
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Prepare goal once
        # ---------------------------------------------------------

        goal_data = self._prepare_goal(
            goal_ws
        )

        self.potential_angles = (
            goal_data[
                "potential_angles"
            ]
        )

        self.most_similar_angle = (
            goal_data[
                "most_similar_angle"
            ]
        )

        goal_config_tree = (
            goal_data[
                "goal_config_tree"
            ]
        )

        if len(
            self.potential_angles
        ) == 0:

            print(
                "solution not found"
            )

            return None

        # ---------------------------------------------------------
        # Start collision
        #
        # This returns cached result from __init__.
        # ZERO repeated FK/collision calculation.
        # ---------------------------------------------------------

        if self.is_in_obstacle_batch(
            self.start_node
        )[0]:

            print(
                "start config in collision"
            )

            return None

        # =========================================================
        # MAIN RRT LOOP
        # =========================================================

        for iteration in tqdm(
            range(max_iterations)
        ):

            # -----------------------------------------------------
            # 1. SAMPLE
            # -----------------------------------------------------

            samples = (
                self.sample_positions_batch(
                    self.batch_size,
                    goal_ws,
                    goal_bias
                )
            )

            # -----------------------------------------------------
            # 2. NEAREST NODE
            # -----------------------------------------------------

            nearest_nodes = (
                self.nearest_nodes_batch(
                    samples
                )
            )

            # -----------------------------------------------------
            # 3. STEER
            # -----------------------------------------------------

            new_nodes = (
                self.steer_batch(
                    nearest_nodes,
                    samples
                )
            )

            # -----------------------------------------------------
            # 4. REMOVE DUPLICATES FIRST
            #
            # Do this before FK/collision.
            # -----------------------------------------------------

            (
                new_nodes,
                nearest_nodes
            ) = self._filter_duplicates(
                new_nodes,
                nearest_nodes
            )

            if len(new_nodes) == 0:
                continue

            # -----------------------------------------------------
            # 5. COLLISION CHECK
            #
            # FK + LineString + collision computed together and
            # cached.
            # -----------------------------------------------------

            collisions = (
                self.is_in_obstacle_batch(
                    new_nodes
                )
            )

            free_mask = ~collisions

            if not np.any(
                free_mask
            ):
                continue

            new_nodes_final = (
                new_nodes[
                    free_mask
                ]
            )

            nearest_nodes_final = (
                nearest_nodes[
                    free_mask
                ]
            )

            # -----------------------------------------------------
            # OPTIONAL CONTINUOUS MOTION CHECK
            # -----------------------------------------------------
            #
            # If you later want:
            #
            # valid_children = []
            # valid_parents = []
            #
            # for parent, child in zip(
            #     nearest_nodes_final,
            #     new_nodes_final
            # ):
            #
            #     valid = self.check_edge_cached(
            #         parent,
            #         child,
            #         lambda a, b:
            #         check_movement_possible(
            #             a,
            #             b,
            #             self.length,
            #             self.obstacle_tree
            #         )
            #     )
            #
            #     if valid:
            #         valid_parents.append(parent)
            #         valid_children.append(child)
            #
            # -----------------------------------------------------

            # -----------------------------------------------------
            # 6. ADD NODES
            # -----------------------------------------------------

            added_nodes = []

            for parent, child in zip(
                nearest_nodes_final,
                new_nodes_final
            ):

                child_tuple = (
                    self._tree_tuple(
                        child
                    )
                )

                # safety check
                if child_tuple in self.tree:
                    continue

                parent_tuple = (
                    self._tree_tuple(
                        parent
                    )
                )

                self.tree[
                    child_tuple
                ] = parent_tuple

                self.tree_nodes.append(
                    child.copy()
                )

                added_nodes.append(
                    child
                )

                # Geometry already exists because collision
                # checking required it.
                rec = self.config_cache[
                    self._config_key(
                        child
                    )
                ]

                self.tree_member_properties[
                    child_tuple
                ] = {
                    "fk": rec["fk"],
                    "line": rec["line"]
                }

            if len(added_nodes) == 0:
                continue

            added_nodes = np.asarray(
                added_nodes,
                dtype=np.float32
            )

            # -----------------------------------------------------
            # 7. UPDATE TREE KD-TREE
            #
            # Only rebuild when nodes were actually added.
            # -----------------------------------------------------

            self.kd_tree = cKDTree(
                np.asarray(
                    self.tree_nodes,
                    dtype=np.float32
                )
            )

            # -----------------------------------------------------
            # 8. CHECK DISTANCE TO GOAL CONFIGURATIONS
            #
            # OLD:
            #
            # distances =
            # new_nodes[:,None,:] - potential_angles[None,:,:]
            #
            # NEW:
            #
            # KD-tree query.
            #
            # No huge NxM matrix.
            # -----------------------------------------------------

            (
                min_distances,
                nearest_goal_indices
            ) = goal_config_tree.query(
                added_nodes,
                k=1
            )

            goal_reached_indices = (
                np.flatnonzero(
                    min_distances
                    <
                    self.tolerance * 2
                )
            )

            if len(
                goal_reached_indices
            ) > 0:

                best_idx = (
                    goal_reached_indices[
                        np.argmin(
                            min_distances[
                                goal_reached_indices
                            ]
                        )
                    ]
                )

                self.error = float(
                    min_distances[
                        best_idx
                    ]
                )

                self.goal_node = (
                    self._tree_tuple(
                        added_nodes[
                            best_idx
                        ]
                    )
                )

                return self.reconstruct_path(
                    goal_ws
                )

        return None

    # =============================================================
    # PATH
    # =============================================================
    def reconstruct_path(self, goal_ws=None):

        path = []
        node = self.goal_node

        while node is not None:
            path.append(node)
            node = self.tree[node]

        # Valid path already found by RRT
        reversed_path = path[::-1]

        # ---------------------------------------------------------
        # No IK refinement requested
        # ---------------------------------------------------------
        if goal_ws is None or len(reversed_path) == 0:
            return reversed_path

        goal_ws = np.asarray(
            goal_ws,
            dtype=np.float32
        )

        # Last valid RRT configuration
        seed = self._as_config(
            reversed_path[-1]
        )

        ik_key = (
            goal_ws.tobytes(),
            self._config_key(seed)
        )

        cached = self.ik_cache.get(
            ik_key
        )

        # =========================================================
        # CALCULATE IK
        # =========================================================

        if cached is None:

            ik_angle, pos = inverse_kinematics(
                goal_ws,
                seed,
                self.length,
                self.limits
            )

            self.ik_cache[ik_key] = (
                ik_angle,
                pos
            )

        else:

            ik_angle, pos = cached

        self.pos = pos

        # =========================================================
        # IK FAILED
        #
        # Do NOT discard RRT path.
        # Just return the path already found.
        # =========================================================

        if ik_angle is None:

            print(
                "IK improvement failed - "
                "returning original RRT path"
            )

            return reversed_path

        ik_angle = self._as_config(
            ik_angle
        )

        # =========================================================
        # CHECK FINAL IK CONFIGURATION
        # =========================================================

        ik_collision = self.is_in_obstacle_batch(
            ik_angle
        )[0]

        if ik_collision:

            print(
                "IK configuration is in collision - "
                "returning original RRT path"
            )

            return reversed_path

        # =========================================================
        # CHECK MOTION:
        # LAST RRT NODE -> IK CONFIGURATION
        # =========================================================

        valid_motion = self.check_edge_cached(
            seed,
            ik_angle,
            lambda a, b:
            check_movement_possible(
                a,
                b,
                self.length,
                self.obstacle_tree
            )
        )

        if not valid_motion:

            print(
                "IK connection is in collision - "
                "returning original RRT path"
            )

            return reversed_path

        # =========================================================
        # IK IMPROVEMENT IS VALID
        # ONLY NOW APPEND IT
        # =========================================================

        reversed_path.append(
            self._tree_tuple(
                ik_angle
            )
        )

        self._ensure_properties(
            ik_angle,
            need_fk=True,
            need_line=True,
            need_collision=True
        )

        return reversed_path
    # =============================================================
    # RENDER
    # =============================================================

    def render(
        self,
        path=None,
        target=None
    ):

        fig, ax = plt.subplots(
            figsize=(4, 4)
        )

        # ---------------------------------------------------------
        # Obstacles
        # ---------------------------------------------------------

        for obstacle in self.obstacles:

            ax.plot(
                *obstacle.boundary.xy
            )

        # ---------------------------------------------------------
        # Tree
        #
        # FK comes from cache.
        # ---------------------------------------------------------

        for node, parent in self.tree.items():

            if parent is None:
                continue

            node_q = self._as_config(
                node
            )

            parent_q = self._as_config(
                parent
            )

            records = self._ensure_properties(
                np.stack(
                    [
                        parent_q,
                        node_q
                    ]
                ),
                need_fk=True
            )

            parent_ee = records[0]["ee"]
            child_ee = records[1]["ee"]

            ax.plot(
                [
                    parent_ee[0],
                    child_ee[0]
                ],
                [
                    parent_ee[1],
                    child_ee[1]
                ],
                color="lightblue",
                linewidth=0.3,
                alpha=0.5
            )

        # ---------------------------------------------------------
        # Path
        #
        # Again FK comes from cache.
        # ---------------------------------------------------------

        if path is not None:

            path = self._as_batch(
                path
            )

            records = self._ensure_properties(
                path,
                need_fk=True
            )

            for count, rec in enumerate(
                records,
                start=1
            ):

                fk = rec["fk"]

                ax.plot(
                    fk[:, 0],
                    fk[:, 1]
                )

                ax.text(
                    fk[-1, 0],
                    fk[-1, 1],
                    str(count)
                )

        if target is not None:

            ax.scatter(
                target[0],
                target[1],
                marker="x"
            )

        ax.set_xlim(
            -0.9,
            0.9
        )

        ax.set_ylim(
            -0.9,
            0.9
        )

        ax.set_aspect(
            "equal"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.set_title(
            f"RRT Path Planning "
            f"(Tree size: {len(self.tree)} nodes)"
        )

        return fig
    
import numpy as np
from shapely.geometry import LineString

def smooth_path(path, rrt, robot_radius=0.05, num_checks=100):
    """
    Smooth a joint-space RRT path by removing unnecessary waypoints.

    path: list/array of joint angle configurations
          Example: [q0, q1, q2, ...]
    rrt: object containing:
         - rrt.obstacle_tree
         - rrt.length
    """

    if path is None or len(path) == 0:
        return []

    if len(path) <= 2:
        return path

    obstacles_tree = rrt.obstacle_tree
    path = [np.asarray(p, dtype=float) for p in path]

    new_path = [path[0]]
    i = 0
    n = len(path)

    def collision_free(q_start, q_end):
        # Interpolate in joint space
        configs = np.linspace(q_start, q_end, num_checks)

        # FK gives robot joint locations for each interpolated config
        robot_points = forward_kinematics_batch(configs, rrt.length)

        for pts in robot_points:
            robot_shape = LineString(pts).buffer(robot_radius)

            hits = obstacles_tree.query(robot_shape, predicate="intersects")
            print(hits)
            if len(hits) > 0:
                return False

        return True

    while i < n - 1:
        # Try to jump as far forward as possible
        best_j = i + 1

        for j in range(i + 2, n):
            if collision_free(path[i], path[j]):
                best_j = j
            else:
                break

        new_path.append(path[best_j])
        i = best_j

    return new_path

import numpy as np
from shapely import LineString,Point
from sensor_msgs.msg import JointState
from matplotlib import pyplot as plt
from geometry_msgs.msg import PoseStamped
import numpy as np

def forward_kinematics_batch(angles, length=np.array([0.325, 0.275, 0.25])):
    n = angles.shape[0]  
    agent_locations = np.zeros((n, 4, 2), dtype=np.float32)
    cumulative_angles = np.cumsum(angles, axis=1)
    cos_angles = np.cos(cumulative_angles)
    sin_angles = np.sin(cumulative_angles)
    dx = length * cos_angles
    dy = length * sin_angles
    agent_locations[:, 1, 0] = dx[:, 0]
    agent_locations[:, 1, 1] = dy[:, 0]
    agent_locations[:, 2, 0] = dx[:, 0] + dx[:, 1]
    agent_locations[:, 2, 1] = dy[:, 0] + dy[:, 1]
    agent_locations[:, 3, 0] = dx[:, 0] + dx[:, 1] + dx[:, 2]
    agent_locations[:, 3, 1] = dy[:, 0] + dy[:, 1] + dy[:, 2]
    return agent_locations

from custom_interfaces.srv import RrtRequest
import numpy as np
from std_msgs.msg import String as RosString


class RRT(Node):
    def __init__(self):
        Node.__init__(self,"read_obs")
        self.create_subscription(PointCloud2,"/obstcale_points",self.update_obs_points,10)
        self.obstacles=None

        self.create_subscription(JointState,"/joint_states",self.update_joint_poistion,10)
        self.poisition=None
        self.joint_publisher=self.create_publisher(JointState,"/joint_states_change_request",10)
        self.create_subscription(PoseStamped,"/goal_pose",self._find_path,10)
        # self.status_publisher_=self.create_publisher()
        self.timmer=self.create_timer(0.025,self.timer_callback)
        self.create_service(RrtRequest,"/rrt_service/solver/rrt_solver",self.solve_rrt)
        self.s_path=None
        self.count=None
        self.wait_count=None
        self.status_publisher=self.create_publisher(RosString,"/path_plan_status",10)
        self.status=""
        # self.retry=0
    def solve_rrt(self,input,output):
        obstacles_raw=np.array(input.obstacles).reshape(-1,2,2)
        obstacles=[LineString(points) for points in obstacles_raw]
        target=input.target
        angles=input.angles
        self.get_logger().info(f"Searching path x: {round(target[0],4)}, y: {round(target[1],4)}")
        r=RRTAngleBatch(angles, obstacles)
        output.angles=[]
        output.sucess=False

        path=r.build_tree(target)
        
        # if path:
        #     self.get_logger().info(f"final error {np.linalg.norm(np.array(target)-np.array(r.pos))}")
        #     output.sucess=True
        #     s_path=smooth_path(path,r)
        #     print(path,s_path)
        #     output.angles=np.array(s_path).flatten().tolist()

        self.get_logger().info(f"sizre of tree {len(r.tree_nodes)}")
        if path is None:
            self.get_logger().warn("Path planning failed: no path found")

        else:
            output.angles=np.array(path).flatten().tolist()
            output.sucess=True
        return output


    def timer_callback(self):
        msg=RosString()
        msg.data=self.status
        self.status_publisher.publish(msg)
        if self.status=="standby":
            return
        
        if self.s_path is None:
            self.status="standby"
            return
        
        if self.count>=len(self.s_path):
            self.status="sucess"
            self.get_logger().info("movement complete")
            for i in range(-0,10):
                msg=RosString()
                msg.data=self.status
                self.status_publisher.publish(msg)
                time.sleep(0.1)
            self.status="standby"
            self.count=None
            self.s_path=None
            self.wait_count=None
            return
        
        if self.wait_count>500: 
            self.status="failure"

            for i in range(-0,10):
                msg=RosString()
                msg.data=self.status
                self.status_publisher.publish(msg)
                time.sleep(0.1)

            self.status="standby"
            self.count=None
            self.s_path=None
            self.wait_count=None
        else:
            self.wait_count+=1
        self.status="running"
        self.get_logger().info(f"running: {self.count}/{len(self.s_path)}")
        angle1,angle2,angle3=self.s_path[self.count]
        if (abs(self.poisition['Angle1']-angle1)<0.02 and
                             abs(self.poisition['Angle2']-angle2)<0.02 and
                              abs(self.poisition['Angle3']-angle3)<0.02):
            self.count+=1
        
        msg=JointState()
        msg.name=list(self.poisition.keys())
        msg.position=list(self.poisition.values())
        msg.position[1]=angle1
        msg.position[2]=angle2
        msg.position[3]=angle3
        self.joint_publisher.publish(msg)



    def update_joint_poistion(self,msg):
        # self.get_logger().info("angles updated")
        self.poisition={}
        for i,j in zip(msg.name,msg.position):
            self.poisition[i]=j
        self.angles=[self.poisition[f"Angle{i+1}"] for i in range(0,3)]


    def update_obs_points(self,msg):
        
        points = point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)

        points=np.asarray([[x,y] for x,y,z in points])
        self.points=points.copy()
        pos=forward_kinematics_batch(np.array([[self.poisition['Angle1'],self.poisition['Angle2'],self.poisition['Angle3']]]))[0]

        off_point=self.points-pos[-2]
        off_robot=pos-pos[-2]
        norms = np.linalg.norm(off_point, axis=1, keepdims=True)
        end_points1 = off_point*1.25 / norms
        end_points2 = off_point*1.1
        self.obstacles=[]

        # robot_line=LineString(pos).buffer(0.04)

        for i,j,k in zip(off_point,end_points1,end_points2):
            
            y=[j,k][np.argmin(np.linalg.norm([j,k],axis=1))]
            # obstacles.append(LineString([i+pos[-2],y+pos[-2]]).buffer(0.02))
            x=i+pos[-2]
            y=y+pos[-2]
            obs=LineString([x,y]).buffer(0.05)
            self.obstacles.append(obs)
        

    def _find_path(self,msg):
        self.status="running"
        smsg=RosString()
        smsg.data=self.status
        self.status_publisher.publish(smsg)
        if self.s_path is None:
            self.path,self.s_path,self.r=self.goto_target([msg.pose.position.x,msg.pose.position.y])
            self.count=0
            self.wait_count=0
            # self.retry=0
        else:
            self.status="failure"
            self.get_logger().warn("wait for task to complete")
        
    def move_angles(self,angle1,angle2,angle3):
        # moe to poistion
        msg=JointState()
        msg.name=list(self.poisition.keys())
        msg.position=list(self.poisition.values())
        msg.position[1]=angle1
        msg.position[2]=angle2
        msg.position[3]=angle3
        self.joint_publisher.publish(msg)
        
    def _get_path(self,target):
        self.get_logger().info(f"Searching path x: {round(target[0],4)}, y: {round(target[1],4)}")

        if self.angles is None:
            self.get_logger().warn("RRT inital angles not updated")
            self.status="failure"
            return None,None,None
        if self.obstacles is None:
            self.obstacles=[]
            self.get_logger().warn("obstcales not updated")
       
        r=RRTAngleBatch(self.angles, self.obstacles)
        if len(r.obstacle_tree.query(np.array(Point(target)),predicate="intersects"))>0:
            self.get_logger().warn("target intersect")
            self.status="failure"
            # self.get_logger().warn("retry")
            
            # if self.retry<3:
            #     self.retry+=1
            #     rclpy.spin_once(self)

            return None,None,r
        
        path=r.build_tree(target)
        if path:
            self.get_logger().info(f"final error {np.linalg.norm(np.array(target)-np.array(r.pos))}")
            
        # if path is None:
        #     path=r.build_tree(target)
        #     if path is None:
        #         return None,None,r
        s_path=smooth_path(path,r)
        if path:
            self.get_logger().info(f"final error {np.linalg.norm(np.array(target)-np.array(r.pos))}")
            
        # if path is None:
        #     path=r.build_tree(target)
        #     if path is None:
        #         return None,None,r
        s_path=smooth_path(path,r)
        if path is not None:
            self.get_logger().info(f"error is {r.error}")
        return path,s_path,r

def main(args=None):
    rclpy.init(args=args)
    rrt=RRT()
    rclpy.spin(rrt)
    rrt.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
