import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.strtree import STRtree
import pickle

# ============================================================
# 1. LOAD FK CACHE
# ============================================================
with open("fk.cahce","rb") as file:
    fk_cache=pickle.load(file)
lengths = np.asarray(fk_cache["lengths"], dtype=float)
limits = np.asarray(fk_cache["limits"], dtype=float)
angles = np.asarray(fk_cache["angles"], dtype=float)
fks = np.asarray(fk_cache["fks"], dtype=float)

# Supports (N, 4, 2) or (N, 2)
ee_positions = fks[:, -1, :] if fks.ndim == 3 else fks

ik_tree = cKDTree(ee_positions)


# ============================================================
# 2. FORWARD KINEMATICS
# ============================================================

def forward_kinematics(q, lengths):
    q = np.asarray(q, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    theta = np.cumsum(q)

    x = np.cumsum(lengths * np.cos(theta))
    y = np.cumsum(lengths * np.sin(theta))

    return np.vstack([
        [0.0, 0.0],
        np.column_stack((x, y))
    ])


# ============================================================
# 3. KD-TREE INVERSE KINEMATICS
# ============================================================

def distance_inverse_kinematics(target, k=100):
    target = np.asarray(target, dtype=float)

    distances, indices = ik_tree.query(
        target,
        k=min(k, len(angles))
    )

    indices = np.atleast_1d(indices)
    distances = np.atleast_1d(distances)

    return angles[indices], distances


# ============================================================
# 4. RRT
# ============================================================

class RRT:

    def __init__(
        self,
        start,
        target,
        lengths,
        limits,
        obstacles=None,
        obstacle_buffer=0.02,
        edge_resolution=0.03,
        step_size=0.15,
        goal_bias=0.2,
        goal_tolerance=0.05,
        ik_tolerance=0.001,
        max_iterations=10000,
        k=100,
        min_ee_radius=0.1,
    ):

        self.start = np.asarray(start, dtype=float)
        self.target = np.asarray(target, dtype=float)

        self.lengths = np.asarray(lengths, dtype=float)
        self.limits = np.asarray(limits, dtype=float)

        self.obstacle_buffer = obstacle_buffer
        self.edge_resolution = edge_resolution
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.ik_tolerance = ik_tolerance
        self.max_iterations = max_iterations
        self.k = k
        self.min_ee_radius = min_ee_radius

        # Create Shapely line obstacles
        self.obstacles = [
            obs if isinstance(obs, LineString) else LineString(obs)
            for obs in (obstacles if obstacles is not None else [])
        ]

        self.obstacle_tree = (
            STRtree(self.obstacles)
            if self.obstacles else None
        )

        self.nodes = [self.start.copy()]
        self.parents = [-1]

        self.goal_solutions = np.empty((0, 3))
        self.best_solution = None
        self.goal_index = None

        self.iterations = 0

    # --------------------------------------------------------
    # Collision checking
    # --------------------------------------------------------

    def in_collision(self, q):

        fk = forward_kinematics(q, self.lengths)
        ee = fk[-1]

        # End effector cannot be within 0.1 m of origin
        if np.linalg.norm(ee) < self.min_ee_radius:
            return True

        if self.obstacle_tree is None:
            return False

        robot = LineString(fk)

        candidates = self.obstacle_tree.query(
            robot.buffer(self.obstacle_buffer)
        )

        for idx in candidates:
            obstacle = self.obstacles[int(idx)]

            if robot.distance(obstacle) <= self.obstacle_buffer:
                return True

        return False

    # --------------------------------------------------------
    # Check motion between two configurations
    # --------------------------------------------------------

    def check_edge(self, q_start, q_end):

        q_start = np.asarray(q_start, dtype=float)
        q_end = np.asarray(q_end, dtype=float)

        distance = np.linalg.norm(q_end - q_start)

        steps = max(
            1,
            int(np.ceil(distance / self.edge_resolution))
        )

        for i in range(steps + 1):

            t = i / steps
            q = q_start + t * (q_end - q_start)

            if self.in_collision(q):
                return False

        return True

    # --------------------------------------------------------
    # Find multiple IK solutions using KD-tree
    # --------------------------------------------------------

    def find_goals(self):

        candidates, distances = distance_inverse_kinematics(
            self.target,
            k=self.k
        )

        candidates = np.asarray(
            candidates, dtype=float
        ).reshape(-1, 3)

        # Filter joint limits
        valid = np.all(
            (candidates >= self.limits[:, 0]) &
            (candidates <= self.limits[:, 1]),
            axis=1
        )

        candidates = candidates[valid]

        # Filter collisions
        candidates = np.array(
            [
                q for q in candidates
                if not self.in_collision(q)
            ],
            dtype=float
        ).reshape(-1, 3)

        if len(candidates) == 0:
            self.goal_solutions = np.empty((0, 3))
            self.best_solution = None
            return self.goal_solutions

        # Sort by joint-space distance from start
        joint_distances = np.linalg.norm(
            candidates - self.start,
            axis=1
        )

        order = np.argsort(joint_distances)

        self.goal_solutions = candidates[order]
        self.best_solution = self.goal_solutions[0].copy()

        return self.goal_solutions

    # --------------------------------------------------------
    # Goal-biased sampling
    # --------------------------------------------------------

    def sample(self):

        if (
            len(self.goal_solutions) > 0
            and np.random.random() < self.goal_bias
        ):

            # 50% nearest IK configuration
            if np.random.random() < 0.5:
                return self.best_solution.copy()

            # 50% random IK candidate
            index = np.random.randint(
                len(self.goal_solutions)
            )

            return self.goal_solutions[index].copy()

        # Uniform random configuration
        return np.random.uniform(
            self.limits[:, 0],
            self.limits[:, 1]
        )

    # --------------------------------------------------------
    # Nearest node
    # --------------------------------------------------------

    def nearest(self, q):

        nodes = np.asarray(self.nodes)

        distances = np.linalg.norm(
            nodes - q,
            axis=1
        )

        return int(np.argmin(distances))

    # --------------------------------------------------------
    # Steer
    # --------------------------------------------------------

    def steer(self, q_near, q_rand):

        direction = q_rand - q_near
        distance = np.linalg.norm(direction)

        if distance < 1e-12:
            return q_near.copy()

        if distance <= self.step_size:
            return q_rand.copy()

        return (
            q_near
            + direction / distance * self.step_size
        )

    # --------------------------------------------------------
    # Check workspace goal
    # --------------------------------------------------------

    def reached_goal(self, q):

        ee = forward_kinematics(
            q, self.lengths
        )[-1]

        return (
            np.linalg.norm(ee - self.target)
            <= self.goal_tolerance
        )

    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    def jacobian(self, q):

        q1, q2, q3 = q
        l1, l2, l3 = self.lengths

        t1 = q1
        t2 = q1 + q2
        t3 = q1 + q2 + q3

        return np.array([
            [
                -l1*np.sin(t1)
                -l2*np.sin(t2)
                -l3*np.sin(t3),

                -l2*np.sin(t2)
                -l3*np.sin(t3),

                -l3*np.sin(t3)
            ],
            [
                l1*np.cos(t1)
                +l2*np.cos(t2)
                +l3*np.cos(t3),

                l2*np.cos(t2)
                +l3*np.cos(t3),

                l3*np.cos(t3)
            ]
        ])

    # --------------------------------------------------------
    # Pseudoinverse IK refinement
    # --------------------------------------------------------

    def refine_ik(
        self,
        q,
        max_iterations=100,
        max_step=0.1,
    ):

        q = np.asarray(q, dtype=float).copy()

        for _ in range(max_iterations):

            ee = forward_kinematics(
                q, self.lengths
            )[-1]

            error = self.target - ee

            if np.linalg.norm(error) <= self.ik_tolerance:
                return q

            J = self.jacobian(q)

            dq = np.linalg.pinv(J) @ error

            norm = np.linalg.norm(dq)

            if norm > max_step:
                dq *= max_step / norm

            q = np.clip(
                q + dq,
                self.limits[:, 0],
                self.limits[:, 1]
            )

        # Check final iterate as well
        ee = forward_kinematics(q, self.lengths)[-1]

        if np.linalg.norm(ee - self.target) <= self.ik_tolerance:
            return q

        return None

    # --------------------------------------------------------
    # Reconstruct path
    # --------------------------------------------------------

    def reconstruct_path(self, index):

        path = []

        while index != -1:
            path.append(self.nodes[index])
            index = self.parents[index]

        return np.asarray(path[::-1])

    # --------------------------------------------------------
    # Try final IK connection
    # --------------------------------------------------------

    def try_finish(self, index):

        q = self.nodes[index]

        if not self.reached_goal(q):
            return None

        refined = self.refine_ik(q)

        if refined is None:
            return None

        # Validate final IK motion
        if not self.check_edge(q, refined):
            return None

        self.goal_index = index

        path = self.reconstruct_path(index)

        if np.linalg.norm(refined - q) > 1e-10:
            path = np.vstack([path, refined])

        return path

    # --------------------------------------------------------
    # Build RRT
    # --------------------------------------------------------

    def build_tree(self):

        self.nodes = [self.start.copy()]
        self.parents = [-1]
        self.goal_index = None
        self.iterations = 0

        # Validate start
        if self.in_collision(self.start):
            # print("RRT failed: start configuration in collision")
            return None

        # Find IK candidates
        self.find_goals()

        # print(
        #     "Collision-free IK candidates:",
        #     len(self.goal_solutions)
        # )

        # Check starting configuration
        path = self.try_finish(0)

        if path is not None:
            return path

        for iteration in range(self.max_iterations):

            self.iterations = iteration + 1

            # Sample
            q_rand = self.sample()

            # Nearest
            nearest_index = self.nearest(q_rand)
            q_near = self.nodes[nearest_index]

            # Steer
            q_new = self.steer(q_near, q_rand)

            if np.linalg.norm(q_new - q_near) < 1e-10:
                continue

            # Collision check
            if not self.check_edge(q_near, q_new):
                continue

            # Add node
            self.nodes.append(q_new)
            self.parents.append(nearest_index)

            new_index = len(self.nodes) - 1

            # Check goal and refine IK
            path = self.try_finish(new_index)

            if path is not None:

                # print(
                #     f"Goal reached at iteration {iteration + 1}"
                # )

                # print("Tree size:", len(self.nodes))

                error = np.linalg.norm(
                    forward_kinematics(
                        path[-1], self.lengths
                    )[-1] - self.target
                )

                # print("Final workspace error:", error)

                return path

        # print("RRT failed: maximum iterations reached")
        # print("Tree size:", len(self.nodes))

        return None

    def smooth_path(self, path):
        path = list(np.asarray(path, dtype=float).copy())

        if len(path) <= 2:
            return np.asarray(path)

        changed = True

        while changed:
            changed = False

            # Work backward from the final node
            i = len(path) - 1

            while i >= 2:
                # Can we skip the middle node?
                if self.check_edge(path[i - 2], path[i]):
                    path.pop(i - 1)
                    changed = True

                    # Continue from the new endpoint position
                    i = min(i, len(path) - 1)
                else:
                    i -= 1

        return np.asarray(path)