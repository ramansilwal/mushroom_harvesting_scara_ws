import os
import pickle
import random
import time

import numpy as np
import rclpy
from matplotlib import pyplot as plt
from rclpy.node import Node
from scipy.spatial import KDTree
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped
from shapely import LineString, Point, STRtree
from std_msgs.msg import String as RosString
from tqdm import tqdm

from custom_interfaces.srv import RrtRequest


ANGLE_PATH = '/home/cotrobot/resources/angle_cache/angles_pos.pkl'
ANGLE_LIMITS = np.array([
    [-np.pi / 2, np.pi / 2],
    [-3 * np.pi / 4, 3 * np.pi / 4],
    [-3 * np.pi / 4, 3 * np.pi / 4],
])

if not os.path.exists(ANGLE_PATH):
    raise FileNotFoundError(ANGLE_PATH)

with open(ANGLE_PATH, 'rb') as file:
    angle_data = pickle.load(file)

random_angles = np.asarray(angle_data['angles'])
random_position = np.asarray(angle_data['positions'])
lengths = np.asarray(angle_data['length'])
ANGLE_LIMITS=np.asarray(angle_data['angle_limits'])
kd_tree_kinematics = KDTree(random_position[:, -1, :])


def forward_kinematics_batch(angles, length=lengths):
    angles = np.asarray(angles)
    cumulative = np.cumsum(angles, axis=1)
    offsets = np.stack((length * np.cos(cumulative),
                        length * np.sin(cumulative)), axis=-1)
    positions = np.zeros((len(angles), len(length) + 1, 2), dtype=np.float32)
    positions[:, 1:] = np.cumsum(offsets, axis=1)
    return positions


def forward_kinematics(angles, length=lengths):
    return forward_kinematics_batch(np.asarray(angles).reshape(1, -1), length)[0, -1]


def jacobian(angles, length=lengths):
    cumulative = np.cumsum(angles)
    dx = -length * np.sin(cumulative)
    dy = length * np.cos(cumulative)
    return np.stack((np.cumsum(dx[::-1])[::-1],
                     np.cumsum(dy[::-1])[::-1]))


def inverse_kinematics(target, initial_angles=None, length=lengths,
                       angle_limits=ANGLE_LIMITS, lr=0.05, max_iters=1000, tol=1e-4):
    angles = (np.zeros(len(length)) if initial_angles is None
              else np.array(initial_angles, dtype=float, copy=True))
    target = np.asarray(target)
    for _ in range(max_iters):
        error = target - forward_kinematics(angles, length)
        if np.linalg.norm(error) < tol:
            return angles, error
        angles += lr * (np.linalg.pinv(jacobian(angles, length)) @ error)
        np.clip(angles, angle_limits[:, 0], angle_limits[:, 1], out=angles)
    return None, None


def check_movemnt_possible(start, end, length, obstacles, n=1000):
    if not obstacles:
        return True
    tree = STRtree([getattr(obstacle, 'shape', obstacle) for obstacle in obstacles])
    positions = forward_kinematics_batch(np.linspace(start, end, n), length)
    lines = [LineString(points) for points in positions]
    return len(tree.query(lines, predicate='intersects')[0]) == 0


def check_direct_movemnt_possible(target, initial_angles, robot_link_lengths, obstacles, n=1000):
    solution, _ = inverse_kinematics(np.asarray(target) + 1e-6,
                                    initial_angles, length=robot_link_lengths)
    if solution is None:
        print('failed to improve path')
        return False, None
    if check_movemnt_possible(initial_angles, solution, robot_link_lengths, obstacles, n):
        return True, solution
    return False, None


def most_similar(potential_angles, start_node):
    candidates = np.diff(np.asarray(potential_angles), axis=1)
    reference = np.diff(np.asarray(start_node))
    return potential_angles[np.argmin(np.abs(candidates - reference).sum(axis=1))]


class RRTAngleBatch:
    def __init__(self, start_pos, obstacles=None, tolerance=0.1, step_size=0.1, batch_size=15):
        self.limits = ANGLE_LIMITS
        self.length = lengths
        self.workspace_size = sum(lengths)
        self.start_node = np.asarray(start_pos, dtype=np.float32)
        self.tolerance = tolerance
        self.step_size = step_size
        self.batch_size = batch_size
        self.discriminant = tolerance
        self.tree = {tuple(self.start_node): None}
        self.tree_nodes = [self.start_node]
        self.goal_node = None
        self.error = None
        self.obstacles = [] if obstacles is None else obstacles
        self.kd_tree = KDTree([self.start_node])
        self.obstacle_tree = STRtree(self.obstacles)

    def forward_kinematics(self, angles):
        return forward_kinematics_batch(np.asarray(angles).reshape(1, -1), self.length)[0].T

    def sample_positions_batch(self, n, goal_ws=None, goal_bias=0.15):
        samples = []
        put_closest_sample = False
        for _ in range(n):
            if random.random() > goal_bias:
                samples.append(np.random.uniform(self.limits[:, 0], self.limits[:, 1]))
            elif not put_closest_sample:
                samples.append(self.most_similar_angle)
                put_closest_sample = True
            else:
                samples.append(random.choice(self.potential_angles))
        return np.asarray(samples)

    def nearest_nodes_batch(self, samples):
        _, indices = self.kd_tree.query(samples, k=1)
        return self.kd_tree.data[indices]

    def steer_batch(self, from_nodes, to_nodes):
        directions = to_nodes - from_nodes
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        return from_nodes + directions / (distances + 1e-10) * self.step_size

    def distance_in_workspace_batch(self, angles_batch, goal_ws):
        end_effectors = forward_kinematics_batch(angles_batch, self.length)[:, -1]
        return np.linalg.norm(end_effectors - goal_ws, axis=1)

    def is_in_obstacle_batch(self, angles_batch):
        angles_batch = np.asarray(angles_batch, dtype=float).reshape(-1, len(self.length))
        if not np.all(np.isfinite(angles_batch)):
            raise ValueError('angles_batch must contain only finite angles')
        collisions = np.zeros(len(angles_batch), dtype=bool)
        if not len(angles_batch) or not self.obstacles:
            return collisions
        positions = forward_kinematics_batch(angles_batch, self.length)
        lines = [LineString(points) for points in positions]
        indices, _ = self.obstacle_tree.query(lines, predicate='dwithin', distance=0.05)
        collisions[indices] = True
        return collisions

    def build_tree(self, goal_ws, max_iterations=10_000, goal_bias=0.2):
        
        self.error = None
        self.goal_node = None
        goal_ws = np.asarray(goal_ws, dtype=np.float32)

        indices = kd_tree_kinematics.query_ball_point(
            goal_ws, self.discriminant
        )

        self.potential_angles = random_angles[indices]

        print(f"Goal candidates: {len(self.potential_angles)}")
        print(f"Search radius: {self.discriminant}")

        if len(self.potential_angles) == 0:
            distance, index = kd_tree_kinematics.query(goal_ws)
            print(f"Nearest cached goal distance: {distance:.6f} m")
            print("No goal configurations within search radius")
            return None

        self.most_similar_angle = most_similar(
            self.potential_angles, self.start_node
        )

        if self.is_in_obstacle_batch(self.start_node)[0]:
            print("Start configuration is in collision")
            return None

        print("Starting RRT iterations")
        goal_tree = KDTree(self.potential_angles)

        # Keep your existing RRT iteration loop below this line.


        for _ in tqdm(range(max_iterations)):
            samples = self.sample_positions_batch(self.batch_size, goal_ws, goal_bias)
            nearest = self.nearest_nodes_batch(samples)
            new_nodes = self.steer_batch(nearest, samples)

            valid = np.array([tuple(node) not in self.tree for node in new_nodes])
            if not valid.any():
                continue
            new_nodes, nearest = new_nodes[valid], nearest[valid]
            free = ~self.is_in_obstacle_batch(new_nodes)
            if not free.any():
                continue
            new_nodes, nearest = new_nodes[free], nearest[free]

            for parent, node in zip(nearest, new_nodes):
                key = tuple(node)
                self.tree[key] = tuple(parent)
                self.tree_nodes.append(node)
            self.kd_tree = KDTree(self.tree_nodes)

            distances, _ = goal_tree.query(new_nodes)
            reached = np.flatnonzero(distances < self.tolerance * 2)
            if len(reached):
                best = reached[np.argmin(distances[reached])]
                self.error = float(distances[best])
                self.goal_node = tuple(new_nodes[best])
                return self.reconstruct_path(goal_ws)
        return None

    
    def reconstruct_path(self, goal_ws=None):
        path = []
        node = self.goal_node

        while node is not None:
            path.append(node)
            node = self.tree[node]

        path.reverse()

        if goal_ws is not None:
            target = np.asarray(goal_ws, dtype=float)
            angles = np.asarray(path[-1], dtype=float).copy()

            lr = 0.1
            tol = 1e-4
            max_iters = 1000

            for _ in range(max_iters):
                pos = forward_kinematics(angles, self.length)
                error = target - pos

                if np.linalg.norm(error) < tol:
                    break

                J = jacobian(angles, self.length)
                delta_angles = lr * (np.linalg.pinv(J) @ error)

                angles += delta_angles
                angles = np.clip(
                    angles,
                    self.limits[:, 0],
                    self.limits[:, 1]
                )

            self.pos = target - forward_kinematics(angles, self.length)

            if np.linalg.norm(self.pos) < tol:
                path.append(angles)

        return path


    def render(self, path=None, target=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        for obstacle in self.obstacles:
            shape = getattr(obstacle, 'shape', obstacle)
            if shape.geom_type == 'Polygon':
                x, y = shape.exterior.xy
                ax.fill(x, y, alpha=0.25, fc='green', ec='black', label='Obstacle')
            elif shape.geom_type == 'LineString':
                x, y = shape.xy
                ax.plot(x, y, color='black', linewidth=1.0)
            original = getattr(obstacle, 'og_shape', None)
            if original is not None and original.geom_type == 'Polygon':
                x, y = original.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Obstacle')

        for node, parent in self.tree.items():
            if parent is None:
                continue
            start = self.forward_kinematics(np.asarray(parent))[:, -1]
            end = self.forward_kinematics(np.asarray(node))[:, -1]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color='lightblue', linewidth=0.3, alpha=0.5)

        if path is not None:
            for count, points in enumerate(forward_kinematics_batch(np.asarray(path), self.length), 1):
                ax.plot(points[:, 0], points[:, 1])
                ax.text(*points[-1], str(count))
        if target is not None:
            ax.scatter(target[0], target[1], marker='x')
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'RRT Path Planning (Tree size: {len(self.tree)} nodes)')
        return fig


def smooth_path(path, rrt, robot_radius=0.05, num_checks=100):
    if path is None or len(path) == 0:
        return []
    if len(path) <= 2:
        return path

    path = [np.asarray(node, dtype=float) for node in path]
    new_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        best = i + 1
        for j in range(i + 2, len(path)):
            positions = forward_kinematics_batch(np.linspace(path[i], path[j], num_checks), rrt.length)
            lines = [LineString(points) for points in positions]
            if len(rrt.obstacle_tree.query(lines, predicate='dwithin', distance=robot_radius)[0]):
                break
            best = j
        new_path.append(path[best])
        i = best
    return new_path


class RRT(Node):
    def __init__(self):
        super().__init__('read_obs')
        self.create_subscription(PointCloud2, '/obstcale_points', self.update_obs_points, 10)
        self.create_subscription(JointState, '/joint_states', self.update_joint_poistion, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._find_path, 10)
        self.joint_publisher = self.create_publisher(JointState, '/joint_states_change_request', 10)
        self.status_publisher = self.create_publisher(RosString, '/path_plan_status', 10)
        self.create_timer(0.025, self.timer_callback)
        self.create_service(RrtRequest, '/rrt_service/solver/rrt_solver', self.solve_rrt)
        self.obstacles = None
        self.poisition = None
        self.angles = None
        self.s_path = None
        self.count = None
        self.wait_count = None
        self.status = 'standby'

    def solve_rrt(self, request, response):
        obstacles = [LineString(points) for points in np.asarray(request.obstacles).reshape(-1, 2, 2)]
        target = request.target
        self.get_logger().info(f'Searching path x: {target[0]:.4f}, y: {target[1]:.4f}')
        rrt = RRTAngleBatch(request.angles, obstacles)
        response.angles = []
        response.sucess = False
        path = rrt.build_tree(target)
        if path is not None:
            self.get_logger().info(f'error: {rrt.error}')
            response.sucess = True
            response.angles = np.asarray(smooth_path(path, rrt)).flatten().tolist()
        else:
            self.get_logger().warn('Path planning failed: no path found')
        self.get_logger().info(f'size of tree {len(rrt.tree_nodes)}')
        return response

    def timer_callback(self):
        self.status_publisher.publish(RosString(data=self.status))
        if self.status == 'standby':
            return
        if self.s_path is None:
            self.status = 'standby'
            return
        if self.count >= len(self.s_path) or self.wait_count > 500:
            self.status = 'sucess' if self.count >= len(self.s_path) else 'failure'
            if self.status == 'sucess':
                self.get_logger().info('movement complete')
            for _ in range(10):
                self.status_publisher.publish(RosString(data=self.status))
                time.sleep(0.1)
            self.status = 'standby'
            self.count = self.s_path = self.wait_count = None
            return

        self.wait_count += 1
        self.status = 'running'
        self.get_logger().info(f'running: {self.count}/{len(self.s_path)}')
        angles = self.s_path[self.count]
        if all(abs(self.poisition[f'Angle{i + 1}'] - angle) < 0.02
               for i, angle in enumerate(angles)):
            self.count += 1
        self.move_angles(*angles)

    def update_joint_poistion(self, msg):
        self.poisition = dict(zip(msg.name, msg.position))
        self.angles = [self.poisition[f'Angle{i + 1}'] for i in range(3)]

    def update_obs_points(self, msg):
        if self.poisition is None:
            return
        points = point_cloud2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        self.points = np.asarray([[x, y] for x, y, _ in points]).reshape(-1, 2)
        if not len(self.points):
            self.obstacles = []
            return
        positions = forward_kinematics_batch(np.asarray([self.angles]), lengths)[0]
        offset = self.points - positions[-2]
        norms = np.linalg.norm(offset, axis=1, keepdims=True)
        end1 = offset * 1.25 / np.maximum(norms, 1e-10)
        end2 = offset * 1.1
        ends = np.where(np.linalg.norm(end1, axis=1, keepdims=True) <
                        np.linalg.norm(end2, axis=1, keepdims=True), end1, end2)
        self.obstacles = [LineString([point, end + positions[-2]]).buffer(0.05)
                          for point, end in zip(self.points, ends)]

    def _find_path(self, msg):
        if self.s_path is not None:
            self.status = 'failure'
            self.get_logger().warn('wait for task to complete')
            return
        self.status = 'running'
        self.status_publisher.publish(RosString(data=self.status))
        self.path, self.s_path, self.r = self._get_path([msg.pose.position.x, msg.pose.position.y])
        if not self.s_path:
            self.status = 'failure'
            return
        self.count = 0
        self.wait_count = 0

    def move_angles(self, angle1, angle2, angle3):
        msg = JointState()
        msg.name = list(self.poisition)
        msg.position = list(self.poisition.values())
        msg.position[1:4] = [angle1, angle2, angle3]
        self.joint_publisher.publish(msg)

    def _get_path(self, target):
        self.get_logger().info(f'Searching path x: {target[0]:.4f}, y: {target[1]:.4f}')
        if self.angles is None:
            self.get_logger().warn('RRT initial angles not updated')
            self.status = 'failure'
            return None, None, None
        if self.obstacles is None:
            self.obstacles = []
            self.get_logger().warn('obstacles not updated')
        rrt = RRTAngleBatch(self.angles, self.obstacles)
        if len(rrt.obstacle_tree.query(Point(target), predicate='intersects')):
            self.get_logger().warn('target intersect')
            self.status = 'failure'
            return None, None, rrt
        path = rrt.build_tree(target)
        if path is None:
            return None, None, rrt
        self.get_logger().info(f'error is {rrt.error}')
        return path, smooth_path(path, rrt), rrt


def main(args=None):
    rclpy.init(args=args)
    rrt = RRT()
    try:
        rclpy.spin(rrt)
    finally:
        rrt.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()