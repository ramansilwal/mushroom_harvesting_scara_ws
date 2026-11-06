
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import torch



class RRTAngle:
    def __init__(self, start_pos, obstacles=[], tolerance=0.1, step_size=0.2,):
        self.length = [0.325, 0.275, 0.23]
        self.workspace_size = sum(self.length)
        self.start_node = np.array(start_pos)
        self.tolerance = tolerance
        self.step_size = step_size
        self.tree = {tuple(self.start_node): None}
        self.goal_node = None
        self.obstacles = obstacles
        self.forward_kinematics_cache = {}  
        self.kd_tree = KDTree([self.start_node])
        
    def add_obstacle(self, shape, params):
        self.obstacles.append((shape, params))

    def forward_kinematics(self, angles):
        angles_tuple = tuple(angles)
        if angles_tuple in self.forward_kinematics_cache:
            return self.forward_kinematics_cache[angles_tuple]

        x0, y0 = 0, 0
        x1 = x0 + self.length[0] * np.cos(angles[0])
        y1 = y0 + self.length[0] * np.sin(angles[0])
        x2 = x1 + self.length[1] * np.cos(angles[0] + angles[1])
        y2 = y1 + self.length[1] * np.sin(angles[0] + angles[1])
        x3 = x2 + self.length[2] * np.cos(angles[0] + angles[1]+angles[2])
        y3 = y2 + self.length[2] * np.sin(angles[0] + angles[1]+angles[2])

        result = np.array([[x0, x1, x2,x3], [y0, y1, y2,y3]], dtype=np.float32)
        self.forward_kinematics_cache[angles_tuple] = result  
        return result

    def sample_position(self, goal_ws=None, goal_bias=0.2):
        if goal_ws is not None and np.random.rand() < goal_bias:
            for _ in range(10):  
                sample = [np.random.uniform(-np.pi/2, np.pi/2),
                        np.random.uniform(-3*np.pi/4, 3*np.pi/4),
                        np.random.uniform(-3*np.pi/4, 3*np.pi/4)]
                xs, ys = self.forward_kinematics(sample)
                end_effector = np.array([xs[-1], ys[-1]])
                if np.linalg.norm(end_effector - goal_ws) <  0.2:
                    return sample
        
        return [np.random.uniform(-np.pi/2, np.pi/2),
                np.random.uniform(-3*np.pi/4, 3*np.pi/4),
                np.random.uniform(-3*np.pi/4, 3*np.pi/4)]

    def is_in_obstacle(self, angles):
        xs, ys = self.forward_kinematics(angles)
        for x1, y1, x2, y2 in zip(xs, ys, xs[1:], ys[1:]):
            for shape, params in self.obstacles:
                if shape == 'circle':
                    cx, cy, r = params
                    if self._line_circle_collision(x1, y1, x2, y2, cx, cy, r):
                        return True
                elif shape == 'rectangle':
                    x, y, w, h = params
                    if self._line_rect_collision(x1, y1, x2, y2, x, y, w, h):
                        return True
        return False

    def _line_circle_collision(self, x1, y1, x2, y2, cx, cy, r):
        d = np.array([x2 - x1, y2 - y1])
        f = np.array([x1 - cx, y1 - cy])
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - r * r
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def _line_rect_collision(self, x1, y1, x2, y2, rx, ry, rw, rh):
        for t in np.linspace(0, 1, 20):
            xt = x1 + t * (x2 - x1)
            yt = y1 + t * (y2 - y1)
            if rx <= xt <= rx + rw and ry <= yt <= ry + rh:
                return True
        return False

    def nearest_node(self, sample):
        
        dist, idx = self.kd_tree.query([sample], k=1)
        nearest_node = tuple(self.kd_tree.data[idx][0])  
        return nearest_node

    def steer(self, from_node, to_node):
        direction = np.array(to_node) - np.array(from_node)
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return np.array(to_node)
        direction /= distance
        return np.array(from_node) + self.step_size * direction

    def distance_in_workspace(self, angles, goal_ws):
        xs, ys = self.forward_kinematics(angles)
        end_effector = np.array([xs[-1], ys[-1]])
        distance = np.linalg.norm(end_effector - goal_ws)
        return distance

    def build_tree(self, goal_ws, max_iterations=5000):
        goal_ws = np.array(goal_ws, dtype=np.float32)

        for i in tqdm(range(max_iterations)):
            sample = self.sample_position()
            nearest = self.nearest_node(sample)
            new_node = self.steer(nearest, sample)
            new_node_tuple = tuple(new_node)

            distance = self.distance_in_workspace(new_node, goal_ws)

            if new_node_tuple in self.tree:
                continue
            if self.is_in_obstacle(new_node):
                continue

            self.tree[new_node_tuple] = tuple(nearest)
            self.kd_tree = KDTree(list(self.tree.keys()))  # Rebuild the KDTree after adding new node

            if distance < self.tolerance:
                self.goal_node = new_node_tuple
                print(f"Goal reached in {i} iterations!")
                path= self.reconstruct_path()
                return path

        print("Goal not reached within iteration limit.")
        return None

    def reconstruct_path(self):
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node)
            node = self.tree[node]
        return path[::-1]

from matplotlib import pyplot as plt
from matplotlib import patches
class RRTStarAngle(RRTAngle):
    def __init__(self, start_pos, obstacles=[], tolerance=0.2, step_size=0.25, radius=0.2):
        super().__init__(start_pos, obstacles, tolerance, step_size)
        self.radius = radius

        
        self.tree = {tuple(self.start_node): {'parent': None, 'cost': 0.0}}

    def build_tree(self, goal_ws, max_iterations=5000):
        goal_ws = np.array(goal_ws, dtype=np.float32)

        for i in tqdm(range(max_iterations)):
            sample = self.sample_position()
            nearest = self.nearest_node(sample)
            new_node = self.steer(nearest, sample)
            new_node_tuple = tuple(new_node)

            if new_node_tuple in self.tree:
                continue
            if self.is_in_obstacle(new_node):
                continue

            parent = nearest
            min_cost = self.tree[parent]['cost'] + self.cost(parent, new_node)

            all_nodes = list(self.tree.keys())
            self.kd_tree = KDTree(all_nodes)
            nearby_indices = self.kd_tree.query_ball_point(new_node, self.radius)
            nearby_nodes = [tuple(self.kd_tree.data[idx]) for idx in nearby_indices]
            for neighbor in nearby_nodes:
                if neighbor == new_node_tuple:
                    continue
                tentative_cost = self.tree[neighbor]['cost'] + self.cost(neighbor, new_node)
                if tentative_cost < min_cost and not self.is_in_obstacle(new_node):
                    parent = neighbor
                    min_cost = tentative_cost

            # Add new node with best parent
            self.tree[new_node_tuple] = {'parent': parent, 'cost': min_cost}

            # Rewire neighbors to go through new node if it's cheaper
            for neighbor in nearby_nodes:
                if neighbor == new_node_tuple:
                    continue
                tentative_cost = self.tree[neighbor]['cost'] + self.cost(neighbor, new_node)


                if tentative_cost < min_cost and self.is_collision_free(neighbor, new_node):
                    parent = neighbor
                    min_cost = tentative_cost
            # Check if goal is reached
            distance = self.distance_in_workspace(new_node, goal_ws)
            if distance < self.tolerance:
                self.goal_node = new_node_tuple
                path= self.reconstruct_path()
                return path

        print("Goal not reached within iteration limit.")
        return None

    def reconstruct_path(self):
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node)
            node = self.tree[node]['parent']
        return path[::-1]

    def cost(self, from_node, to_node):
        return np.linalg.norm(np.array(from_node) - np.array(to_node))

    def plot_tree(self):
        for key,values in self.tree.items():
            if values['parent'] is not None:
                plt.plot([key[0],values['parent'][0]],[key[1],values['parent'][1]])
        plt.show()

    def plot_ws_and_obstcale(self,path=None):

        fig, ax = plt.subplots(figsize=(6, 6))

        # Set workspace limits
        ws = self.workspace_size
        ax.set_xlim(-ws, ws)
        ax.set_ylim(-ws, ws)
        ax.set_aspect('equal')
        ax.set_title("RRT Tree with Obstacles")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Plot each obstacle
        for shape, params in self.obstacles:
            if shape == 'circle':
                cx, cy, r = params
                circle = patches.Circle((cx, cy), r, color='red', alpha=0.5, label='Circle')
                ax.add_patch(circle)
            elif shape == 'rectangle':
                x, y, w, h = params
                rect = patches.Rectangle((x, y), w, h, color='blue', alpha=0.5, label='Rectangle')
                ax.add_patch(rect)

        # Plot the RRT tree
        for key, parent in self.tree.items():
            if parent['parent'] is None:
                continue
            child_coords = self.forward_kinematics(key)
            parent_coords = self.forward_kinematics(parent['parent'])
            # Draw a line between end effectors of child and parent
            ax.plot(
                [child_coords[0, -1], parent_coords[0, -1]],
                [child_coords[1, -1], parent_coords[1, -1]],
                color='green',
                linewidth=0.5
            )
        if path:
            for key, parent in zip(path,path[1:]):
                child_coords = self.forward_kinematics(key)
                parent_coords = self.forward_kinematics(parent)
                # Draw a line between end effectors of child and parent
                ax.plot(
                    [child_coords[0, -1], parent_coords[0, -1]],
                    [child_coords[1, -1], parent_coords[1, -1]],
                    color='red',
                    linewidth=0.5
                )


        # Plot robot base
        ax.plot(0, 0, 'ko', label='Base')

        # Add legend (avoid duplicates)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        plt.grid(True)
        plt.show()
    def is_collision_free(self, from_node, to_node, steps=10):
        """Check for collisions along the interpolated path between from_node and to_node."""
        from_node = np.array(from_node)
        to_node = np.array(to_node)

        for alpha in np.linspace(0, 1, steps):
            interp = (1 - alpha) * from_node + alpha * to_node
            if self.is_in_obstacle(interp):
                return False
        return True
