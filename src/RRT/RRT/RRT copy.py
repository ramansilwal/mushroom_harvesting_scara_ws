import rclpy

from rclpy.node import Node


from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 

from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree
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


def check_movemnt_possible(start,end,length,obstcales,n=1000):

    obstacle_tree = STRtree([obstacle.shape for obstacle in obstcales])

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
    if check_movemnt_possible(initial_angles,angles_sol,robot_link_lengths,obstcales,n):
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
    def __init__(self, start_pos, obstacles=[], tolerance=0.1, step_size=0.1, batch_size=15):
        self.limits = np.array([
                [-np.pi/2, np.pi/2],
                [-3*np.pi/4, 3*np.pi/4],
                [-3*np.pi/4, 3*np.pi/4]
                        ])
        
        self.length = lengths
        self.workspace_size = sum(self.length)
        self.start_node = np.array(start_pos,dtype=np.float32)
        self.tolerance = tolerance
        self.step_size = step_size
        self.batch_size = batch_size
        self.discriminant = tolerance
        self.tree = {tuple(self.start_node): None}
        self.tree_nodes = [self.start_node]
        self.goal_node = None
        self.error = None  # Set only when a goal configuration is reached.
        self.obstacles = obstacles
        
        self.kd_tree = KDTree([self.start_node])
        # self.load_kinematics_tree()
        self.obstacle_tree = STRtree([obstacle for obstacle in obstacles])
    
    def forward_kinematics(self, angles):
        """Single configuration FK for compatibility"""
        result = forward_kinematics_batch(angles.reshape(1, -1), self.length)
        return result[0].T  # Shape: (2, 4)
    
    def sample_positions_batch(self, n, goal_ws=None, goal_bias=0.15):
        """Sample n configurations at once"""
        
        samples=[]
        put_closest_sample=False
        for i in range(0,n):
            if random.uniform(0,1)>goal_bias:
                samples.append([np.random.uniform(*i) for i in self.limits])
            else:
                if not put_closest_sample:
                    samples.append(self.most_similar_angle)
                    put_closest_sample=True
                    continue
                samples.append(random.choice(self.potential_angles))
        return np.array(samples)
    
    def nearest_nodes_batch(self, samples):
        """Find nearest node for each sample"""
        dist, idx = self.kd_tree.query(samples, k=1)
        nearest_nodes = self.kd_tree.data[idx]
        return nearest_nodes

    def steer_batch(self, from_nodes, to_nodes):
        """
        Steer from from_nodes toward to_nodes with dynamic step size.
        Step size is proportional to the distance to the target, clipped between min_step and max_step.
        """
        directions = to_nodes - from_nodes
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        
        
        # Compute new nodes
        new_nodes = from_nodes + (directions / (distances + 1e-10)) * self.step_size

        return new_nodes

    def distance_in_workspace_batch(self, angles_batch, goal_ws):
        """Compute workspace distance for batch of configurations"""
        configs = forward_kinematics_batch(angles_batch, self.length)
        end_effectors = configs[:, -1, :]  # Shape: (n, 2)
        distances = np.linalg.norm(end_effectors - goal_ws, axis=1)
        return distances


    def is_in_obstacle_batch(self, angles_batch):
        """Return one collision flag per configuration, with 5 cm link clearance.

        Accept an (N, 3) batch or a single (3,) configuration. Contact at
        the clearance boundary counts as collision.
        """
        angles_batch = np.asarray(angles_batch, dtype=float)
        if angles_batch.shape == (0,):
            angles_batch = angles_batch.reshape(0, len(self.length))
        elif angles_batch.ndim == 1:
            angles_batch = angles_batch.reshape(1, -1)
        if angles_batch.ndim != 2 or angles_batch.shape[1] != len(self.length):
            raise ValueError("angles_batch must have shape (N, 3) or (3,)")
        if not np.all(np.isfinite(angles_batch)):
            raise ValueError("angles_batch must contain only finite angles")

        collisions = np.zeros(len(angles_batch), dtype=bool)
        if len(angles_batch) == 0:
            return collisions

        configs = forward_kinematics_batch(angles_batch, self.length)
        lines_batch = np.array([LineString(cfg) for cfg in configs], dtype=object)
        # Distance avoids the polygonal approximation of round buffer caps
        # and joins, which can miss obstacles just inside the clearance.
        colliding_indices, _ = self.obstacle_tree.query(
            lines_batch, predicate="dwithin", distance=0.05
        )
        # The first query row indexes configurations, not obstacles.
        collisions[colliding_indices] = True
        return collisions
    
    def build_tree(self, goal_ws, max_iterations=10_000, goal_bias=0.2):
        """Build RRT tree toward goal in workspace"""
        self.error = None
        goal_ws = np.array(goal_ws, dtype=np.float32)
        mask=kd_tree_kineamtics.query_ball_point(goal_ws,self.discriminant)
        self.potential_angles=random_angles[mask]
        self.most_similar_angle=most_similar(self.potential_angles, self.start_node)
        if len(self.potential_angles)==0:
            print("solution not found")
            return None
        col=self.is_in_obstacle_batch(np.array([self.start_node]))[0]
        if col:
            print("start config in collision")
            return None

        for iteration in tqdm(range(max_iterations)):
            # Sample batch of configurations
            samples = self.sample_positions_batch(self.batch_size, goal_ws, goal_bias)
            
            # Find nearest nodes for each sample
            nearest_nodes = self.nearest_nodes_batch(samples)
            
            # Steer toward samples
            new_nodes = self.steer_batch(nearest_nodes, samples)
            
            # Check for duplicates in tree
            valid_mask = np.ones(self.batch_size, dtype=bool)
            for i in range(self.batch_size):
                new_node_tuple = tuple(new_nodes[i])
                if new_node_tuple in self.tree:
                    valid_mask[i] = False
            
            # Filter valid nodes
            if not np.any(valid_mask):
                continue
            
            new_nodes_valid = new_nodes[valid_mask]
            nearest_nodes_valid = nearest_nodes[valid_mask]
            
            # Check collisions
            collisions = self.is_in_obstacle_batch(new_nodes_valid)
            collision_free_mask = ~collisions
            
            if not np.any(collision_free_mask):
                continue
            
            # Add collision-free nodes to tree
            new_nodes_final = new_nodes_valid[collision_free_mask]
            nearest_nodes_final = nearest_nodes_valid[collision_free_mask]
            valid_nodes_final=[]
            valid_nearest_nodes=[]
            for i,j in zip(nearest_nodes_final,new_nodes_final):
                # if simulate_movement(i,j,self.length,self.obstacle_tree,n=10):
                valid_nearest_nodes.append(i)
                valid_nodes_final.append(j)
            for i,j in zip(valid_nearest_nodes,valid_nodes_final):
                new_node_tuple = tuple(j)
                nearest_tuple = tuple(i)
                
                self.tree[new_node_tuple] = nearest_tuple
                self.tree_nodes.append(j)
            
            # Rebuild KD-tree with new nodes
            if iteration%1==0:
                self.kd_tree = KDTree(self.tree_nodes)
            
            distances = np.linalg.norm(new_nodes_final[:, None, :] - self.potential_angles[None, :, :],axis=2)
            
            min_distances = distances.min(axis=1)

            # New nodes close enough to goal configuration
            goal_reached_indices = np.where(min_distances < self.tolerance*2)[0]



            if len(goal_reached_indices) > 0:
                # Pick the closest one
                best_idx = goal_reached_indices[np.argmin(min_distances[goal_reached_indices])]
                self.error=np.min(min_distances[goal_reached_indices])
                self.goal_node = tuple(new_nodes_final[best_idx])

                path = self.reconstruct_path(goal_ws)
                return path

        
        return None

    def reconstruct_path(self,goal_ws=None):
        path = []
        node = self.goal_node
        while node is not None:
            path.append(node)
            node = self.tree[node]
        revered_path=path[::-1]
        if goal_ws is not None:
            ik_angle,pos=inverse_kinematics(goal_ws,np.array(revered_path[-1]))
            self.pos=pos
            if ik_angle is not None:
                revered_path.append(ik_angle)


        return revered_path
    
    def render(self, path=None, target=None):
        fig, ax = plt.subplots(figsize=(4,4))

        # Plot obstacles
        for obstacle in self.obstacles:
            shape = getattr(obstacle, "shape", obstacle)
            if shape.geom_type == "Polygon":
                x, y = shape.exterior.xy
                ax.fill(x, y, alpha=0.25, fc="green", ec="black", label="Obstacle")
            elif shape.geom_type == "LineString":
                x, y = shape.xy
                ax.plot(x, y, color="black", linewidth=1.0)

            og_shape = getattr(obstacle, "og_shape", None)
            if og_shape is not None and og_shape.geom_type == "Polygon":
                x, y = og_shape.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Obstacle")

        # Plot tree
        for node, parent_node in self.tree.items():
            if parent_node is None:
                continue
            parent = self.forward_kinematics(np.array(parent_node))
            child = self.forward_kinematics(np.array(node))
            ax.plot(
                [parent[0, -1], child[0, -1]],
                [parent[1, -1], child[1, -1]],
                color="lightblue",
                linewidth=0.3,
                alpha=0.5,
            )

        # Plot path
        if path is not None:
            path=np.array(path)
            count=0
            for i in forward_kinematics_batch(path, self.length):
                ax.plot(i[:,0],i[:,1])
                count+=1
                ax.text(i[-1][0],i[-1][1],f"{count}")

        # Plot target
        if target is not None:
            ax.scatter(target[0],target[1],marker="x")
        
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'RRT Path Planning (Tree size: {len(self.tree)} nodes)')
        
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
        if path:
            self.get_logger().info(f"final error {np.linalg.norm(np.array(target)-np.array(r.pos))}")
            output.sucess=True
            s_path=smooth_path(path,r)
            print(path,s_path)
            output.angles=np.array(s_path).flatten().tolist()

        self.get_logger().info(f"sizre of tree {len(r.tree_nodes)}")
        if path is None:
            self.get_logger().warn("Path planning failed: no path found")
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
