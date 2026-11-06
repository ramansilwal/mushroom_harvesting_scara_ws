from sensor_msgs.msg import LaserScan,JointState
from rclpy.node import Node
import time
import rclpy

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from std_msgs.msg import Header


def forward_kinematics_batch(angles, length=np.array([0.325, 0.275, 0.225])):
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


import numpy as np

def distance_to_line_segment(point, a, b):
    point = np.array(point, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    ab = b - a
    ap = point - a

    # projection factor of point onto line AB
    t = np.dot(ap, ab) / np.dot(ab, ab)

    # clamp t so closest point stays on the segment
    t = np.clip(t, 0, 1)

    closest_point = a + t * ab

    distance = np.linalg.norm(point - closest_point)

    return distance, closest_point


class LaserScanNode(Node):

    def __init__(self):
        super().__init__("laser_scan")
        self.initalize()
        self.timmer=self.create_timer(1,self.process)
        self.distance_filter=0.2
        self.upper_range=1.5
        self.min_intensity=5
        self.pc2_publisher = self.create_publisher(PointCloud2, "obstcale_points", 20)
        self.pc2_publisher_removed = self.create_publisher(PointCloud2, "obstcale_points_removed", 20)

        self.tof_sensor=np.inf

        self.create_subscription(Float32,"/tof_data",self.update_tof,10)
    
    def update_tof(self,msg):
        self.tof_sensor=msg.data

    def initalize(self):
        self.laser_scan = None
        self.robot_poistion=None

        self.laser_subscriber = self.create_subscription(
            LaserScan,
            "/scan",
            self.update_laser,
            10
        )
        self.robot_poistion_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self.update_robot,
            10
        )

        
    def update_robot(self,msg):
        self.robot_poistion=msg
    
    def update_laser(self, msg):
        self.laser_scan = msg

    
    def process(self):
        laser_scan=self.laser_scan
        robot_poistion=self.robot_poistion
        if laser_scan is None:
            self.get_logger().warn("Laser processer cannot find laser Scan")
        if robot_poistion is None:
            self.get_logger().warn("Laser processer cannot find Robot joint poistion")
        if laser_scan is None or robot_poistion is None:
            return
        jp={x:y for x,y in zip(robot_poistion.name,robot_poistion.position)}
        position=forward_kinematics_batch(np.array([[jp["Angle1"],jp["Angle2"],jp["Angle3"]]]))[0]
        pos=[]
        angle=laser_scan.angle_min#-15*laser_scan.angle_increment
        count=0
        while count<len(laser_scan.ranges):#angle<laser_scan.angle_max-4*laser_scan.angle_increment:
            x=np.cos(angle-jp["Angle1"]-jp["Angle2"]-jp["Angle3"])*laser_scan.ranges[count]
            y=np.sin(angle-jp["Angle1"]-jp["Angle2"]-jp["Angle3"])*laser_scan.ranges[count]
            if laser_scan.intensities[count]>self.min_intensity and np.linalg.norm([x,y])<self.upper_range:
                pos.append([-x,y])
            # print(count,angle,laser["ranges"][count],laser["intensities"][count])

            count+=1
            angle+=laser_scan.angle_increment
        pos=np.array(pos)

        pos_true=pos+position[2]
        # pos_true=pos_true-np.array([0,0.02])
        mask = (
            (np.linalg.norm(pos_true, axis=1) > 0.35) &
            (np.linalg.norm(pos_true - position[-1], axis=1) > 0.15) &
            (np.linalg.norm(pos_true - position[-2], axis=1) > 0.12)

        )
        pos_filtered=pos_true[mask]
        removed_points=pos_true[~mask]

        selected_points=[[x,y,0] for x,y in pos_filtered]
        
        # if self.tof_sensor<1:
        #     position=forward_kinematics_batch(
        #         np.array([[jp["Angle1"],jp["Angle2"],jp["Angle3"]]]),
        #         np.array([0.325, 0.275, 0.225+self.tof_sensor]))[0][-1]
            
        #     if np.linalg.norm(position)<self.upper_range:
        #         selected_points.append([position[0],position[1],0])

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            selected_points
        )

        self.pc2_publisher.publish(cloud_msg)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "base_link"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            [[x,y,0] for x,y in removed_points]
        )

        self.pc2_publisher_removed.publish(cloud_msg)
        # return np.array(points_selected)


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

