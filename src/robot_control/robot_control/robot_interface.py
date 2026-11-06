import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from .Hitbot.RobotWrapper import ScaraRobot
from example_interfaces.srv import Trigger

from custom_interfaces.msg import GripperAngle
from std_msgs.msg import Float32 as ROSFloat
from custom_interfaces.srv import RobotListAngleMove
import numpy as np
import time

class Robot(Node, ScaraRobot):
    def __init__(self):
        Node.__init__(self, 'robot_control_node') 
        ScaraRobot.__init__(self)
        self.robot_move_status = self.create_publisher(String, "move_status", 10)
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        
        self.joint_state_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.joint_state_changer = self.create_subscription(JointState, "/joint_states_change_request",self.update_joint_states, 10)
        self.joint_position = {"PrismaticJointZ": self.z/1000, "Angle1":np.deg2rad(self.angle1), "Angle2":np.deg2rad(self.angle2),"Angle3":np.deg2rad(self.angle3),
                               "GripperJoint1":0.0, "GripperJoint2":0.0, "GripperJoint3":0.0}
        self.move_flag=False
        self.reset_home = self.create_service(Trigger,"/go_to_home_poisition",self.rest_to_home)
        self.joint_name = ["PrismaticJointZ", "Angle1", "Angle2", "Angle3", "GripperJoint1", "GripperJoint2", "GripperJoint3"]
        self.publish_joint_state()
        self.gripper_joint_subscriber=self.create_subscription(GripperAngle, "/current_gripper_angle",self.update_gripper_angle,10) 
        self.create_subscription(ROSFloat,"/z_change_request",self.process_change_z,10)
        self.create_service(RobotListAngleMove,"/robot_service/move/config_rep_list",self.config_rep_list)

    def config_rep_list(self,request,respone):
        for rep in request.list_of_angles:
            if rep.speed<50:
                rep.speed=50.0
            self.move_joint_radian2(rep.angle1,rep.angle2,rep.angle3,rep.z,rep.speed)
        self.wait_stop()
        self.update()
        self.publish_joint_state()
        # time.sleep(0.1)
        respone.sucess=True
        return respone

    def update_gripper_angle(self,msg):
        self.joint_position["GripperJoint1"]=msg.angle1
        self.joint_position["GripperJoint2"]=msg.angle2
        self.joint_position["GripperJoint3"]=msg.angle3
    
    def process_change_z(self,msg):
        self.move_z(int(msg.data * 1000),speed=80)
        self.publish_joint_state()

    def publish_joint_state(self):
        self.update()
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_name

        joint_msg.position = [self.z/1000, 
                              float(np.deg2rad(self.angle1)), float(np.deg2rad(self.angle2)), float(np.deg2rad(self.angle3)), 
                              self.joint_position["GripperJoint1"], self.joint_position["GripperJoint2"], self.joint_position["GripperJoint3"]]
        self.joint_state_publisher.publish(joint_msg)

    def rest_to_home(self,request,response):
        try:
            self.go_home()
            response.success = True
            response.message = "Robot reset to home poistion base_link."
        except Exception as e:
            response.success = False
            response.message = e
        finally:
            self.publish_joint_state()            
            return response

    def update_joint_states(self, msg):
        # Save joint positions as a dictionary
        self.joint_position = {name: pos for name, pos in zip(msg.name, msg.position)}
        self.get_logger().info(f"new_joint_pos_recviced")
        self.move_flag=True

    def move_status_publisher(self,stat):
        move_msg = String()
        move_msg.data = stat
        self.robot_move_status.publish(move_msg)
        #self.get_logger().info(f"Robot status: {stat}")

    def timer_callback(self):
        if self.joint_position is None:
            return
        
        if self.move_flag:
            self.move_status_publisher("Moving")

            angle1 = self.joint_position.get("Angle1", 0.0)
            angle2 = self.joint_position.get("Angle2", 0.0)
            angle3 = self.joint_position.get("Angle3", 0.0)
            prismatic_z = self.joint_position.get("PrismaticJointZ", 0.0)
            self.get_logger().info(f"del angle 1:{abs(self.angle1-angle1)}")
            self.get_logger().info(f"del angle 2:{abs(self.angle2-angle2)}")
            self.get_logger().info(f"del angle 3:{abs(self.angle3-angle3)}")


            cost= self.move_joint_radian(angle1, angle2, angle3,speed=80)
            self.get_logger().info(f"cost of movement{cost}")
            self.update()
            self.get_logger().info(f"move_joint_radian,{angle1}, {angle2}, {angle3}")  
            self.move_z(int(prismatic_z * 1000),speed=80)
            self.get_logger().info(f"new_z,{self.z}")  
            self.move_status_publisher("Free")
            self.move_flag=False
        else:
            self.move_status_publisher("Free")
            self.move_flag=False
        self.publish_joint_state()

        
def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = Robot()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()