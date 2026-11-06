import rclpy
from rclpy.node import Node
from adafruit_servokit import ServoKit
import time
import board
import busio
from adafruit_ina219 import INA219
import time
from example_interfaces.srv import Trigger
from custom_interfaces.msg import GripperAngle
from custom_interfaces.srv import GripperAngleService
import numpy as np
from std_msgs.msg import String
from collections import deque

class Gripper(Node):
    def __init__(self):
        super().__init__("GripperNode")
        self.kit = ServoKit(channels=16)
        i2c_bus = busio.I2C(board.SCL, board.SDA)
        self.ina219 = INA219(i2c_bus, addr=0x41)
        self.ids=[4,6,2]
        self.init_angle=40
        self.init_angles()  # Starting angle
        self.max_tol_angle = 175
        self.open_gripper_service = self.create_service(
            Trigger, "open_gripper", self.open_gripper
        )
        self.close_gripper_service = self.create_service(
            Trigger, "close_gripper", self.close_gripper
        )
        self.tasks_=self.create_subscription(String,"/gripper_task",self.update_task,10)
        self.time_period=1.0
        self.timmer_ = self.create_timer(self.time_period,self.timer_callback)
        self.angle_publisher=self.create_publisher(GripperAngle,"/current_gripper_angle",10)
        self.close_to_angle_service_ = self.create_service(GripperAngleService,"close_gripper_angle",self.close_angle_to_service)
        self.alpha = 0.1
        self.buffer = deque(maxlen=10)
        self.filtered = None

    def read_current(self):
        # Read raw current
        current = self.ina219.current

        # Remove spikes using median
        self.buffer.append(current)
        median = np.median(self.buffer)

        # Smooth using EMA
        if self.filtered is None:
            self.filtered = median
        else:
            self.filtered = (
                self.alpha * median
                + (1 - self.alpha) * self.filtered
            )

        print(self.filtered)

        return float(self.filtered)
    def update_task(self,msg):
        if "close" in msg.data:
            self.close()
        if "open" in msg.data:
            self.open()
        
    def close_angle_to_service(self,request,response):
        angle1=int(np.rad2deg(abs(request.angle1)))
        angle2=int(np.rad2deg(abs(request.angle2)))
        angle3=int(np.rad2deg(abs(request.angle3)))
        self.get_logger().info(f"request to change angle:{angle1},{angle2},{angle3}")
        if angle1>self.max_tol_angle:
            self.get_logger().warn("angle1 more than 150")
            angle1=self.max_tol_angle
        if angle2>self.max_tol_angle:
            self.get_logger().warn("angle2 more than 150")
            angle2=self.max_tol_angle
        if angle3>self.max_tol_angle:
            self.get_logger().warn("angle3 more than 150")
            angle3=self.max_tol_angle
        id1,id2,id3=self.ids
        self.kit.servo[id1].angle=angle1
        self.kit.servo[id2].angle=angle2
        self.kit.servo[id3].angle=angle3
        self.angle1,self.angle2,self.angle3 = self.kit.servo[id1].angle,self.kit.servo[id2].angle,self.kit.servo[id3].angle

        response.success=True
        return response

    def init_angles(self):
        id1,id2,id3=self.ids
        self.kit.servo[id1].angle=self.init_angle
        self.kit.servo[id2].angle=self.init_angle
        self.kit.servo[id3].angle=self.init_angle
        self.angle1,self.angle2,self.angle3 = self.kit.servo[id1].angle,self.kit.servo[id2].angle,self.kit.servo[id3].angle
        print("angles set to", self.angle1, self.angle2, self.angle3)

    def open_gripper(self, request, response):
        try:
            self.open()
            response.success=True
            response.message="gripper open sucess"
        except Exception as e:
            response.success=False
            response.message=e
        finally:
            return response

    def close_gripper(self, request, response):
        try:
            self.close()
            response.success=True
            response.message="gripper close sucess"
        except Exception as e:
            response.success=False
            response.message=e
        finally:
            return response
    
    def step(self,id1,id2,id3,step_size=3):
        self.angle1 = self.angle1 + step_size if self.angle1 < self.max_tol_angle else self.angle1
        self.kit.servo[id1].angle = self.angle1
        self.angle2 = (
            self.angle2 + step_size if self.angle2 < self.max_tol_angle else self.angle2
        )
        self.kit.servo[id2].angle = self.angle2
        self.angle3 = (
            self.angle3 + step_size if self.angle3 < self.max_tol_angle else self.angle3
        )
        self.kit.servo[id3].angle = self.angle3
        
    def close(self):
        id1,id2,id3=self.ids

        while True:
            self.step(id1,id2,id3)
            
            try:
                current = self.read_current()
                if (current >250):
                    self.get_logger().info("current condition statified")
                    break
            except Exception as e:
                self.get_logger().warn("error in reading current ")
            
            
            if (self.angle1 + self.angle2 + self.angle3) > 480:
                self.get_logger().info("closing completed max angle condition statified")
                return
            time.sleep(0.025) 
        time.sleep(0.5)
        while True:
            self.step(id1,id2,id3)
            time.sleep(0.05) 
            try:
                current = self.read_current()
                if (current > 350):
                    self.get_logger().info("current condition statified")
                    break
            except Exception as e:
                self.get_logger().warn("error in reading current ")
            
            
            if (self.angle1 + self.angle2 + self.angle3) > 480:
                self.get_logger().info("closing completed max angle condition statified")
                break
            time.sleep(0.025) 
        self.step(id1,id2,id3)
        self.step(id1,id2,id3)

    def open(self):
        self.get_logger().info("opening gripper")
        id1,id2,id3=self.ids
        self.angle1 = self.init_angle
        self.angle2 = self.init_angle
        self.angle3 = self.init_angle
        self.kit.servo[id1].angle = self.angle1
        self.kit.servo[id2].angle = self.angle2
        self.kit.servo[id3].angle = self.angle3
        self.get_logger().info("opening gripper completed")
    def timer_callback(self):
        msg=GripperAngle()
        msg.angle1 = float(np.deg2rad(-self.angle1))
        msg.angle2 = float(np.deg2rad(-self.angle2))
        msg.angle3 = float(np.deg2rad(-self.angle3))
        # self.get_logger().info(f"publishing {msg.angle1},{msg.angle2},{msg.angle3}")
        self.angle_publisher.publish(msg)


# class Gripper(Node):
#     def __init__(self):
#         super().__init__("GripperNode")
#         self.kit = ServoKit(channels=16)
#         i2c_bus = busio.I2C(board.SCL, board.SDA)
#         self.ina219 = INA219(i2c_bus, addr=0x41)
#         self.ids=[4,6,2]
#         self.init_angle=40
#         self.init_angles()  # Starting angle
#         self.max_tol_angle = 165
#         self.open_gripper_service = self.create_service(
#             Trigger, "open_gripper", self.open_gripper
#         )
#         self.close_gripper_service = self.create_service(
#             Trigger, "close_gripper", self.close_gripper
#         )
#         self.tasks_=self.create_subscription(String,"/gripper_task",self.update_task,10)
#         self.time_period=1.0
#         self.timmer_ = self.create_timer(self.time_period,self.timer_callback)
#         self.angle_publisher=self.create_publisher(GripperAngle,"/current_gripper_angle",10)
#         self.close_to_angle_service_ = self.create_service(GripperAngleService,"close_gripper_angle",self.close_angle_to_service)
    
#     def update_task(self,msg):
#         if "close" in msg.data:
#             self.close()
#         if "open" in msg.data:
#             self.open()
        
#     def close_angle_to_service(self,request,response):
#         angle1=int(np.rad2deg(abs(request.angle1)))
#         angle2=int(np.rad2deg(abs(request.angle2)))
#         angle3=int(np.rad2deg(abs(request.angle3)))
#         self.get_logger().info(f"request to change angle:{angle1},{angle2},{angle3}")
#         if angle1>self.max_tol_angle:
#             self.get_logger().warn("angle1 more than 150")
#             angle1=self.max_tol_angle
#         if angle2>self.max_tol_angle:
#             self.get_logger().warn("angle2 more than 150")
#             angle2=self.max_tol_angle
#         if angle3>self.max_tol_angle:
#             self.get_logger().warn("angle3 more than 150")
#             angle3=self.max_tol_angle
#         id1,id2,id3=self.ids
#         self.kit.servo[id1].angle=angle1
#         self.kit.servo[id2].angle=angle2
#         self.kit.servo[id3].angle=angle3
#         self.angle1,self.angle2,self.angle3 = self.kit.servo[id1].angle,self.kit.servo[id2].angle,self.kit.servo[id3].angle

#         response.success=True
#         return response

#     def init_angles(self):
#         id1,id2,id3=self.ids
#         self.kit.servo[id1].angle=self.init_angle
#         self.kit.servo[id2].angle=self.init_angle
#         self.kit.servo[id3].angle=self.init_angle
#         self.angle1,self.angle2,self.angle3 = self.kit.servo[id1].angle,self.kit.servo[id2].angle,self.kit.servo[id3].angle
#         print("angles set to", self.angle1, self.angle2, self.angle3)

#     def open_gripper(self, request, response):
#         try:
#             self.open()
#             response.success=True
#             response.message="gripper open sucess"
#         except Exception as e:
#             response.success=False
#             response.message=e
#         finally:
#             return response

#     def close_gripper(self, request, response):
#         try:
#             self.close()
#             response.success=True
#             response.message="gripper close sucess"
#         except Exception as e:
#             response.success=False
#             response.message=e
#         finally:
#             return response
#     def step(self,id1,id2,id3,step_size=3):
#         self.angle1 = self.angle1 + step_size if self.angle1 < self.max_tol_angle else self.angle1
#         self.kit.servo[id1].angle = self.angle1
#         self.angle2 = (
#             self.angle2 + step_size if self.angle2 < self.max_tol_angle else self.angle2
#         )
#         self.kit.servo[id2].angle = self.angle2
#         self.angle3 = (
#             self.angle3 + step_size if self.angle3 < self.max_tol_angle else self.angle3
#         )
#         self.kit.servo[id3].angle = self.angle3
        
#     def close(self):
#         id1,id2,id3=self.ids
        
#         current=0
#         all_current=[0 for i in range(25)]
#         s_time=time.time()
#         while True:
#             self.step(id1,id2,id3)
#             if time.time()-s_time>2:
#                 self.get_logger().info(f"TIME {time.time()-s_time}")
#                 break   
#             try:
#                 current = self.ina219.current
#                 all_current=all_current[1:]
#                 all_current.append(current)
#             except Exception as e:
#                 self.get_logger().warn("error in reading current ")
#             sum_current=sum([all_current[i]*(i+1)/25 for i in range(len(all_current))]) 
#             self.get_logger().info(f"current:{current},{sum_current}")
#             if (self.angle1 + self.angle2 + self.angle3) > 450:
#                 self.get_logger().info("closing completed max angle condition statified")
#                 break
#             if (sum_current > 2200):
#                 self.get_logger().info("current condition statified")
#                 break
            
#             time.sleep(0.025) 
#         self.step(id1,id2,id3)
#         self.step(id1,id2,id3)

#     def open(self):
#         self.get_logger().info("opening gripper")
#         id1,id2,id3=self.ids
#         self.angle1 = self.init_angle
#         self.angle2 = self.init_angle
#         self.angle3 = self.init_angle
#         self.kit.servo[id1].angle = self.angle1
#         self.kit.servo[id2].angle = self.angle2
#         self.kit.servo[id3].angle = self.angle3
#         self.get_logger().info("opening gripper completed")
#     def timer_callback(self):
#         msg=GripperAngle()
#         msg.angle1 = float(np.deg2rad(-self.angle1))
#         msg.angle2 = float(np.deg2rad(-self.angle2))
#         msg.angle3 = float(np.deg2rad(-self.angle3))
#         # self.get_logger().info(f"publishing {msg.angle1},{msg.angle2},{msg.angle3}")
#         self.angle_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    gripper_node = Gripper()
    try:
        rclpy.spin(gripper_node)
    finally:
        gripper_node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()