import time
import math
# if __name__=="__main__":
#     from HitbotInterface import HitbotInterface
# else:
#     from .HitbotInterface import HitbotInterface
from .HitbotInterface import HitbotInterface
import numpy as np

class ScaraRobot(HitbotInterface):
    """
    A wrapper class around HitbotInterface for controlling a SCARA robot.
    Provides convenience methods for initialization, homing, and updates.
    """

    def __init__(self, robot_id=92, port=210):
        super().__init__(robot_id)  # Properly call parent constructor

        # Initialize communication and connect
        self.net_port_initial()
        ret = self.initial(1, port)
        
        # Verify connection
        if not self.is_connect():
            raise ConnectionError("Failed to connect to SCARA robot.")

        # Unlock and retrieve parameters
        self.unlock_position()
        self.get_scara_param()

        print("Robot ready.")

        # Initialize internal state variables
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.angle3 = 0.0
        self.r = 0.0
        self.limit_degree=[[-90.0,90.0],[-135.0,135.0],[-135.0,135.0]]
        self.update()
    
    def within_limits(self,angle1,angle2,angle3):
        angles = [angle1, angle2, angle3]
        for i, angle in enumerate(angles):
            lower, upper = self.limit_degree[i]
            if not (lower <= angle <= upper):
                return False
        return True

    def go_home(self, speed=50, roughly=0):
        """
        Moves the robot to its home (zero) position.
        """
        self.unlock_position()
        
        ret = self.movej_angle(
            goal_angle1=0,
            goal_angle2=0,
            goal_z=0,
            goal_r=0,
            speed=speed,
            roughly=roughly
        )
        if ret == 1:
            print("Robot homed successfully.")
        else:
            print(f"Failed to move home. Error code: {ret}")
        self.wait_stop()
        return ret

    def within_limits_z(self,z):
        return True if z<=0 and z>=-170 else False

    def update(self):
        self.get_scara_param()
        self.angle3=self.r-self.angle1-self.angle2
    
    def safe_move_degrees(self,angle1,angle2,angle3,z=None,speed="auto"):
        if z is None:
            z=self.z

        self.update()
        if not self.within_limits(angle1,angle2,angle3):
            print("angles over the limit")
            return 0
        if not self.within_limits_z(z):
            print("Z out of limit")
            return 0
        # cost=abs(self.angle1-angle1)*5+abs(self.angle2-angle2)*1.5+abs(self.angle3-angle3)/10
        # if abs(self.angle1-angle1)<1 and abs(self.angle2-angle2)<1:
        #     cost=0
        # print("cost of movement", cost)
        # if speed=="auto":
        #     if cost<1 and z==self.z:
        #         print("already in position")
        #         return 1
        #     elif cost<100:
        #         speed=50
        #     elif cost<150:
        #         speed=100
        #     elif cost<200:
        #         speed=175
        #     else:
        #         speed=250
        
        # if not isinstance(speed, (int, float)):
        #     speed = 100
        # print(speed)
        if (abs(self.angle2-angle2)+ abs(self.angle1-angle1))<10:
            ret = self.movej_angle(
                        goal_angle1=angle1,
                        goal_angle2=angle2,
                        goal_z=z,
                        goal_r=angle1+angle2+angle3,
                        speed=35,
                        roughly=0)
        elif (abs(self.angle2-angle2)+abs(self.angle1-angle1))<20:
            ret = self.movej_angle(
                        goal_angle1=angle1,
                        goal_angle2=angle2,
                        goal_z=z,
                        goal_r=angle1+angle2+angle3,
                        speed=50,
                        roughly=0)
        elif (abs(self.angle2-angle2)+ abs(self.angle1-angle1))<30:
            ret = self.movej_angle(
                        goal_angle1=angle1,
                        goal_angle2=angle2,
                        goal_z=z,
                        goal_r=angle1+angle2+angle3,
                        speed=80,
                        roughly=0)
        else:
            ret = self.movej_angle(
                        goal_angle1=angle1,
                        goal_angle2=angle2,
                        goal_z=z,
                        goal_r=angle1+angle2+angle3,
                        speed=350,
                        roughly=0)
        self.wait_stop()
        
        self.update()
        
        return 1
        
    def move_joint_degree_step(self,theta1,theta2,theta3,speed=120, roughly=0):
        return self.safe_move_degrees(theta1,theta2,theta3)

 
    def move_joint_radian(self,theta1,theta2,theta3,speed=50, roughly=0):
        theta1=int(round(np.rad2deg(theta1),0))
        theta2=int(round(np.rad2deg(theta2),0))
        theta3=int(round(np.rad2deg(theta3),0))

        return self.safe_move_degrees(theta1,theta2,theta3)
    
    def move_z(self,z,speed=150, roughly=0):
        self.get_scara_param()
        if self.z>-170:
            ret = self.movej_angle(
                goal_angle1=self.angle1,
                goal_angle2=self.angle2,
                goal_z=z,
                goal_r=self.r,
                speed=200,
                roughly=roughly
            )
            self.wait_stop()
        else:
            print("z out of range")


    def safe_move_degrees2(self,angle1,angle2,angle3,z,speed):
       
        if not self.within_limits(angle1,angle2,angle3):
            print("angles over the limit")
            return 0
        if not self.within_limits_z(z):
            print("Z out of limit")
            return 0
        
        ret = self.movej_angle(
                        goal_angle1=angle1,
                        goal_angle2=angle2,
                        goal_z=z,
                        goal_r=angle1+angle2+angle3,
                        speed=speed,
                        roughly=0)
        
        
        return 1
    

    def move_joint_radian2(self,theta1,theta2,theta3,z=None,speed=50, roughly=0):
        theta1=int(round(np.rad2deg(theta1),0))
        theta2=int(round(np.rad2deg(theta2),0))
        theta3=int(round(np.rad2deg(theta3),0))
        self.update()
        if z is None:
            z=self.z
        return self.safe_move_degrees2(theta1,theta2,theta3,z=z,speed=speed)

if __name__=="__main__":
    scara=ScaraRobot()
    scara.go_home()
    val=scara.move_joint_degree_step(30,30,30)
    for i in val:
        print(i)
    #scara.move_joint_radian(0.1,0.1,0.1,70)
    #scara.move_z(-10)
