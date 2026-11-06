from qwiic_i2c.linux_i2c import LinuxI2C
from rclpy.node import Node
import rclpy
import qwiic_vl53l5cx 
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from math import sqrt
import numpy as np
import time
from std_msgs.msg import Float32 as RosFloat32
class TOF(Node):
    def __init__(self):
        super().__init__("TOF_sensor")
        self.driver = LinuxI2C(iBus=7)
        self.tof= qwiic_vl53l5cx.QwiicVL53L5CX(address=0x29,i2c_driver=self.driver ) 
        if self.tof.is_connected() == False:
            self.get_logger().error("The device isn't connected to the system. Please check your connection")
        self.get_logger().warn("Initializing sensor board. This can take up to 10s. Please wait.")
        if self.tof.begin() == False:
            self.get_logger().error("Sensor initialization unsuccessful. Exiting...")
        self.raw_tof_point_publisher= self.create_publisher(PointCloud2, "raw_tof_points", 20)

        self.timer_period = 0.5  
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.tof.set_resolution(8*8) # enable all 64 pads
        self.image_resolution = self.tof.get_resolution()  # Query sensor for current resolution - either 4x4 or 8x8
        self.pub=self.create_publisher(RosFloat32,"tof_data",10)

        self.image_width = int(sqrt(self.image_resolution)) # Calculate printing width
        self.tof.start_ranging()

    def timer_callback(self):
        data=None
        # self.tof.start_ranging()
        while data is None:
            if self.tof.check_data_ready():
                
                measurement_data=self.tof.get_ranging_data()
                data = measurement_data.distance_mm.copy()
                
                time.sleep(0.005)
        # self.tof.stop_ranging()
        data=np.array(data)
        data=data/(1_000*3.75)
        depth=np.flipud(np.fliplr(data.reshape(8,-1)))
        
        mn=np.mean(depth[2:8][3:5])
        
                
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "gripper_base"

        cloud_msg = pc2.create_cloud_xyz32(header, [[mn, 0, 0]])

        self.raw_tof_point_publisher.publish(cloud_msg)
        msg=RosFloat32()
        
        msg.data=mn
        self.pub.publish(msg)



# import rclpy

# import time
# import board
# import busio
# import adafruit_vl53l0x

# from std_msgs.msg import Float32 as RosFloat32
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import String as RosString

# class TOF(Node):
#     def __init__(self):
#         super().__init__("TOF_sensor")
#         i2c = busio.I2C(board.SCL, board.SDA)

#         self.sensor = adafruit_vl53l0x.VL53L0X(i2c, address=0x29)

#         self.pub=self.create_publisher(RosFloat32,"tof_data",10)
#         self.timmer_=self.create_timer(1,self.read_and_publish)
    
#     def read_and_publish(self):
#         msg=RosFloat32()
        
#         msg.data=self.sensor.range/1000
#         self.pub.publish(msg)

    
        
def main(args=None):
    rclpy.init(args=args)
    tof_node = TOF()
    rclpy.spin(tof_node)
    

if __name__ == "__main__":
    main()