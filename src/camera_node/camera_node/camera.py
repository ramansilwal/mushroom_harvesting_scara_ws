import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

import numpy as np
import cv2
from std_msgs.msg import Header
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


from example_interfaces.srv import Trigger
from ultralytics import YOLO
from .depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from custom_interfaces.msg import Mushroom
from custom_interfaces.srv import GetMushroom

from ament_index_python.packages import get_package_share_directory
import os

# Get the absolute path of the package
package_share = get_package_share_directory('camera_node')

# Construct the full path to the model weights
weights_path = os.path.join(package_share, 'depth_anything_v2', 'depth_anything_v2_vits.pth')

print(weights_path)

def load_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Define model configurations for different encoders
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vits'  # You can choose from 'vits', 'vitb', 'vitg', 'vitl'

    # Initialize DepthAnythingV2 model
    depth_model = DepthAnythingV2(**model_configs[encoder])
    
    # Define the path to the model weights (assuming the weight file is located here)
    weights_path = "/home/cotrobot/depth_yolo_ws/src/camera_node/camera_node/depth_anything_v2/depth_anything_v2_vits.pth"

    # Construct the full path to the model weights
    #weights_path = os.path.join(package_share, 'depth_anything_v2', 'depth_anything_v2_vits.pth')

    print(weights_path)
    # Load the model weights into the model and set the device
    torch.cuda.empty_cache()
    depth_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    return depth_model,DEVICE
    
class Camera(Node):
    def __init__(self):
        """Camera node, takes picture with realsense camera and pulishes it to topic"""
        super().__init__("camera_node")
        self.timer_period = 1  
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.publisher_color_image = self.create_publisher(ROSImage, "color_image", 10)
        self.publisher_depth_image = self.create_publisher(ROSImage, "depth_image", 10)
        self.pc2_publisher = self.create_publisher(PointCloud2, "pointcloud", 20)
        self.pc2_mushroom_points_publisher = self.create_publisher(PointCloud2, "mushroom_points", 20)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.bridge = CvBridge()
        self.color_image, self.depth_image,self.pts=None,None,None

        self.depth_model,self.device = load_model()

        # self.processed_image_publisher_ = self.create_publisher(ROSImage, "processed_image", 10)
        self.detected_img = self.create_publisher(ROSImage, "detected_img", 10)
        self.depth_yolo_model = YOLO("/home/cotrobot/mushroom_harvesting_scara_ws/resources/mushroom_depth_yolo.pt")
        self.processed_image_publisher_ = self.create_publisher(ROSImage, "processed_image", 10)
        self.create_service(Trigger, "get_depth_image", self.get_depth_image)
        self.create_service(GetMushroom, "get_mushroom_location", self.get_mushroom)
        self.filtered_pts_raw=[]

    def get_mushroom(self, request, response):
        self.process_frame()
        # print(len(self.filtered_pts_raw))
        if len(self.filtered_pts_raw)<=0:
            return response
        for i in self.filtered_pts_raw:
            mushroom=Mushroom()
            for j in i:
                mushroom.points_flat.extend(j)
            
            response.mushrooms.append(mushroom)
 
        return response

    def capture_frames(self):
        """ method that takes images"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None ,None

        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        pts = []
        pixel_pts = [] 
        
        for x in range(0, 480, 5):
            for y in range(0, 640, 5):
                depth_value = np.mean(depth_image[x, y]) * 0.0001

                point = rs.rs2_deproject_pixel_to_point(
                    self.color_intrin, [y, x], depth_value
                )
                if np.linalg.norm(np.array([point[0], point[1], -point[2]])) > 1:
                    continue
                pts.append([point[0], -point[1], -point[2]])
                pixel_pts.append((x, y)) 
        self.color_image, self.depth_image,self.pts=color_image, depth_image,pts
        self.pixel_pts=pixel_pts
        return color_image, depth_image,pts
        
    def publish_raw_images(self, color_image, depth_image):
        """method to publish images"""
        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
        self.publisher_color_image.publish(color_msg)
        self.publisher_depth_image.publish(depth_msg)

    def timer_callback(self):
        """callback method"""
        color_image, depth_image,pts = self.capture_frames()
        if color_image is None:
            return
        self.publish_raw_images(color_image, depth_image)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "gripper_base"

        cloud_msg = pc2.create_cloud_xyz32(header, pts)

        self.pc2_publisher.publish(cloud_msg)
    
    # def get_depth_image(self, request, response):
    #     if self.color_image is not None:
            
            
    #         # input_image = Image.fromarray(self.color_image)
    #         depth_img = self.depth_model.infer_image(self.color_image)
    #         depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
    #         depth_image_uint8 = np.uint8(depth_map_normalized)

    #         depth_image_bgr = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)
    #         ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image_bgr, encoding="bgr8")
    #         self.processed_image_publisher_.publish(ros_depth_image)
    #         response.success = True
    #         img=Image.fromarray(depth_image_uint8)
    #         self.get_logger().info("sending depth image into yolo")
    #         results=self.depth_yolo_model(img,conf=0.60)
    #         r=results[0].to("cpu")
    #         r = results[0].to("cpu")
    #         result_image = r.plot()
    #         result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    #         # Publish the YOLO result image as a ROS image
    #         ros_result_image = self.bridge.cv2_to_imgmsg(result_image_bgr, encoding="bgr8")
    #         self.detected_img.publish(ros_result_image)

            
    #     else:
    #         self.get_logger().error("No image received yet.")
    #         response.success = False

    #     return response
    def process_frame(self):
        
        torch.cuda.empty_cache()
        # ---- Depth estimation ----
        with torch.inference_mode():

            small = cv2.resize(self.color_image, (0,0), fx=0.5, fy=0.5) 

            depth_img = self.depth_model.infer_image(small)
            depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
            depth_image_uint8 = np.uint8(depth_map_normalized)

            depth_image_bgr = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)
            ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image_bgr, encoding="bgr8")
            self.processed_image_publisher_.publish(ros_depth_image)

            # ---- YOLO inference ----
            img = Image.fromarray(depth_image_uint8)
            self.get_logger().info("sending depth image into yolo")

            results = self.depth_yolo_model(img, conf=0.60)
            r = results[0].to("cpu")

        # ---- Publish detection image ----
        result_image = r.plot()
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        ros_result_image = self.bridge.cv2_to_imgmsg(result_image_bgr, encoding="bgr8")
        self.detected_img.publish(ros_result_image)

        filtered_pts = []

        # ---- Vectorized mask filtering ----
        # masks = r.masks

        # if masks is not None:
        #     mask_data = masks.data.cpu().numpy()  # (N, H, W)

        #     pts_array = np.array(self.pts)              # (M, 3)
        #     pixel_array = np.array(self.pixel_pts)      # (M, 2) -> (row, col)

        #     rows = pixel_array[:, 0]
        #     cols = pixel_array[:, 1]

        #     for mask in mask_data:
        #         # Resize mask to match original image size
        #         mask_resized = cv2.resize(
        #             mask,
        #             (self.color_image.shape[1], self.color_image.shape[0])
        #         )

        #         mask_resized = mask_resized > 0.5  # binarize

        #         # Vectorized indexing
        #         valid = mask_resized[rows, cols]

        #         filtered_pts.append(pts_array[valid])
        #         self.filtered_pts_raw=filtered_pts.copy()
        #     if filtered_pts:
        #         filtered_pts = np.vstack(filtered_pts)
        #     else:
        #         filtered_pts = np.empty((0, 3))

        # else:
        #     filtered_pts = np.empty((0, 3))
        
        # filtered_pts = filtered_pts[filtered_pts[:,-1] < -0.02] 

        # # ---- Publish filtered point cloud ----
        # header = Header()
        # header.stamp = self.get_clock().now().to_msg()
        # header.frame_id = "gripper_base"

        # cloud_msg = pc2.create_cloud_xyz32(header, filtered_pts.tolist())
        # self.pc2_mushroom_points_publisher.publish(cloud_msg)

        # self.filtered_pts = filtered_pts
        # return filtered_pts


        filtered_pts = []
        
        masks = r.masks

        if masks is not None:

            mask_data = masks.data.cpu().numpy()  # (N, H, W)

            pts_array = np.asarray(self.pts)          # (M, 3)
            pixel_array = np.asarray(self.pixel_pts)  # (M, 2)

            rows = pixel_array[:, 0].astype(np.int32)
            cols = pixel_array[:, 1].astype(np.int32)

            H, W = self.color_image.shape[:2]

            # ---- Remove invalid pixel indices ----
            valid_idx = (
                (rows >= 0) & (rows < H) &
                (cols >= 0) & (cols < W)
            )

            rows = rows[valid_idx]
            cols = cols[valid_idx]
            pts_array = pts_array[valid_idx]

            for mask in mask_data:

                # ---- Resize correctly for masks ----
                mask_resized = cv2.resize(
                    mask,
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                )

                # ---- Binary mask ----
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # ---- Morphological cleanup ----
                kernel = np.ones((5, 5), np.uint8)

                mask_binary = cv2.morphologyEx(
                    mask_binary,
                    cv2.MORPH_OPEN,
                    kernel
                )

                mask_binary = cv2.morphologyEx(
                    mask_binary,
                    cv2.MORPH_CLOSE,
                    kernel
                )

                # ---- Vectorized point filtering ----
                valid_mask = mask_binary[rows, cols]

                mask_pts = pts_array[valid_mask.astype(bool)]

                # ---- Remove depth outliers ----
                if len(mask_pts) > 0:

                    z = mask_pts[:, 2]

                    z_median = np.median(z)

                    # keep points close to median depth
                    depth_filter = np.abs(z - z_median) < 0.03

                    mask_pts = mask_pts[depth_filter]

                    filtered_pts.append(mask_pts)

            self.filtered_pts_raw = filtered_pts.copy()

            if filtered_pts:
                filtered_pts = np.vstack(filtered_pts)
            else:
                filtered_pts = np.empty((0, 3))

        else:
            filtered_pts = np.empty((0, 3))

        # ---- Final z filtering ----
        filtered_pts = filtered_pts[filtered_pts[:, -1] < -0.02]

        # ---- Publish cloud ----
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "gripper_base"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            filtered_pts.tolist()
        )

        self.pc2_mushroom_points_publisher.publish(cloud_msg)

        self.filtered_pts = filtered_pts

        return filtered_pts
    
    def get_depth_image(self, request, response):
        if self.color_image is None:
            self.get_logger().error("No image received yet.")
            response.success = False
            return response
        
        with torch.inference_mode():

    
            depth_img = self.depth_model.infer_image(self.color_image)
            depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
            depth_image_uint8 = np.uint8(depth_map_normalized)

            depth_image_bgr = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)
            ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image_bgr, encoding="bgr8")
            self.processed_image_publisher_.publish(ros_depth_image)


        
        response.success = True

        return response

    
    
        
def main(args=None):
    rclpy.init(args=args)
    camera_node = Camera()
    try:
        rclpy.spin(camera_node)
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
