import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from matplotlib import pyplot as plt
import numpy as np
import cv2
from std_msgs.msg import Header
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import time
from example_interfaces.srv import Trigger
from ultralytics import YOLO
from .depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from custom_interfaces.msg import Mushroom
from custom_interfaces.srv import GetMushroom

from ament_index_python.packages import get_package_share_directory
import os


import numpy as np

def scale_depth_patchwise(
    real_depth,
    pred_depth,
    patch_size=100,
    stride=25
):
    """
    Scale predicted depth using local overlapping patches
    and blend the patches using Hanning weights.

    Parameters
    ----------
    real_depth : np.ndarray
        Real depth image, e.g. D405 depth.

    pred_depth : np.ndarray
        Predicted depth image.

    patch_size : int
        Size of each local patch.

    stride : int
        Step between patches. Smaller stride = more overlap.

    Returns
    -------
    scaled_depth : np.ndarray
        Locally scaled and smoothly blended depth image.
    """

    real_depth = real_depth.astype(np.float32)
    pred_depth = pred_depth.astype(np.float32)

    h, w = pred_depth.shape

    scaled_sum = np.zeros_like(pred_depth, dtype=np.float32)
    weight_sum = np.zeros_like(pred_depth, dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):

            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)

            real_patch = real_depth[y:y2, x:x2]
            pred_patch = pred_depth[y:y2, x:x2]

            valid = (
                (real_patch > 1) &
                np.isfinite(real_patch) &
                np.isfinite(pred_patch)
            )

            if np.sum(valid) < 10:
                continue

            real_valid = real_patch[valid]
            pred_valid = pred_patch[valid]

            real_min = real_valid.min()
            real_max = real_valid.max()

            pred_min = pred_valid.min()
            pred_max = pred_valid.max()

            if pred_max - pred_min < 1e-6:
                continue

            # inverse linear scaling
            scaled_patch = (
                (pred_max - pred_patch)
                / (pred_max - pred_min)
                * (real_max - real_min)
                + real_min
            )

            # smooth blending weights
            ph, pw = pred_patch.shape

            wy = np.hanning(ph + 2)[1:-1]
            wx = np.hanning(pw + 2)[1:-1]

            weight = np.outer(wy, wx).astype(np.float32)

            scaled_sum[y:y2, x:x2] += scaled_patch * weight
            weight_sum[y:y2, x:x2] += weight

    scaled_depth = np.zeros_like(pred_depth, dtype=np.float32)

    valid_weight = weight_sum > 0

    scaled_depth[valid_weight] = (
        scaled_sum[valid_weight]
        / weight_sum[valid_weight]
    )

    return scaled_depth


# Get the absolute path of the package
package_share = get_package_share_directory('camera_node')

# Construct the full path to the model weights
weights_path = os.path.join(package_share, 'depth_anything_v2', 'depth_anything_v2_vits.pth')

print(weights_path)

def load_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vits' 

    
    depth_model = DepthAnythingV2(**model_configs[encoder])
    
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
        self.timer_period = 0.25  
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.pipeline_d405=None
        self.pipeline_d435i=None
        
        self.init_camera()
        self.img_publisher={}
        self.img_publisher["top_color"] = self.create_publisher(ROSImage, "color_image", 10)
        self.img_publisher["top_depth"] = self.create_publisher(ROSImage, "depth_image", 10)

        self.img_publisher["side_color"] = self.create_publisher(ROSImage, "side_color_image", 10)
        self.img_publisher["side_depth"] = self.create_publisher(ROSImage, "side_depth_image", 10)

        self.pc2_publisher = self.create_publisher(PointCloud2, "pointcloud", 20)
        self.pc2_mushroom_points_publisher = self.create_publisher(PointCloud2, "mushroom_points", 20)
        self.side_pc2_publisher = self.create_publisher(PointCloud2, "side_pointcloud", 20)

        self.side_pc2_mushroom_points_publisher = self.create_publisher(PointCloud2, "side_mushroom_points", 20)


        self.bridge = CvBridge()
        self.color_image_d405, self.depth_image_d405,self.pts_d405=None,None,None

        self.depth_model,self.device = load_model()

        test_img= cv2.imread("/home/cotrobot/Downloads/PXL_20260406_153437175.jpg")
        self.get_logger().info("***Checking depth model***")
        with torch.inference_mode():
            self.depth_model.infer_image(test_img,input_size=240)

        # self.processed_image_publisher_ = self.create_publisher(ROSImage, "processed_image", 10)
        self.detected_img = self.create_publisher(ROSImage, "detected_img", 10)
        self.detected_side_img = self.create_publisher(ROSImage, "detected_side_img", 10)
        
        
        self.depth_yolo_model = YOLO("/home/cotrobot/resources/side_camera.engine",task="segment")
        # self.depth_yolo_model(np.zeros((320,480,3)))
        self.processed_image_publisher_ = self.create_publisher(ROSImage, "processed_image", 10)
        self.create_service(Trigger, "get_depth_image", self.get_depth_image)
        self.create_service(GetMushroom, "get_mushroom_location", self.get_mushroom)



        self.processed_side_image_publisher_ = self.create_publisher(ROSImage, "processed_side_image", 10)

        self.create_service(Trigger, "get_side_depth_image", self.get_side_depth_image)
        self.create_service(GetMushroom, "get_side_mushroom_location", self.get_side_mushroom)

        self.filtered_pts_raw=[]
        self.get_logger().info("***Camera node is alive and ready***")
    
    
    def init_camera(self):

        ctx = rs.context()
        serials={}
        # Find all connected devices
        for device in ctx.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)

            print(f"Found: {name} ({serial})")
            serials[name.split(" ")[-1]]=serial
        
        self.config_d405=None
        self.config_d435i=None
        if "D405" in serials:
            self.pipeline_d405 = rs.pipeline()
            self.pipeline_d405
            self.align = rs.align(rs.stream.color)

            # Configure D405
            self.config_d405 = rs.config()
            self.config_d405.enable_device(serials["D405"])
            self.config_d405.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.config_d405.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.pipeline_d405.start(self.config_d405)

        if "D435I" in serials:
            self.pipeline_d435i = rs.pipeline()

            # Configure D405
            self.config_d435i = rs.config()
            self.config_d435i.enable_device(serials["D435I"])
            self.config_d435i.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.config_d435i.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            self.pipeline_d435i.start(self.config_d435i)
            
    def get_side_mushroom(self, request, response):
        self.process_side_frame()
        print(len(self.side_filtered_pts_raw))
        if len(self.side_filtered_pts_raw)<=0:
            return response
        for i in self.side_filtered_pts_raw:
            mushroom=Mushroom()
            for j in i:
                mushroom.points_flat.extend(j)
            
            response.mushrooms.append(mushroom)
 
        return response

    def get_mushroom(self, request, response):
        self.process_frame()
        if len(self.filtered_pts_raw)<=0:
            return response
        for i,mask in zip(self.filtered_pts_raw,self.filtered_masks):
            mushroom=Mushroom()
            for j in i:
                mushroom.points_flat.extend(j)
            mushroom.mask=mask
            response.mushrooms.append(mushroom)
 
        return response

    def capture_frames(self):
        """ method that takes images"""

        if self.config_d405 is not None:
            frames_d405 = self.pipeline_d405.wait_for_frames()

            # aligned_frames_d405 = self.align.process(frames_d405)
            depth_frame_d405 = frames_d405.get_depth_frame()
            color_frame_d405 = frames_d405.get_color_frame()

            self.color_intrin_d405 = color_frame_d405.profile.as_video_stream_profile().intrinsics
            img_color_d405 = np.asanyarray(color_frame_d405.get_data())
            img_depth_d405 = np.asanyarray(depth_frame_d405.get_data())
            height, width,_ = img_color_d405.shape
            self.d405_pts_x = np.zeros((height, width))
            self.d405_pts_y = np.zeros((height, width))
            self.d405_pts_z = np.zeros((height, width))

            increment=5
            points=[]
            for x in range(0, height, increment):
                for y in range(0, width, increment):
                    depth_value = img_depth_d405[x, y] * 0.0001

                    point = rs.rs2_deproject_pixel_to_point(
                        self.color_intrin_d405, [y, x], depth_value
                    )
                    self.d405_pts_x[x:x+increment,y:y+increment]=point[0]
                    self.d405_pts_y[x:x+increment,y:y+increment]=-point[1]
                    self.d405_pts_z[x:x+increment,y:y+increment]=-point[2]

                    points.append(tuple([point[0],-point[1],-point[2]]))

            set_pts=np.array([i for i in set(points)])
            set_pts=set_pts[set_pts[:,2]<-0.001]
            pts=set_pts.reshape(-1,3).tolist()

            self.color_image_d405, self.depth_image_d405,self.pts_d405=img_color_d405, img_depth_d405,pts
            
        if self.config_d435i is not None:
            frames_d435i = self.pipeline_d435i.wait_for_frames()
    
            aligned_frames_d435i = self.align.process(frames_d435i)
            depth_frame_d435i = aligned_frames_d435i.get_depth_frame()
            color_frame_d435i = aligned_frames_d435i.get_color_frame()

            self.color_intrin_d435i = color_frame_d435i.profile.as_video_stream_profile().intrinsics
            img_color_d435i = np.asanyarray(color_frame_d435i.get_data())
            img_depth_d435i = np.asanyarray(depth_frame_d435i.get_data())
            height, width,_ = img_color_d435i.shape
            self.d435i_pts_x = np.zeros((height, width))
            self.d435i_pts_y = np.zeros((height, width))
            self.d435i_pts_z = np.zeros((height, width))
            increment=5
            points=[]
            for x in range(0, height, increment):
                for y in range(0, width, increment):
                    depth_value = img_depth_d435i[x, y] * 0.001

                    point = rs.rs2_deproject_pixel_to_point(
                        self.color_intrin_d435i, [y, x], depth_value
                    )
                
                    self.d435i_pts_x[x:x+increment,y:y+increment]=point[0]
                    self.d435i_pts_y[x:x+increment,y:y+increment]=-point[1]
                    self.d435i_pts_z[x:x+increment,y:y+increment]=-point[2]

                    points.append(tuple([point[0],-point[1],-point[2]]))

            
            set_pts=np.array([i for i in set(points)])
            set_pts=set_pts[set_pts[:,2]<-0.001]
            pts=set_pts.reshape(-1,3).tolist()

            self.color_image_d435i, self.depth_image_d435i,self.pts_d435i=img_color_d435i, img_depth_d435i,pts



        
    def publish_raw_images(self, image,topic_name,encoding=None):
        """method to publish images"""

        if "color" in topic_name:
            msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.img_publisher[topic_name].publish(msg)
        elif "depth" in topic_name:
            if encoding is None:
                encoding="8UC3"
            msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
            self.img_publisher[topic_name].publish(msg)
    

    def publish_raw_points(self, points, publisher,frame_id):
        # Convert NumPy array to a Python list if needed
        if isinstance(points, np.ndarray):
            points = points.tolist()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id =frame_id

        cloud_msg = pc2.create_cloud_xyz32(header, points)
        publisher.publish(cloud_msg)

    def timer_callback(self):
        """callback method"""
        self.capture_frames()
        if self.color_image_d405 is None:
            return
        self.publish_raw_images(self.color_image_d405,"top_color")
        self.publish_raw_images(self.depth_image_d405,"top_depth",encoding="16UC1")
        self.publish_raw_points(self.pts_d405, self.pc2_publisher, "gripper_base")

        if self.color_image_d435i is None:
            return
        self.publish_raw_images(self.color_image_d435i,"side_color")
        self.publish_raw_images(self.color_image_d435i,"side_depth")
        ts=[self.transform_d435i(x,y,z) for x,y,z in self.pts_d435i]

        self.publish_raw_points(ts, self.side_pc2_publisher, "Link1")
    

    def transform_d435i(self,x,y,z):
        return [-z-0.025,-x+0.05,y-0.05]
    

    def process_side_frame(self):
                
                    
        crop1 = self.color_image_d435i[250:, :350]

        crop2 = self.color_image_d435i[250:, 300:]
        crops=[]
        results=[]
        with torch.inference_mode():
            for img in [crop1,crop2]:
                depth_img = self.depth_model.infer_image(img,input_size=240)

                depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
                depth_image_uint8 = np.uint8(depth_map_normalized)
                depth_image_bgr = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)

                r=self.depth_yolo_model(depth_image_bgr,conf=0.35)
                results.append(r[0])
                # crops.append(depth_image_bgr)
        # with torch.inference_mode():
        #     results = self.depth_yolo_model(crops,conf=0.35)

        r0=results[0]
        result_image0 = r0.plot()
        result_image_bgr0 = cv2.cvtColor(result_image0, cv2.COLOR_RGB2BGR)

        r1=results[1]
        result_image1 = r1.plot()
        result_image_bgr1 = cv2.cvtColor(result_image1, cv2.COLOR_RGB2BGR)





        offsets = [(0, 250),(300, 250)]


        all_masks = []
        binary_images = []
        H, W = self.color_image_d435i.shape[:2]
        for i, r in enumerate(results):
            r = r.cpu()

            if r.masks is None:
                continue

            for mask in r.masks.xy:
                # Apply offsets
                polygon = np.array(
                    [[x + offsets[i][0], y + offsets[i][1]] for x, y in mask],
                    dtype=np.int32
                )

                # all_masks.append(polygon.tolist())

                # Create binary mask
                binary_image = np.zeros((H,W), dtype=np.uint8)
                cv2.fillPoly(binary_image, [polygon], 1)

                binary_images.append(binary_image)
                mask = binary_image.astype(bool)
                all_masks.append(mask)
        filtered_pts=[]
        for mask in all_masks:
            filtered_pts.append([tuple(self.transform_d435i(x,y,z)) for x,y,z in zip(self.d435i_pts_x[mask],self.d435i_pts_y[mask],self.d435i_pts_z[mask]) if abs(z)>0.001 ])

        set_pts=[np.array([i for i in set(items)]) for items in filtered_pts]
        all_mushroom_pts=[]
        filtered_pts=[]
        for i in set_pts:
            all_mushroom_pts.extend(i)
            filtered_pts.append(i.tolist())

        all_mushroom_pts=np.array(all_mushroom_pts)

        self.side_filtered_pts_raw = filtered_pts.copy()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "Link1"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            all_mushroom_pts.reshape(-1,3).tolist(),
        )

        self.side_pc2_mushroom_points_publisher.publish(cloud_msg)


        
        return filtered_pts
    def process_frame(self):
        self.capture_frames()

        # -------------------------
        # Depth estimation
        # -------------------------
        with torch.inference_mode():
            depth_img = self.depth_model.infer_image(
                self.color_image_d405,
                input_size=240
            )

        # Normalize depth
        depth_min = depth_img.min()
        depth_max = depth_img.max()

        depth_range = depth_max - depth_min

        if depth_range > 0:
            depth_uint8 = (
                (depth_img - depth_min) * (255.0 / depth_range)
            ).astype(np.uint8)
        else:
            depth_uint8 = np.zeros_like(depth_img, dtype=np.uint8)

        depth_bgr = cv2.cvtColor(
            depth_uint8,
            cv2.COLOR_GRAY2BGR
        )

        # -------------------------
        # YOLO inference
        # -------------------------
        s_time = time.perf_counter()

        with torch.inference_mode():
            r = self.depth_yolo_model(
                depth_bgr,
                verbose=False
            )[0]

        self.get_logger().info(
            f"YOLO runtime: {time.perf_counter() - s_time:.4f}s"
        )

        # -------------------------
        # Publish detection image
        # -------------------------
        result_image = r.plot()

        ros_result_image = self.bridge.cv2_to_imgmsg(
            result_image,
            encoding="bgr8"
        )

        self.detected_img.publish(ros_result_image)

        # -------------------------
        # Check masks
        # -------------------------
        if r.masks is None:
            self.filtered_pts_raw = []
            self.filtered_masks = []
            return []

        height, width = self.color_image_d405.shape[:2]

        masks = r.masks.data.cpu().numpy()

        # -------------------------
        # Precompute point cloud
        # -------------------------
        points = np.stack(
            (
                self.d405_pts_x - 0.01,
                self.d405_pts_y,
                self.d405_pts_z
            ),
            axis=-1
        )

        valid_depth = np.abs(self.d405_pts_z) > 0.001

        filtered_pts = []
        self.filtered_masks = []

        # -------------------------
        # Process each mushroom
        # -------------------------
        for mask in masks:

            mask_uint8 = mask.astype(np.uint8)

            # Full resolution mask
            mask_full = cv2.resize(
                mask_uint8,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # Gripper mask
            mask_small = cv2.resize(
                mask_uint8,
                (320, 256),
                interpolation=cv2.INTER_NEAREST
            )

            self.filtered_masks.append(
                mask_small.ravel().tolist()
            )

            # Extract valid 3D points
            selected = points[mask_full & valid_depth]

            # Remove duplicate points
            selected = np.unique(selected, axis=0)

            filtered_pts.append(selected)

        # -------------------------
        # Combine point clouds
        # -------------------------
        if filtered_pts:
            all_mushroom_pts = np.concatenate(
                filtered_pts,
                axis=0
            ).astype(np.float32)
        else:
            all_mushroom_pts = np.empty(
                (0, 3),
                dtype=np.float32
            )

        # Preserve original output
        self.filtered_pts_raw = [
            pts.tolist() for pts in filtered_pts
        ]

        # -------------------------
        # Publish ROS point cloud
        # -------------------------
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "gripper_base"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            all_mushroom_pts.tolist()
        )

        self.pc2_mushroom_points_publisher.publish(
            cloud_msg
        )

        return self.filtered_pts_raw
    # def process_frame(self):
    #     self.capture_frames()
    #     with torch.inference_mode():
    #         depth_img = self.depth_model.infer_image(self.color_image_d405,input_size=240)
    #     depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
    #     depth_image_uint8 = np.uint8(depth_map_normalized)
    #     depth_image_bgr = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)
    #     s_time=time.time()
    #     r=self.depth_yolo_model(depth_image_bgr)[0]
    #     self.get_logger().info(f"run time {time.time()-s_time}")
    #     result_image = r.plot()
    #     result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    #     ros_result_image = self.bridge.cv2_to_imgmsg(result_image_bgr, encoding="bgr8")
    #     self.detected_img.publish(ros_result_image)

    #     height,width,_ = self.color_image_d405.shape
    #     if r.masks is None:
    #         print("nothing")
    #         self.filtered_pts_raw=[]
    #         return []
    #     masks = r.masks.data.cpu().numpy()
    #     masks.shape
    #     filtered_pts=[]
    #     self.filtered_masks=[]
    #     for mask in masks:
    #         mask_resized = cv2.resize(
    #                                 mask.astype(np.uint8),
    #                                 (width, height),  # OpenCV expects (width, height)
    #                                 interpolation=cv2.INTER_NEAREST
    #                         ).astype(bool)
            
    #         mask_resized2 = cv2.resize(
    #                                 mask.astype(np.uint8),
    #                                 (320,256),  # OpenCV expects (width, height)
    #                                 interpolation=cv2.INTER_NEAREST
    #                         ).astype(bool)
            
    #         self.filtered_masks.append([1 if i else 0 for i in mask_resized2.flatten()])
    #         filtered_pts.append([(x-0.01,y,z) for x,y,z in zip(self.d405_pts_x[mask_resized],self.d405_pts_y[mask_resized],self.d405_pts_z[mask_resized]) if abs(z)>0.001 ])

    #     set_pts=[np.array([i for i in set(items)]) for items in filtered_pts]
    #     all_mushroom_pts=[]
    #     filtered_pts=[]
    #     for i in set_pts:
    #         all_mushroom_pts.extend(i)
    #         filtered_pts.append(i.tolist())

    #     all_mushroom_pts=np.array(all_mushroom_pts)
        
    #     self.filtered_pts_raw = filtered_pts.copy()

    #     header = Header()
    #     header.stamp = self.get_clock().now().to_msg()
    #     header.frame_id = "gripper_base"

    #     cloud_msg = pc2.create_cloud_xyz32(
    #         header,
    #         all_mushroom_pts.reshape(-1,3).tolist(),
    #     )

    #     self.pc2_mushroom_points_publisher.publish(cloud_msg)

    #     return filtered_pts
    
    def get_side_depth_image(self, request, response):
        if self.color_image_d435i is None:
            self.get_logger().error("No image received yet.")
            response.success = False
            return response
        
        with torch.inference_mode():

    
            depth_img = self.depth_model.infer_image(self.color_image_d435i)
            depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
            depth_image_uint8 = np.uint8(depth_map_normalized)

            depth_image_bgr = cv2.cvtColor(depth_image_uint8, cv2.COLOR_GRAY2BGR)
            ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image_bgr, encoding="bgr8")
            self.processed_side_image_publisher_.publish(ros_depth_image)


        
        response.success = True

        return response

    

    def get_depth_image(self, request, response):
        if self.color_image_d405 is None:
            self.get_logger().error("No image received yet.")
            response.success = False
            return response
        
        with torch.inference_mode():

    
            depth_img = self.depth_model.infer_image(self.color_image_d405)
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