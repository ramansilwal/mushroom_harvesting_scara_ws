"""ROS2 RealSense + direct RGB TensorRT YOLO segmentation camera node.

Replace the original camera node module with this file after checking the
configuration paths and verifying geometry against the original system.
"""
import time
from threading import Lock

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
from cv_bridge import CvBridge
from example_interfaces.srv import Trigger
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from ultralytics import YOLO

from custom_interfaces.msg import Mushroom
from custom_interfaces.srv import GetMushroom

# ------------ Adjustable performance settings ------------
YOLO_IMAGE_SIZE = 480  # Match the TensorRT engine export size
SIDE_CONFIDENCE = 0.35
TOP_CONFIDENCE = 0.5
POINT_STRIDE = 5
CAMERA_FPS = 15
TIMER_PERIOD = 0.25
PUBLISH_RAW_CLOUDS = False
PUBLISH_RAW_IMAGES = True
PUBLISH_DETECTIONS = True
PROFILE_GPU = True

# Single TensorRT segmentation model shared by both cameras.
ENGINE_PATH = '/home/cotrobot/mushroom_harvesting_scara_ws/resources/latest.engine'


class Camera(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.lock = Lock()
        self.pipeline_d405 = None
        self.pipeline_d435i = None
        self.align_d405 = rs.align(rs.stream.color)
        self.align_d435i = rs.align(rs.stream.color)
        self.cameras = {}
        self.frames = {}
        self.filtered_pts_raw = []
        self.side_filtered_pts_raw = []
        self.filtered_masks = []

        self.img_publisher = {
            'top_color': self.create_publisher(ROSImage, 'color_image', 10),
            'top_depth': self.create_publisher(ROSImage, 'depth_image', 10),
            'side_color': self.create_publisher(ROSImage, 'side_color_image', 10),
            'side_depth': self.create_publisher(ROSImage, 'side_depth_image', 10),
        }
        self.pc2_publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)
        self.side_pc2_publisher = self.create_publisher(PointCloud2, 'side_pointcloud', 10)
        self.pc2_mushroom_points_publisher = self.create_publisher(PointCloud2, 'mushroom_points', 10)
        self.side_pc2_mushroom_points_publisher = self.create_publisher(PointCloud2, 'side_mushroom_points', 10)
        self.detected_img = self.create_publisher(ROSImage, 'detected_img', 10)
        self.detected_side_img = self.create_publisher(ROSImage, 'detected_side_img', 10)
        self.processed_image_publisher_ = self.create_publisher(ROSImage, 'processed_image', 10)
        self.processed_side_image_publisher_ = self.create_publisher(ROSImage, 'processed_side_image', 10)

        self.init_camera()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            raise RuntimeError('TensorRT .engine inference requires an NVIDIA CUDA GPU')
        self.yolo_model = YOLO(ENGINE_PATH, task='segment')
        self._warm_up()

        self.create_service(Trigger, 'get_depth_image', self.get_depth_image)
        self.create_service(Trigger, 'get_side_depth_image', self.get_side_depth_image)
        self.create_service(GetMushroom, 'get_mushroom_location', self.get_mushroom)
        self.create_service(GetMushroom, 'get_side_mushroom_location', self.get_side_mushroom)
        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)
        self.get_logger().info('Optimized camera node ready')

    def _warm_up(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            with torch.inference_mode():
                self.yolo_model.predict(dummy, imgsz=YOLO_IMAGE_SIZE, device=0, verbose=False)
        except Exception as exc:
            self.get_logger().warning(f'Model warm-up failed: {exc}')

    def init_camera(self):
        found = {}
        for dev in rs.context().query_devices():
            name = dev.get_info(rs.camera_info.name).upper()
            serial = dev.get_info(rs.camera_info.serial_number)
            if 'D405' in name:
                found['top'] = serial
            elif 'D435' in name:
                found['side'] = serial
        for key in ('top', 'side'):
            if key not in found:
                self.get_logger().warning(f'{key} camera not detected')
                continue
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(found[key])
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, CAMERA_FPS)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, CAMERA_FPS)
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            scale = depth_sensor.get_depth_scale()
            self.cameras[key] = {
                'pipeline': pipeline, 'scale': scale,
                'align': self.align_d405 if key == 'top' else self.align_d435i,
            }
            if key == 'top':
                self.pipeline_d405 = pipeline
            else:
                self.pipeline_d435i = pipeline
            self.get_logger().info(f'{key} depth scale: {scale:.8f} m/unit')

    def _sync(self):
        if PROFILE_GPU and self.device == 'cuda':
            torch.cuda.synchronize()

    def _update_camera(self, key, blocking=False):
        """Capture aligned color/depth without building XYZ until needed."""
        if key not in self.cameras:
            return False
        cam = self.cameras[key]
        frames = cam['pipeline'].wait_for_frames(timeout_ms=2000) if blocking else cam['pipeline'].poll_for_frames()
        if not frames:
            return False
        aligned = cam['align'].process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return False
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        self.frames[key] = (color, depth, intr, cam['scale'])
        if key == 'top':
            self.color_image_d405 = color
            self.depth_image_d405 = depth
        else:
            self.color_image_d435i = color
            self.depth_image_d435i = depth
        return True

    def capture_frames(self):
        """Nonblocking updates for both cameras; preserves latest snapshots."""
        for key in self.cameras:
            self._update_camera(key, blocking=False)

    def _snapshot(self, key):
        # Prefer a newly acquired frame rather than reusing an old cached frame.
        if not self._update_camera(key, blocking=True) and key not in self.frames:
            raise RuntimeError(f'No frames from {key} camera')
        return self.frames[key]

    @staticmethod
    def _xyz_at_pixels(depth, intr, depth_scale, ys, xs):
        """Vectorized deprojection using aligned depth intrinsics."""
        z = depth[ys, xs].astype(np.float32) * np.float32(depth_scale)
        x = (xs.astype(np.float32) - intr.ppx) * z / intr.fx
        y = (ys.astype(np.float32) - intr.ppy) * z / intr.fy
        return x, -y, -z  # Preserve original camera sign convention

    @staticmethod
    def _sampled_xyz(depth, intr, scale, stride=POINT_STRIDE):
        """Generate sparse points directly, without filling repeated 5x5 blocks."""
        h, w = depth.shape
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        x, y, z = Camera._xyz_at_pixels(depth, intr, scale, ys, xs)
        valid = np.isfinite(z) & (z < -0.001)
        return np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)

    @staticmethod
    def _side_transform(points):
        """Exact vectorization of transform_d435i: [-z-.025, -x+.05, y-.05]."""
        out = np.empty_like(points, dtype=np.float32)
        out[:, 0] = -points[:, 2] - 0.025
        out[:, 1] = -points[:, 0] + 0.05
        out[:, 2] = points[:, 1] - 0.05
        return out

    def transform_d435i(self, x, y, z):
        return [-z - 0.025, -x + 0.05, y - 0.05]

    def _publish_image(self, image, publisher, encoding):
        publisher.publish(self.bridge.cv2_to_imgmsg(image, encoding=encoding))

    def _publish_cloud(self, points, publisher, frame_id):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        publisher.publish(pc2.create_cloud_xyz32(header, points.tolist()))

    def timer_callback(self):
        # Do NOT construct point clouds for every timer tick unless enabled.
        self.capture_frames()
        if PUBLISH_RAW_IMAGES:
            if 'top' in self.frames:
                color, depth, _, _ = self.frames['top']
                self._publish_image(color, self.img_publisher['top_color'], 'bgr8')
                self._publish_image(depth, self.img_publisher['top_depth'], '16UC1')
            if 'side' in self.frames:
                color, depth, _, _ = self.frames['side']
                self._publish_image(color, self.img_publisher['side_color'], 'bgr8')
                self._publish_image(depth, self.img_publisher['side_depth'], '16UC1')
        if PUBLISH_RAW_CLOUDS:
            for key, pub, frame in (
                ('top', self.pc2_publisher, 'gripper_base'),
                ('side', self.side_pc2_publisher, 'Link1'),
            ):
                if key not in self.frames:
                    continue
                _, depth, intr, scale = self.frames[key]
                pts = self._sampled_xyz(depth, intr, scale)
                if key == 'side':
                    pts = self._side_transform(pts)
                self._publish_cloud(pts, pub, frame)

    def _mask_points(self, mask, depth, intr, scale, origin_x=0, origin_y=0, side=False):
        """Only deproject pixels inside masks, at a configurable stride."""
        ys, xs = np.nonzero(mask[::POINT_STRIDE, ::POINT_STRIDE])
        if not len(xs):
            return np.empty((0, 3), dtype=np.float32)
        ys = ys * POINT_STRIDE + origin_y
        xs = xs * POINT_STRIDE + origin_x
        x, y, z = self._xyz_at_pixels(depth, intr, scale, ys, xs)
        valid = np.isfinite(z) & (np.abs(z) > 0.001)
        points = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)
        if side:
            return self._side_transform(points)
        points[:, 0] -= 0.01
        return points

    def process_side_frame(self):
        self._sync()
        t0 = time.perf_counter()
        color, depth, intr, scale = self._snapshot('side')
        h, w = color.shape[:2]
        crops = [(0, 250, min(350, w), h), (300, 250, w, h)]
        crop_images = [np.ascontiguousarray(color[y0:y1, x0:x1])
                       for x0, y0, x1, y1 in crops]
        self._sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            # Sequential inference supports fixed batch=1 TensorRT engines.
            results = [self.yolo_model.predict(img, imgsz=YOLO_IMAGE_SIZE,
                                                    conf=SIDE_CONFIDENCE, device=0,
                                                    verbose=False)[0]
                       for img in crop_images]
        self._sync()
        t2 = time.perf_counter()
        point_groups = []
        for result, (x0, y0, x1, y1) in zip(results, crops):
            if PUBLISH_DETECTIONS:
                self._publish_image(result.plot(), self.detected_side_img, 'bgr8')
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask_crop = cv2.resize(mask.astype(np.uint8), (x1-x0, y1-y0),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                point_groups.append(self._mask_points(mask_crop, depth, intr, scale,
                                                      origin_x=x0, origin_y=y0, side=True))
        t3 = time.perf_counter()
        self.side_filtered_pts_raw = [p.tolist() for p in point_groups]
        all_points = np.concatenate(point_groups) if point_groups else np.empty((0, 3), np.float32)
        self._publish_cloud(all_points, self.side_pc2_mushroom_points_publisher, 'Link1')
        t4 = time.perf_counter()
        self.get_logger().info(f'[SIDE] capture/crop={t1-t0:.3f}s YOLO={t2-t1:.3f}s '
                               f'points={t3-t2:.3f}s publish={t4-t3:.3f}s total={t4-t0:.3f}s')
        return self.side_filtered_pts_raw

    def process_frame(self):
        self._sync()
        t0 = time.perf_counter()
        color, depth, intr, scale = self._snapshot('top')
        h, w = color.shape[:2]
        # Use the original full-resolution BGR image directly.
        self._sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            result = self.yolo_model.predict(color, imgsz=YOLO_IMAGE_SIZE,
                                                 conf=TOP_CONFIDENCE, device=0,
                                                 verbose=False)[0]
        self._sync()
        t2 = time.perf_counter()

        
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                self.get_logger().info(
                    f"Mask {i}: shape={tuple(mask.shape)}, "
                    f"area={int(mask.sum().item())} pixels"
                )
        if PUBLISH_DETECTIONS:
            self._publish_image(result.plot(), self.detected_img, 'bgr8')
        point_groups = []
        masks_320 = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask_u8 = mask.astype(np.uint8)
                full = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                small = cv2.resize(mask_u8, (320, 256), interpolation=cv2.INTER_NEAREST)
                masks_320.append(small.ravel().tolist())
                point_groups.append(self._mask_points(full, depth, intr, scale))
        t3 = time.perf_counter()
        self.filtered_masks = masks_320
        self.filtered_pts_raw = [p.tolist() for p in point_groups]
        all_points = np.concatenate(point_groups) if point_groups else np.empty((0, 3), np.float32)
        self._publish_cloud(all_points, self.pc2_mushroom_points_publisher, 'gripper_base')
        t4 = time.perf_counter()
        self.get_logger().info(f'[TOP] capture={t1-t0:.3f}s YOLO={t2-t1:.3f}s '
                               f'points={t3-t2:.3f}s publish={t4-t3:.3f}s total={t4-t0:.3f}s')
        return self.filtered_pts_raw

    def get_mushroom(self, request, response):
        try:
            with self.lock:
                self.process_frame()
                for points, mask in zip(self.filtered_pts_raw, self.filtered_masks):
                    mushroom = Mushroom()
                    mushroom.points_flat = np.asarray(points, np.float32).ravel().tolist()
                    mushroom.mask = mask
                    response.mushrooms.append(mushroom)
        except Exception as exc:
            self.get_logger().error(f'Top detection failed: {exc}')
        return response

    def get_side_mushroom(self, request, response):
        try:
            with self.lock:
                self.process_side_frame()
                for points in self.side_filtered_pts_raw:
                    mushroom = Mushroom()
                    mushroom.points_flat = np.asarray(points, np.float32).ravel().tolist()
                    response.mushrooms.append(mushroom)
        except Exception as exc:
            self.get_logger().error(f'Side detection failed: {exc}')
        return response

    def get_depth_image(self, request, response):
        """Legacy service name: publish the top camera's original RGB image."""
        try:
            color, _, _, _ = self._snapshot('top')
            self._publish_image(color, self.processed_image_publisher_, 'bgr8')
            response.success = True
        except Exception as exc:
            self.get_logger().error(str(exc))
            response.success = False
        return response

    def get_side_depth_image(self, request, response):
        """Legacy service name: publish the side camera's original RGB image."""
        try:
            color, _, _, _ = self._snapshot('side')
            self._publish_image(color, self.processed_side_image_publisher_, 'bgr8')
            response.success = True
        except Exception as exc:
            self.get_logger().error(str(exc))
            response.success = False
        return response

    def destroy_node(self):
        for cam in self.cameras.values():
            try:
                cam['pipeline'].stop()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Camera()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()