class Camera(Node):
    def __init__(self):
        """Camera node, takes picture with realsense camera and pulishes it to topic"""
        super().__init__("camera_node")
        self.timer_period = 0.5  
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
        small = cv2.resize(test_img, (0,0), fx=0.01, fy=0.01) 
        self.depth_model.infer_image(small)

        # self.processed_image_publisher_ = self.create_publisher(ROSImage, "processed_image", 10)
        self.detected_img = self.create_publisher(ROSImage, "detected_img", 10)
        self.detected_side_img = self.create_publisher(ROSImage, "detected_side_img", 10)
        
        
        self.depth_yolo_model = YOLO("/home/cotrobot/resources/side_camera.pt")
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
            self.config_d405.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config_d405.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline_d405.start(self.config_d405)

        if "D435I" in serials:
            self.pipeline_d435i = rs.pipeline()

            # Configure D405
            self.config_d435i = rs.config()
            self.config_d435i.enable_device(serials["D435I"])
            self.config_d435i.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config_d435i.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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
        print(len(self.filtered_pts_raw))
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

            pts = []
            pixel_pts = [] 
            
            for x in range(0, 480, 5):
                for y in range(0, 640, 5):
                    depth_value = np.mean(img_depth_d405[x, y]) * 0.0001

                    point = rs.rs2_deproject_pixel_to_point(
                        self.color_intrin_d405, [y, x], depth_value
                    )
                    if np.linalg.norm(np.array([point[0], point[1], -point[2]])) > 1:
                        continue
                    pts.append([point[0], -point[1], -point[2]])
                    pixel_pts.append((x, y)) 
            self.color_image_d405, self.depth_image_d405,self.pts_d405=img_color_d405, img_depth_d405,pts
            self.pixel_pts_d405=pixel_pts
            
        if self.config_d435i is not None:
            frames_d435i = self.pipeline_d435i.wait_for_frames()
    
            aligned_frames_d435i = self.align.process(frames_d435i)
            depth_frame_d435i = aligned_frames_d435i.get_depth_frame()
            color_frame_d435i = aligned_frames_d435i.get_color_frame()

            self.color_intrin_d435i = color_frame_d435i.profile.as_video_stream_profile().intrinsics
            img_color_d435i = np.asanyarray(color_frame_d435i.get_data())
            img_depth_d435i = np.asanyarray(depth_frame_d435i.get_data())

            pts = []
            pixel_pts = [] 
            
            for x in range(0, 480, 5):
                for y in range(0, 640, 5):
                    depth_value = np.mean(img_depth_d435i[x, y]) * 0.001

                    point = rs.rs2_deproject_pixel_to_point(
                        self.color_intrin_d435i, [y, x], depth_value
                    )
                    if np.linalg.norm(np.array([point[0], point[1], -point[2]])) > 1:
                        continue
                    pts.append([point[0], -point[1], -point[2]])
                    pixel_pts.append((x, y)) 
            self.color_image_d435i, self.depth_image_d435i,self.pts_d435i=img_color_d435i, img_depth_d435i,pts
            self.pixel_pts_d435i=pixel_pts

        
    def publish_raw_images(self, image,topic_name):
        """method to publish images"""

        if "color" in topic_name:
            msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.img_publisher[topic_name].publish(msg)
        elif "depth" in topic_name:
            msg = self.bridge.cv2_to_imgmsg(image, encoding="8UC3")
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
        self.publish_raw_images(self.color_image_d405,"top_depth")
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
        self.get_logger().info("side image inferenance")
        # torch.cuda.empty_cache()
        filtered_pts = []
        # ---- Depth estimation ----
        with torch.inference_mode():
            # small = cv2.resize(self.color_image_d435i, (0,0), fx=0.5, fy=0.5) 
            # small= self.color_image_d435i[300:,:]
            


            crop1 = self.color_image_d435i[250:, :350]

            crop2 = self.color_image_d435i[250:, 300:]
            crops=[]
            for img in [crop1,crop2]:
                depth_img = self.depth_model.infer_image(img)

                depth_map_normalized = (depth_img - np.min(depth_img)) * 255 / np.max(depth_img)
                depth_image_uint8 = np.uint8(depth_map_normalized)
                crops.append(Image.fromarray(depth_image_uint8))
            # depth2=self.color_image_d435i.copy()

            # # depth2[250:, 300:]=depth_image_uint8
            # depth_image_bgr = cv2.cvtColor(depth2, cv2.COLOR_GRAY2BGR)
            # ros_depth_image = self.bridge.cv2_to_imgmsg(depth_image_bgr, encoding="bgr8")
            # self.processed_side_image_publisher_.publish(ros_depth_image)

        # crop1 = Image.fromarray(depth_image_uint8[130:, :130])
        # crop2 = Image.fromarray(depth_image_uint8[130:, 100:100+130])
        # crop3 = Image.fromarray(depth_image_uint8[130:, 200:])

        results = self.depth_yolo_model(crops,conf=0.35)
        # offsets = [(0, 130), (100, 130), (200, 130)]
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

                all_masks.append(polygon.tolist())

                # Create binary mask
                binary_image = np.zeros((H,W), dtype=np.uint8)
                cv2.fillPoly(binary_image, [polygon], 1)

                binary_images.append(binary_image)


        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(small)
        # for i in all_masks:
        #     i=np.array(i)
        #     ax.plot(i[:,0],i[:,1])
        # fig.canvas.draw()
        # result_image=np.array(fig.canvas.renderer.buffer_rgba())
        # result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        # ros_result_image = self.bridge.cv2_to_imgmsg(result_image_bgr, encoding="bgr8")
        # self.detected_side_img.publish(ros_result_image)

        
        H, W = self.color_image_d435i.shape[:2]
        pts_array = np.asarray(self.pts_d435i)          # (M, 3)
        pixel_array = np.asarray(self.pixel_pts_d435i)  # (M, 2)

        rows = pixel_array[:, 0].astype(np.int32)
        cols = pixel_array[:, 1].astype(np.int32)

        

        for mask in binary_images:

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
            valid_mask = mask_binary[rows, cols]

            mask_pts = pts_array[valid_mask.astype(bool)]
            if len(mask_pts)>1:
                filtered_pts.append(mask_pts)
        self.side_filtered_pts_raw = [[self.transform_d435i(x,y,z) for x,y,z in mushroom if (abs(x)+abs(y)+abs(z)>0.0001)] for mushroom in filtered_pts]
        # ts=[[-z-0.05,-x,y-0.03] for x,y,z in filtered_pts]
        filtered_pts=[]
        if len(self.side_filtered_pts_raw)>0:
            for i in self.side_filtered_pts_raw:
                filtered_pts.extend(i)
        else:
            filtered_pts=[[0,0,0]]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "Link1"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            filtered_pts
        )

        self.side_pc2_mushroom_points_publisher.publish(cloud_msg)

        
        return filtered_pts
    
    def process_frame(self):
        
        torch.cuda.empty_cache()
        # ---- Depth estimation ----
        with torch.inference_mode():

            small = cv2.resize(self.color_image_d405, (0,0), fx=0.5, fy=0.5) 

            depth_img = self.depth_model.infer_image(small)
            depth_img_og = cv2.resize(depth_img, (0,0), fx=2, fy=2)
            # scaled_depth= scale_prediction_to_depth_distribution(
            #     self.depth_image_d405,
            #     depth_img_og,
            #     lower_percentile=0.01,
            #     upper_percentile=99.99
            # )
            scaled_depth = scale_depth_patchwise(
            self.depth_image_d405,
            depth_img_og,
            patch_size=100,
            stride=25
        )
            
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

        pts=[]
        pts2=[]

        height, width = scaled_depth.shape
        # print("scaled_depth",scaled_depth.shape,"depth_image",self.depth_image_d405.shape)
        pts_x = np.zeros((height, width))
        pts_y = np.zeros((height, width))
        pts_z = np.zeros((height, width))

        increment=5
        for x in range(0, height, increment):
            for y in range(0, width, increment):
                depth_value = scaled_depth[x, y] * 0.0001

                point = rs.rs2_deproject_pixel_to_point(
                    self.color_intrin_d405, [y, x], depth_value
                )

                depth_value = self.depth_image_d405[x, y] * 0.0001

                
                
                
                pts_x[x:x+increment,y:y+increment]=point[0]
                pts_y[x:x+increment,y:y+increment]=-point[1]
                pts_z[x:x+increment,y:y+increment]=-point[2]
                if -point[2]>-0.2 and  -point[2]<-0.1:
                    pts.append([point[0], -point[1], -point[2]])
        if results[0].masks is None:
            self.get_logger().warn("nothing found")
            self.filtered_pts_raw=np.array([])
            return 


        masks = results[0].masks.data.cpu().numpy()
        

        all_mushroom_points=[]
        self.filtered_masks=[]
        if len(masks)>0:

            for mask in masks:
                
                
                print(mask.shape)
                mask_resized = cv2.resize(
                                mask.astype(np.uint8),
                                (width, height),  # OpenCV expects (width, height)
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                
                filtered_pts.append([[x-0.01,y,z] for x,y,z in zip(pts_x[mask_resized],pts_y[mask_resized],pts_z[mask_resized])])
                all_mushroom_points.extend([[x-0.01,y,z] for x,y,z in zip(pts_x[mask_resized],pts_y[mask_resized],pts_z[mask_resized])])
                self.filtered_masks.append([1 if i else 0 for i in mask.flatten()])
        self.filtered_pts_raw = filtered_pts.copy()

        # if filtered_pts:
        #     filtered_pts = np.array(filtered_pts)
        # else:
        #     filtered_pts = np.empty((0, 3))

        # ---- Final z filtering ----
        # filtered_pts = filtered_pts[filtered_pts[:, -1] < -0.02]
        # print(filtered_pts)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "gripper_base"

        cloud_msg = pc2.create_cloud_xyz32(
            header,
            all_mushroom_points
        )

        self.pc2_mushroom_points_publisher.publish(cloud_msg)

        return filtered_pts
    
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

        return respons