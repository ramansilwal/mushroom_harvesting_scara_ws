import numpy as np
from rclpy.node import Node
import rclpy


import tf2_ros
from scipy.spatial.transform import Rotation as R

def transform_mushroom_to_global(picked_mushroom,T):
    if T is None:
        T=get_transform()

    points = picked_mushroom
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points_hom = (T @ points_hom.T).T  
    # x,y,z,_=np.mean(transformed_points_hom,axis=0)
    return transformed_points_hom[:,:3]


def get_transform():
    try:
        node=Node("temp")
        node.tf_buffer = tf2_ros.Buffer()
        node.tf_listener = tf2_ros.TransformListener(node.tf_buffer, node)
        for i in range(0,10):
            rclpy.spin_once(node)
            if i%5==0:
                try:
                    transform = node.tf_buffer.lookup_transform(
                                                "base_link",  # target frame
                                                "gripper_base",  # source frame
                                                rclpy.time.Time(),  # time (latest)
                                                timeout=rclpy.duration.Duration(seconds=0.5),
                                            )
                    break
                except Exception as e:
                    print(e)
        node.destroy_node()
        t=transform.transform.translation 
        q=transform.transform.rotation 
        translation = np.array([t.x, t.y, t.z])  
        quaternion = [q.x, q.y, q.z, q.w]

        rotation_matrix = R.from_quat(quaternion).as_matrix()  
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation 
        return T
    except Exception as e:
        print(e)
        print("Failed to get transform")
        return None


import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


tip_data = {
    1.80: [(305,145),(303,100), (150,20),(110,42),   (105,235),(140,256)],
    1.85: [(297,145),(297,100), (151,21),(113,44),   (107,235),(140,256)],
    1.90: [(294,148),(292,101), (154,35),(119,53),   (112,222),(149,243)],
    1.95: [(288,145),(287,100), (158,31),(120,56),   (114,216),(150,236)],
    2.00: [(279,144),(280,100), (160,40),(124,62),   (120,210),(154,231)],
    2.05: [(277,146),(276,100), (162,48),(126,67),   (119,206),(157,228)],
    2.10: [(270,145),(272,101), (165,53),(129,74),   (121,202),(159,224)],
    2.15: [(266,148),(265,103), (170,55),(128,77),   (124,197),(162,219)],
    2.20: [(257,146),(260,102), (170,58),(134,85),   (128,192),(166,214)],
    2.25: [(257,148),(256,103), (175,60),(134,84),   (130,187),(171,212)],
    2.30: [(250,148),(250,102), (174,71),(139,88),   (132,180),(171,206)],
    2.35: [(243,151),(245,102), (180,70),(143,94),   (133,180),(172,199)],
    2.40: [(240,152),(239,102), (183,74),(142,97),   (134,172),(176,194)],
    2.45: [(235,148),(235,103), (187,80),(146,103),  (140,169),(179,192)],
    2.50: [(229,150),(228,103), (189,86),(150,108),  (141,164),(181,185)],
    2.55: [(223,148),(223,102), (191,89),(153,113),  (145,158),(184,179)],
    2.60: [(218,149),(218,102), (195,93),(153,116),  (146,151),(192,186)],
    2.65: [(213,150),(213,102), (199,98),(158,124),  (153,146),(193,171)],
    2.70: [(207,151),(207,102), (202,103),(161,131), (151,144),(194,166)],
    2.75: [(200,164),(202,111), (202,111),(165,136), (165,136),(200,164)],
}

def get_tip_positions(mask, tip_data, tolerance=5):
    """
    Returns first gripper position at which each tip reaches the mask.

    Example:
        [2.05, 2.10, 2.35]

    mask:
        binary numpy array, shape (256, 320)

    tolerance:
        distance in pixels considered contact.
    """
    import numpy as np
    mask = np.asarray(mask)

    # assert mask.shape == (256, 320), \
    #     f"Expected (256,320), got {mask.shape}"

    # convert to boolean
    mask_binary = mask > 0

    # distance of every background pixel to the mask
    distance_map = distance_transform_edt(~mask_binary)

    positions = sorted(tip_data.keys())

    result = []

    # tip 0, 1, 2
    for tip in range(3):

        contact_position = None
        best_position = None
        best_distance = np.inf

        for position in positions:

            points = tip_data[position]

            # each tip has two endpoints
            p1 = points[tip * 2]
            p2 = points[tip * 2 + 1]

            # make temporary image containing only this tip line
            line_mask = np.zeros(mask.shape, dtype=np.uint8)

            cv2.line(
                line_mask,
                p1,
                p2,
                255,
                thickness=1
            )

            ys, xs = np.where(line_mask > 0)

            # remove coordinates outside image
            valid = (
                (xs >= 0) &
                (xs < mask.shape[1]) &
                (ys >= 0) &
                (ys < mask.shape[0])
            )

            xs = xs[valid]
            ys = ys[valid]

            if len(xs) == 0:
                continue

            # closest point of tip to mushroom mask
            distance = distance_map[ys, xs].min()

            # remember globally closest position
            if distance < best_distance:
                best_distance = distance
                best_position = position

            # FIRST position at which this tip touches mushroom
            if distance <= tolerance:
                contact_position = position
                break

        # if no actual contact occurred,
        # return closest observed position
        if contact_position is None:
            contact_position = best_position

        result.append(contact_position)

    return result


    import numpy as np

class Mushroom:
    def __init__(self, points,T=None,mask=None):
        if hasattr(points, "points_flat"):
            points = points.points_flat
        points = np.array(points)
        # Only reshape if needed
        if points.ndim != 2:
            points = points.reshape(-1, 3)
        self.points_original=points.copy()
        if T is not None:
            self.points=transform_mushroom_to_global(points,T)
        else:
            self.points = points.copy()
        self.x = np.mean(self.points[:, 0])
        self.y = np.mean(self.points[:, 1])
        self.xy= np.array([self.x,self.y])
        self.distance=np.linalg.norm(self.xy)
        if mask:
            
            mask=np.array(mask)
            self.mask=mask.reshape(256,320)
        
        if T is not None:
            self.gripper_x = np.mean(self.points_original[:, 0])
            self.gripper_y = np.mean(self.points_original[:, 1])
            self.distance_from_gripper=np.linalg.norm([self.gripper_x,self.gripper_y])
        else:
            self.gripper_x=None
            self.gripper_y=None

    def __repr__(self):
        if self.gripper_x is not None:
            val = f"x: {self.x:.4f}, y: {self.y:.4f} \n gripper: x: {self.gripper_x:.4f} , y {self.gripper_y:.4f}"
        else:
            val = f"x: {self.x:.4f}, y: {self.y:.4f}"
        return val
    def __sub__(self,other):
        return np.linalg.norm(self.xy-other.xy)
