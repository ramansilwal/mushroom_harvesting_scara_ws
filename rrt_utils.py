
import numpy as np
from shapely import LineString

def forward_kinematics(angles, length):
    cumulative_angles = np.cumsum(angles)
    x = np.sum(length * np.cos(cumulative_angles))
    y = np.sum(length * np.sin(cumulative_angles))
    return np.array([x, y])

def jacobian(angles, length):
    dof = len(angles)
    J = np.zeros((2, dof))
    cumulative_angles = np.cumsum(angles)
    sin_angles = np.sin(cumulative_angles)
    cos_angles = np.cos(cumulative_angles)
    for i in range(dof):
        J[0, i] = -np.sum(length[i:] * sin_angles[i:])
        J[1, i] = np.sum(length[i:] * cos_angles[i:])
    return J

def inverse_kinematics(target, initial_angles, length,
                       angle_limits, lr=0.05, max_iters=1000, tol=1e-4):
    if initial_angles is None:
        angles = np.zeros(len(length))
    else:
        angles = initial_angles.copy()

    for _ in range(max_iters):
        pos = forward_kinematics(angles, length)
        error = target - pos
        if np.linalg.norm(error) < tol:
             return angles,error
        J = jacobian(angles, length)
        J_pinv = np.linalg.pinv(J)
        angles += lr * (J_pinv @ error)
        angles = np.clip(angles, angle_limits[:,0], angle_limits[:,1])
    return None,None


def forward_kinematics_batch(angles, length):
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


def check_movement_possible(start,end,length,obstacle_tree,n=10):
    fks=forward_kinematics_batch(np.linspace(start,end,n),length)
    robot_line=np.array([LineString(i) for i in fks])
    query,_ = obstacle_tree.query(robot_line,"intersects")
    if len(query)>0:
        return False
    
    return True