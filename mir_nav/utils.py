import numpy as np

def transform_2d(target_x, target_y, target_yaw, origin_x, origin_y, origin_yaw):
    t_mat = np.identity(3)
    t_mat[:2,2] = [-origin_x, -origin_y]
    r_mat = np.array([
                [np.cos(-origin_yaw), -np.sin(-origin_yaw), 0],
                [np.sin(-origin_yaw), np.cos(-origin_yaw), 0],
                [0, 0, 1]
            ])
    af = np.dot(r_mat, t_mat)
    target_xy = np.array([target_x, target_y, 1])
    transed_xy = np.dot(af, target_xy)
    
    return transed_xy[0], transed_xy[1], target_yaw - origin_yaw

def cartesian_to_polar_2d(x, y):
    return np.norm.linalg([x,y]), np.arctan2(y,x)

def polar_to_cartesian_2d(r, theta):
    return r*np.cos(theta), r*np.sin(theta)