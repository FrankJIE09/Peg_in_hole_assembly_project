import numpy as np

def calculate_offset(axis_pos, hole_pos):
    return hole_pos - axis_pos

def transform_to_robot_frame(camera_pos, transformation_matrix):
    homogenous_pos = np.append(camera_pos, 1)
    robot_frame_pos = transformation_matrix.dot(homogenous_pos)
    return robot_frame_pos[:3]