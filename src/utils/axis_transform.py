import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def rpy_to_transformation_matrix(x, y, z, roll, pitch, yaw):
    # Create rotation matrix from RPY angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)  # Specify the sequence of axes 'xyz'
    rotation_matrix = r.as_matrix()

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Start with identity matrix

    # Set the rotation part (3x3) of the transformation matrix
    transformation_matrix[:3, :3] = rotation_matrix

    # Set the translation part (position) in the last column
    transformation_matrix[:3, 3] = [x, y, z]

    return transformation_matrix


# Function to convert a 4x4 transformation matrix to position and RPY
def transformation_matrix_to_rpy(matrix):
    # Extract the translation (x, y, z) from the last column of the matrix
    translation = matrix[:3, 3]
    x, y, z = translation

    # Extract the rotation matrix (top-left 3x3 part of the transformation matrix)
    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to RPY (roll, pitch, yaw) in radians
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)

    return x, y, z, roll, pitch, yaw
