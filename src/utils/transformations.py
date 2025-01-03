# vision_utils.py

"""
此模块包含与视觉相关的实用函数，如从相机捕获图像、使用YOLO进行目标检测，
以及计算检测到的目标在相机坐标系中的3D位置。具体包括：
- get_image_and_detect: 获取图像并使用YOLO进行目标检测。
- calculate_position: 计算检测到的目标物体的3D位置。
"""
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_position(head_detections, testtube_detection, depth_image, depth_frame, camera):
    # 计算检测到的目标物体的3D位置的函数。
    if head_detections is None or len(head_detections) == 0:
        print("检测失败，没有检测到任何目标")
        return None  # 或者返回一个默认值，例如 (0, 0, 0)

    # 假设检测到的第一个目标为所需目标，获取其四个顶点的坐标
    x1, y1, x2, y2, x3, y3, x4, y4 = head_detections.flatten()

    # 计算目标的中心点
    center_x = int((x1 + x2 + x3 + x4) // 4)
    center_y = int((y1 + y2 + y3 + y4) // 4)

    # 获取中心点的深度值
    center_depth = camera.get_depth_for_color_pixel(depth_frame=depth_frame,
                                                    color_point=[center_x, center_y])

    if center_depth == 0:
        # 如果中心点的深度为0，则查找最近的不为0的深度值
        found = False
        search_radius = 1
        while not found:
            # 确定搜索范围的边界
            min_x = max(center_x - search_radius, 0)
            max_x = min(center_x + search_radius + 1, depth_frame.width)  # 使用 depth_frame 的宽度
            min_y = max(center_y - search_radius, 0)
            max_y = min(center_y + search_radius + 1, depth_frame.height)  # 使用 depth_frame 的高度

            # 遍历搜索范围内的每个像素点
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    depth = camera.get_depth_for_color_pixel(depth_frame=depth_frame,
                                                             color_point=[x, y])
                    if depth > 0:
                        center_depth = depth  # 获取最近的非零深度值
                        found = True
                        break
                if found:
                    break

            if not found:
                search_radius += 1  # 增加搜索半径

    # 获取深度比例因子
    depth_scale = camera.depth_scale

    # 计算实际深度（Z轴）
    real_z = center_depth * depth_scale

    fx, fy = camera.fx, camera.fy  # 获取相机的焦距参数
    real_x = (center_x - camera.ppx) * real_z / fx  # 计算目标的实际X坐标
    real_y = (center_y - camera.ppy) * real_z / fy  # 计算目标的实际Y坐标

    x1, y1, x2, y2, x3, y3, x4, y4 = testtube_detection.flatten()

    # 计算目标的旋转角度 theta
    delta_x = x2 - x1
    delta_y = y2 - y1
    theta = np.arctan2(delta_y, delta_x)  # 计算目标的旋转角度（弧度）
    theta_degrees = np.degrees(theta)  # 将角度从弧度转换为度数

    print(
        f"目标位置（使用depth_image计算）（相机坐标系）：x={real_x}, y={real_y}, z={real_z}, 旋转角度 theta={theta_degrees}°")
    return real_x, real_y, real_z, theta_degrees  # 返回目标的实际坐标和旋转角度


def calculate_position_rack(rack_detections, camera, depth_frame, ):
    # 计算检测到的目标物体的3D位置的函数。
    if rack_detections is None or len(rack_detections) == 0:
        print("检测失败，没有检测到任何目标")
        return None  # 或者返回一个默认值，例如 (0, 0, 0)

    # 假设检测到的第一个目标为所需目标，获取其四个顶点的坐标
    x1, y1, x2, y2, x3, y3, x4, y4 = rack_detections.flatten()

    # 计算目标的中心点
    # 计算目标的中心点
    center_x = int((x1 + x2 + x3 + x4) // 4)
    center_y = int((y1 + y2 + y3 + y4) // 4)
    # 计算实际深度（Z轴）
    # 获取中心点的深度值
    center_depth = 281
    depth_scale = camera.depth_scale
    real_z = center_depth * depth_scale

    fx, fy = camera.fx, camera.fy  # 获取相机的焦距参数
    real_x = (center_x - camera.ppx) * real_z / fx  # 计算目标的实际X坐标
    real_y = (center_y - camera.ppy) * real_z / fy  # 计算目标的实际Y坐标
    theta_degrees = 0
    print(
        f"目标位置（使用depth_image计算）（相机坐标系）：x={real_x}, y={real_y}, z={real_z}, 旋转角度 theta={theta_degrees}°")
    # real_x, real_y, real_z,

    return real_x, real_y, real_z, theta_degrees  # 返回目标的实际坐标和旋转角度


def calculate_circles_position_in_rack_area(pixels, camera, robot):
    # 计算检测到的目标物体的3D位置的函数。
    real_position = []
    for center_x, center_y in pixels:
        # 计算实际深度（Z轴）
        # 获取中心点的深度值
        center_depth = 281

        depth_scale = camera.depth_scale
        real_z = center_depth * depth_scale

        fx, fy = camera.fx, camera.fy  # 获取相机的焦距参数
        real_x = (center_x - camera.ppx) * real_z / fx  # 计算目标的实际X坐标
        real_y = (center_y - camera.ppy) * real_z / fy  # 计算目标的实际Y坐标
        xyz_rpy = convert_to_arm_coords(real_x, real_y, real_z, 0, robot, camera, )
        real_position.append(xyz_rpy)

    return real_position  # 返回目标的实际坐标和旋转角度


def convert_to_arm_coords(real_x, real_y, real_z, theta, robot, camera, tube_head_rotation=False):
    base_to_end_effector = robot.get_pose()  # 获取机械臂末端相对于基座的位姿矩阵
    base_to_end_effector[:3, -1] = base_to_end_effector[:3, -1] / 1000
    rotation = R.from_euler('z', theta + 90, degrees=True)  # 创建绕 Z 轴旋转的 Rotation 对象
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()  # 将 3x3 旋转矩阵嵌入到 4x4 齐次变换矩阵中
    camera_point = np.eye(4)
    camera_point[:4, -1] = np.array([real_x, real_y, real_z, 1])  # 将相机坐标转换为齐次坐标
    # camera_point[:3, :3] =rotation_matrix[:3, :3]
    # 通过矩阵运算将相机坐标转换为机械臂坐标
    arm_point = (base_to_end_effector @ camera.extrinsic_matrix
                 @ camera_point @ rotation_matrix)
    # 提取机械臂坐标系中的 X、Y、Z
    arm_x, arm_y, arm_z = arm_point[:3, -1]  # 获取机械臂坐标中的 X、Y、Z

    # 提取旋转矩阵
    arm_rotation_matrix = arm_point[:3, :3]
    rpy = R.from_matrix(arm_rotation_matrix).as_euler('xyz', degrees=True)
    print(f"目标位置（机械臂坐标系）：")
    print(f"x={arm_x}, y={arm_y}, z={arm_z}")
    print(f"Roll={rpy[0]}, Pitch={rpy[1]}, Yaw={rpy[2]}")
    if tube_head_rotation:
        rpy[-1] = rpy[-1] + 180
    xyz_rpy = np.concatenate((np.array([arm_x, arm_y, arm_z]), rpy))
    return xyz_rpy  # 返回转换后的机械臂坐标
