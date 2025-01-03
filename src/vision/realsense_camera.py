import time

import numpy as np  # 导入NumPy库，用于数组和矩阵操作
from pyrealsense2 import pyrealsense2 as rs  # 导入Intel RealSense库，用于相机操作
import yaml
import json


def adjust_exposure_based_on_brightness(camera, target_brightness=128):
    """
    根据图像亮度自动调节曝光值。
    参数:
    - camera: RealSenseCamera 对象。
    - target_brightness: 目标亮度值 (0-255)。
    """
    # 设置调整步长和曝光范围
    exposure_step = 50
    min_exposure = 50
    max_exposure = 10000

    while True:
        color_image, _, _ = camera.get_frames()  # 获取一帧图像

        # 根据人眼感知计算亮度（加权平均法）
        current_brightness = (
                0.299 * color_image[:, :, 2] +  # Red 通道
                0.587 * color_image[:, :, 1] +  # Green 通道
                0.114 * color_image[:, :, 0]  # Blue 通道
        ).mean()

        print(f"当前亮度: {current_brightness}")

        # 根据亮度调整曝光值
        current_exposure = camera.color_sensor.get_option(rs.option.exposure)
        if current_brightness < target_brightness - 10:
            new_exposure = min(current_exposure + exposure_step, max_exposure)
            camera.set_exposure(new_exposure)
            print(f"亮度过低，增加曝光值到: {new_exposure}")
        elif current_brightness > target_brightness + 10:
            new_exposure = max(current_exposure - exposure_step, min_exposure)
            camera.set_exposure(new_exposure)
            print(f"亮度过高，减少曝光值到: {new_exposure}")
        else:
            print("亮度已调整至合理范围，无需进一步调整。")
            break


class RealSenseCamera:
    def __init__(self, config_extrinsic='./config/hand_eye_config.yaml',
                 ):
        # 将配置项移动到实例变量中

        context = rs.context()
        device_id = _find_connected_devices(context)[0]
        self.resolution_width = 1280
        self.resolution_height = 720
        self.frame_rate = 30
        self.depth_min = 0.1
        self.depth_max = 10.0

        self.device_id = device_id
        self._context = context

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self.device_id)
        self._config.enable_stream(rs.stream.depth, self.resolution_width, self.resolution_height, rs.format.z16,
                                   self.frame_rate)
        self._config.enable_stream(rs.stream.color, self.resolution_width, self.resolution_height, rs.format.bgr8,
                                   self.frame_rate)

        # 启动数据流并获取设备的profile信息
        self._pipeline_profile = self._pipeline.start(self._config)
        self._depth_scale = self._pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale = self._depth_scale
        self.color_profile = self._pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.depth_profile = self._pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()

        # 获取内参和外参
        self.color_intrinsics = self.color_profile.get_intrinsics()
        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.color_to_depth_extrinsics = self.color_profile.get_extrinsics_to(self.depth_profile)
        self.depth_to_color_extrinsics = self.depth_profile.get_extrinsics_to(self.color_profile)

        # Filters initialization
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

        # Configure filters based on JSON settings
        self.decimation_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.1)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal_filter.set_option(rs.option.holes_fill, 3)

        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.max_distance, 4.0)
        self.threshold_filter.set_option(rs.option.min_distance, 0.1)

        self.load_extrinsic(config_extrinsic)

        # Get camera intrinsic details
        self.intrinsics = self._pipeline_profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx = self.intrinsics.fx
        self.fy = self.intrinsics.fy
        self.ppx = self.intrinsics.ppx
        self.ppy = self.intrinsics.ppy
        self.color_sensor = self._pipeline_profile.get_device().first_color_sensor()
        adjust_exposure_based_on_brightness(self, target_brightness=158)

    def load_extrinsic(self, config_path):
        """从配置文件中加载外参."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                transformation_matrix = config['hand_eye_transformation_matrix']
                # 将外参矩阵转换为 NumPy 数组
                self.extrinsic_matrix = np.array(transformation_matrix)
        except Exception as e:
            raise RuntimeError(f"加载外参失败: {e}")

    def get_frames(self):
        """Returns a frames object with each available frame type"""
        frames = self._pipeline.wait_for_frames()  # 等待获取一帧数据
        # print("realsense camera get frames!")
        depth_frame: rs.depth_frame = frames.get_depth_frame()  # 获取深度帧
        color_frame: rs.video_frame = frames.get_color_frame()  # 获取彩色帧
        color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())  # 将过滤后的深度帧转换为numpy数组

        return color_image, depth_image, depth_frame  # 返回获取到的帧数据

    def set_exposure(self, value):
        """
        设置相机的曝光值。
        参数:
        value (int): 曝光值（范围可能因设备而异，一般在50到10000之间）。
        """
        try:
            self.color_sensor.set_option(rs.option.exposure, value)
            print(f"曝光值已设置为: {value}")
        except Exception as e:
            print(f"设置曝光值时发生错误: {e}")

    def close(self):
        """关闭相机的数据流"""
        self._pipeline.stop()  # 停止数据流管道

    def get_depth_for_color_pixel(self, depth_frame, color_point: np.array, ):
        """将图像像素点转换为3D对象点"""
        color_point = np.array(color_point)
        depth_pixels = [
            rs.rs2_project_color_pixel_to_depth_pixel(
                data=depth_frame.get_data(),  # 深度帧数据
                depth_scale=self._depth_scale,  # 深度比例
                depth_min=self.depth_min,  # 最小深度值
                depth_max=self.depth_max,  # 最大深度值
                depth_intrin=self.depth_intrinsics,  # 深度内参
                color_intrin=self.color_intrinsics,  # 彩色内参
                depth_to_color=self.depth_to_color_extrinsics,  # 深度到彩色的外参
                color_to_depth=self.color_to_depth_extrinsics,  # 彩色到深度的外参
                from_pixel=color_point  # 当前彩色像素点
            )
        ]

        object_points = []  # 存储计算得到的3D对象点
        depth_data = np.asanyarray(depth_frame.get_data())  # 获取深度图像的 NumPy 数组

        def get_valid_depth(x, y):
            """辅助函数，寻找非零深度值"""
            if depth_data[y, x] != 0:
                return depth_data[y, x]

            # 如果深度为0，查找周围的像素
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < depth_data.shape[1] and 0 <= ny < depth_data.shape[0]:  # 检查索引边界
                    if depth_data[ny, nx] != 0:
                        return depth_data[ny, nx]

            return 0  # 如果周围像素也没有有效深度，返回0

        for pixel in depth_pixels:  # 遍历计算得到的深度像素点
            depth_x = int(round(pixel[0]))
            depth_y = int(round(pixel[1]))
            depth = get_valid_depth(depth_x, depth_y)  # 获取有效深度值并考虑深度比例

            if depth > 0:
                object_points.append(
                    rs.rs2_deproject_pixel_to_point(intrin=self.depth_intrinsics, pixel=pixel, depth=depth)
                )  # 将深度像素点反投影为3D对象点

        return depth  # 返回计算得到的3D对象点列表

    def depth_frame_to_object_points(self, frames: rs.composite_frame):
        """将深度帧转换为3D对象点数组"""
        depth_frame = frames.get_depth_frame()  # 获取深度帧
        object_points = []  # 存储计算得到的3D对象点

        # 遍历深度图像的宽度和高度
        for x in range(self.depth_intrinsics.width):
            for y in range(self.depth_intrinsics.height):
                pixel = [x, y]  # 当前像素点的坐标
                depth = depth_frame.get_distance(x, y)  # 获取该像素点的深度值
                object_points.append(
                    rs.rs2_deproject_pixel_to_point(intrin=self.depth_intrinsics, pixel=pixel,
                                                    depth=depth))  # 将像素点反投影为3D对象点

        return np.array(object_points)  # 返回计算得到的3D对象点数组


def extract_color_image(frames: rs.composite_frame) -> np.ndarray:
    """定义一个函数，从帧数据中提取彩色图像"""
    return np.asanyarray(frames.get_color_frame().get_data())  # 返回彩色帧数据的NumPy数组


def _find_connected_devices(context):
    """定义一个函数，查找所有已连接的相机设备"""
    devices = []  # 存储已连接设备的列表
    for device in context.devices:  # 遍历上下文中的所有设备
        if device.get_info(rs.camera_info.name).lower() != 'platform camera':  # 排除平台相机
            devices.append(device.get_info(rs.camera_info.serial_number))  # 将设备的序列号添加到设备列表中
    return devices  # 返回已连接设备的序列号列表


def initialize_connected_cameras():
    """初始化所有已连接的相机"""
    context = rs.context()  # 创建一个新的上下文对象
    device_ids = _find_connected_devices(context)  # 查找所有已连接的设备ID
    devices = [RealSenseCamera(device_id, context) for device_id in device_ids]  # 为每个设备ID创建一个Camera对象
    return devices  # 返回所有创建的Camera对象列表


def close_connected_cameras(cameras):
    """关闭所有已连接的相机"""
    for camera in cameras:  # 遍历所有Camera对象
        camera.close()  # 关闭相机的数据流


def main():
    # 初始化连接的所有相机
    cameras = initialize_connected_cameras()

    if len(cameras) == 0:
        # print("No cameras connected.")
        return

    # 假设只使用第一个相机
    camera = cameras[0]

    try:
        # 获取一帧图像数据
        for i in range(100):
            color_image, depth_image, depth_frame = camera.get_frames()

            # 指定要查询深度的彩色像素点列表（x, y），这里以(832, 379)为例
            color_pixels = np.array([[1098, 197]])

            # 计算该彩色像素点对应的3D对象点（即深度信息）
            object_points = camera.get_depth_for_color_pixel(depth_frame, color_pixels, )

            # for idx, point in enumerate(object_points):
            #     print(f"Color Pixel {color_pixels[idx]} -> Object Point {point} (x, y, z in meters)")

    finally:
        # 关闭相机
        close_connected_cameras(cameras)


if __name__ == "__main__":
    main()