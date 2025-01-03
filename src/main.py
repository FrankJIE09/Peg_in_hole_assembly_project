import time
import math
import yaml
import cv2
import numpy as np
import pyrealsense2 as rs


# 假设的机械臂控制库
# 您需要根据实际使用的机械臂替换此部分
class RobotArm:
    def __init__(self):
        # 初始化机械臂连接
        pass

    def move_to(self, position):
        # 根据位置字典移动机械臂
        # position包含x, y, z, rx, ry, rz等信息
        print(f"移动到位置: {position}")
        # 实现具体的移动代码
        time.sleep(1)  # 模拟移动时间

    def set_pose(self, pose):
        # 设置机械臂的姿态，包括位置和角度
        print(f"设置姿态: {pose}")
        # 实现具体的设置代码
        time.sleep(1)

    def move_along_z(self, delta_z):
        # 沿z轴移动
        print(f"沿z轴移动: {delta_z} 米")
        # 实现具体的移动代码
        time.sleep(0.1)

    def rotate_insert(self, angle):
        # 旋转机械臂进行插入操作
        print(f"旋转插入: {angle} 度")
        # 实现具体的旋转代码
        time.sleep(1)


# 假设的力传感器库
# 您需要根据实际使用的力传感器替换此部分
class ForceSensor:
    def __init__(self):
        # 初始化力传感器连接
        pass

    def get_force(self):
        # 返回当前的力值，格式为字典
        # 例如: {'fx': 0, 'fy': 0, 'fz': 12, 'tx': 0, 'ty': 0, 'tz': 0}
        # 这里用模拟数据
        force = {'fx': 0, 'fy': 0, 'fz': 12, 'tx': 0, 'ty': 0, 'tz': 0}
        return force


def read_position_config(config_path='position_config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def find_circle_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # 假设第一个检测到的圆是目标圆
        return (circles[0][0], circles[0][1])
    else:
        print("未检测到圆形")
        return None


def calculate_relative_coordinates(circle_center, image_size, camera_intrinsics, depth=0.1):
    # 计算图像中圆心的像素坐标相对于机械臂的实际坐标
    # 这里需要根据相机的内参和机械臂的坐标系进行转换
    cx, cy = circle_center
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx_intr, cy_intr = camera_intrinsics['cx'], camera_intrinsics['cy']

    X = (cx - cx_intr) * depth / fx
    Y = (cy - cy_intr) * depth / fy
    Z = depth
    return (X, Y, Z)


def calculate_offset(force):
    # 根据力传感器的数据计算偏移量
    # 这是一个示例，您需要根据实际算法实现
    # 假设偏移量与Fz成比例
    k = 0.01  # 比例系数，根据实际情况调整
    offset = k * (force['fz'] - 10)  # 例如，当Fz > 10时计算偏移
    return offset


def main():
    # 读取配置文件
    config = read_position_config()

    # 初始化机械臂和力传感器
    robot = RobotArm()
    force_sensor = ForceSensor()

    # 初始化Realsense相机
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 设置相机曝光
    device = pipeline.get_active_profile().get_device()
    color_sensor = device.first_color_sensor()
    exposure = config.get('exposure', 100)
    color_sensor.set_option(rs.option.exposure, exposure)

    try:
        # 步骤1：移动到初始目标位置
        print("步骤1：移动到初始目标位置")
        initial_position = config['initial_position']
        robot.move_to(initial_position)

        # 步骤2：读取position_config.yaml移动到目标位置
        print("步骤2：读取position_config.yaml移动到目标位置")
        target_position = config['target_position']
        robot.move_to(target_position)

        # 步骤3：使用Realsense相机获取图像并检测圆心
        print("步骤3：使用Realsense相机获取图像并检测圆心")
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            print("未获取到颜色帧")
            return
        color_image = np.asanyarray(color_frame.get_data())

        # 仅保留图像中间区域，其他部分进行蒙版处理
        mask = np.zeros_like(color_image)
        height, width = color_image.shape[:2]
        center_region = cv2.rectangle(mask.copy(),
                                      (width // 4, height // 4),
                                      (3 * width // 4, 3 * height // 4),
                                      (255, 255, 255),
                                      -1)
        masked_image = cv2.bitwise_and(color_image, center_region)

        # 检测圆心
        circle_center = find_circle_center(masked_image)
        if circle_center is None:
            print("无法找到圆心，终止程序")
            return
        print(f"检测到的圆心坐标（像素）：{circle_center}")

        # 显示检测结果（可选）
        # cv2.circle(color_image, circle_center, 5, (0, 255, 0), -1)
        # cv2.imshow("Detected Circle", color_image)
        # cv2.waitKey(0)

        # 步骤4：计算圆心到机械臂的相对坐标关系
        print("步骤4：计算圆心到机械臂的相对坐标关系")
        image_size = (color_frame.get_width(), color_frame.get_height())
        camera_intrinsics = config['camera_intrinsics']
        relative_coords = calculate_relative_coordinates(circle_center, image_size, camera_intrinsics)
        print(f"圆心相对于机械臂的坐标：{relative_coords}")

        # 步骤5：设定机械臂的角度为ry=45度，位置在圆心的斜上方
        print("步骤5：设定机械臂的角度为ry=45度，位置在圆心的斜上方")
        ry = 45  # 角度
        offset_height = 0.05  # 斜上方的高度偏移，单位米
        target_pose = {
            'x': relative_coords[0],
            'y': relative_coords[1],
            'z': relative_coords[2] + offset_height,
            'rx': 0,  # 假设rx和rz为0，根据需要调整
            'ry': ry,
            'rz': 0
        }
        robot.set_pose(target_pose)

        # 步骤6：依靠末端的z方向移动，根据力传感器的六个方向受力F=kv，直到Fz>10
        print("步骤6：依靠末端的z方向移动，根据力传感器的六个方向受力F=kv，直到Fz>10")
        while True:
            robot.move_along_z(-0.001)  # 向下微小移动，单位米
            force = force_sensor.get_force()
            print(f"当前力传感器读数：{force}")
            if force['fz'] > 10:  # 单位根据实际传感器
                print("达到Fz > 10，停止移动")
                break
            time.sleep(0.1)  # 等待一段时间再读取

        # 步骤7：依据算法计算偏移量，给机械臂补充后，直接旋转插入
        print("步骤7：依据算法计算偏移量，给机械臂补充后，直接旋转插入")
        offset = calculate_offset(force)
        adjusted_pose = {
            'x': target_pose['x'] + offset,
            'y': target_pose['y'] + offset,
            'z': target_pose['z'],  # 这里假设z不变，根据实际情况调整
            'rx': target_pose['rx'],
            'ry': target_pose['ry'],
            'rz': target_pose['rz']
        }
        robot.set_pose(adjusted_pose)
        robot.rotate_insert(angle=90)  # 假设旋转90度进行插入

        print("轴孔装配完成")

    except Exception as e:
        print(f"发生异常: {e}")

    finally:
        # 释放相机资源
        pipeline.stop()


if __name__ == "__main__":
    main()
