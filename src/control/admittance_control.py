import cv2
import time
import math
import numpy as np
from elibot.CPS import CPSClient
import yaml
from scipy.spatial.transform import Rotation as R
from src.utils.axis_transform import rpy_to_transformation_matrix, transformation_matrix_to_rpy
from src.utils.FT_reading.xjc_sensor.readForce import ModbusSensorReader


class RobotController:
    def __init__(self, box_id=0, rbt_id=0, position_file='./config/position.yaml',
                 calibration_file='./config/hand_eye_config.yaml',
                 step_size=5, rotation_step_size=0.1 * 10):
        """
        初始化机器人控制器
        :param step_size: 控制机器人每次移动的步进量（默认值为1）
        :param rotation_step_size: 控制机器人每次旋转的步进量（默认值为5）
        """
        self.box_id = box_id
        self.rbt_id = rbt_id
        self.position_file = position_file
        self.calibration_file = calibration_file
        self.client = CPSClient(ip="192.168.11.8")
        self.step_size = step_size  # 步进量
        self.rotation_step_size = rotation_step_size  # 旋转步进量
        self.FT_sensor = ModbusSensorReader()
        # 连接到电箱和控制器
        # 获取当前位置
        self.current_pose = self.client.getTcpPos()
        print(f"Initial Current Pose: {self.current_pose}")

        # 读取目标位置
        self.positions = self.read_position()
        self.hole = 30
        self.shaft = 29.6

    def read_position(self, tag='positions'):
        with open(self.position_file, 'r') as file:
            positions = yaml.safe_load(file)
        return positions[tag]

    def cal_offset(self, alpha):
        offset_x = self.hole * np.cos(alpha) - self.hole
        offset_y = 0
        offset_z = (self.shaft * np.cos(alpha) * np.cos(alpha) - 2 * self.hole * np.cos(alpha) + self.shaft) / np.sin(alpha)
        offset = np.array([offset_x, offset_y, offset_z])


        rpy = [math.pi, 0, math.pi / 2]  # rotate z 90 degrees ; rotate x 180 degrees

        RBW = calculate_rbw(rpy, 1)
        offset = RBW.dot(offset.transpose())
        pose = self.getTCPPose_reWrite()
        pose[3:6] = rot_vec2rpy(pose)
        RBW = calculate_rbw(pose)
        offset = RBW.transpose().dot(offset.transpose())
        return offset

    def adjust_pose_assembly(self, target_position):
        # 1. 目标位置
        target = np.array(target_position)  # 转化为numpy向量

        # 2. 计算绕目标轴旋转 135 度的旋转矩阵
        rotation_angle = 115  # 绕X轴旋转角度，单位为度
        r1 = R.from_euler('x', rotation_angle, degrees=True)  # 绕X轴旋转135度

        # 计算旋转角度绕Z轴，使用atan2计算xy平面上的角度
        rotation_angle = np.atan2(target[1], target[0])  # y/x的反正切值
        rotation_angle = np.rad2deg(rotation_angle)  # 转换为度数
        r2 = R.from_euler('z', rotation_angle, degrees=True)  # 绕Z轴旋转

        # 3. 获取旋转矩阵并合并
        rotation_matrix = r1.as_matrix() @ r2.as_matrix()
        r_final = R.from_matrix(rotation_matrix)

        # 4. 计算目标末端的姿态（旋转后的欧拉角）
        roll, pitch, yaw = r_final.as_euler('xyz', degrees=True)

        # 目标的偏移量
        TCP1 = np.array([0, 0, -258])  # 定义偏移量（TCP1）

        # 计算TCP1的旋转位置
        TCP1_posation = target + TCP1  # 向量加法，目标位置加上偏移量
        Rot_TCP1 = r_final.apply(TCP1)  # 应用旋转
        target_position1 = TCP1_posation + Rot_TCP1  # 向量加法，旋转后的目标位置

        # 更新目标位置的坐标
        x, y, z = target_position1

        # 5. 将目标位置和旋转结合，形成目标位姿
        target_pose = [x, y, z, roll, pitch, yaw]

        # 执行机器人运动
        self.client.moveLine(target_pose)

        # 返回当前位姿
        return

    def admittance_control(self):
        """
        使用 OpenCV 显示图像窗口并监听键盘输入来控制机械臂
        """
        # 初始化坐标
        sensor_data = np.zeros(6)
        while abs(sensor_data[2]) < 60:
            sensor_data = self.FT_sensor.read_sensor_data()
            print(sensor_data)
            x, y, z = sensor_data[:3].copy()  # x, y, z 为前3个系数

            rx, ry, rz = sensor_data[3:6]  # rx, ry, rz 为后3个系数
            rx, ry, rz = [0, 0, 0]
            self.current_pose = self.client.getTcpPos()
            roll, pitch, yaw = self.current_pose[3], self.current_pose[4], self.current_pose[5]
            base2end_rpy_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

            x, y, z = np.array([x, y, z]) @ base2end_rpy_matrix.T
            z = z - 10
            y = y - 10
            # 更新目标位置并控制机器人
            acc = 100  # 位移加速度
            arot = 10  # 姿态加速度
            t = 0.1  # 执行时间
            pose_diff = np.array([x, y, z, rx, ry, rz], dtype=float).tolist()

            pose_diff = [0 if abs(x) < 0.01 else x for x in pose_diff]
            # print(pose_diff)

            suc, result, _ = self.client.moveBySpeedl(pose_diff, acc, arot, t)
            ###############################################################
            self.current_pose = self.client.getTcpPos()
            rpy_to_transformation_matrix(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                         self.current_pose[3], self.current_pose[4], self.current_pose[5])
            # 打印当前坐标


def main():
    controller = RobotController()
    print(controller.client.getTcpPos())
    cycle_position = [675.9798681795281, -11.532278246412439, 348.8985805411238]
    cycle_position[2] = cycle_position[2] + 10
    cycle_position[1] = cycle_position[1] + 2

    cycle_position_pre = cycle_position.copy()
    cycle_position_pre[2] = cycle_position_pre[2] + 70
    controller.adjust_pose_assembly(target_position=cycle_position_pre)
    controller.adjust_pose_assembly(target_position=cycle_position)
    controller.admittance_control()  # 启动cv2控制


# 启动程序
if __name__ == "__main__":
    main()
