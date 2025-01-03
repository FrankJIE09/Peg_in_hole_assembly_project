import cv2
import time
import math
import numpy as np
from dazu.CPS import CPSClient
import yaml


class RobotController:
    def __init__(self, box_id=0, rbt_id=0, position_file='./config/position.yaml',
                 calibration_file='./config/hand_eye_config.yaml',
                 step_size=0.3, rotation_step_size=0.5):
        """
        初始化机器人控制器
        :param step_size: 控制机器人每次移动的步进量（默认值为1）
        :param rotation_step_size: 控制机器人每次旋转的步进量（默认值为5）
        """
        self.box_id = box_id
        self.rbt_id = rbt_id
        self.position_file = position_file
        self.calibration_file = calibration_file
        self.client = CPSClient()
        self.step_size = step_size  # 步进量
        self.rotation_step_size = rotation_step_size  # 旋转步进量

        # 连接到电箱和控制器
        self.client.HRIF_Connect(self.box_id, '192.168.11.7', 10003)
        self.client.HRIF_Connect2Controller(self.box_id)

        # 获取当前位置
        self.current_pose = self.client.read_pos()
        print(f"Initial Current Pose: {self.current_pose}")

        # 读取目标位置
        self.positions = self.read_position()

    def read_position(self, tag='positions'):
        with open(self.position_file, 'r') as file:
            positions = yaml.safe_load(file)
        return positions[tag]

    def move_arm(self, pose, dServoTime):
        """
        控制机械臂移动到指定的pose，pose包含位置和姿态的6个参数（x, y, z, Rx, Ry, Rz）。
        """
        ucs = [0, 0, 0, 0, 0, 0]  # 运动坐标系的占位符
        tcp = [0, 0, 0, 0, 0, 0]  # 末端执行器坐标系的占位符
        res = self.client.HRIF_PushServoP(self.box_id, self.rbt_id, pose, ucs, tcp)
        print(res)
        time.sleep(0.001)  # 延迟，防止命令重叠

    def control_arm_with_cv2(self):
        """
        使用 OpenCV 显示图像窗口并监听键盘输入来控制机械臂
        """
        # 初始化坐标

        # 创建窗口
        cv2.namedWindow('Robot Control')
        dServoTime = 0.025
        dLookaheadTime = 0.05

        while True:
            # 显示图像（简单的文本提示）
            img = 255 * np.ones(shape=[620,900 , 3], dtype=np.uint8)  # 创建一个白色背景图像
            cv2.putText(img, "Use Arrow Keys to control the robot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, f"Step Size: {self.step_size}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, f"Rotation Step Size: {self.rotation_step_size}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
            cv2.putText(img, "W: Move Forward (Increase X)", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "S: Move Backward (Decrease X)", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "A: Move Left (Decrease Y)", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "D: Move Right (Increase Y)", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "Q: Move Up (Increase Z)", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "E: Move Down (Decrease Z)", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "I: Rotate Counter-Clockwise (Increase Rx)", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 2)
            cv2.putText(img, "K: Rotate Clockwise (Decrease Rx)", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),
                        2)
            cv2.putText(img, "Press ESC to exit", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # 显示图像
            cv2.imshow('Robot Control', img)
            self.current_pose = self.client.read_pos()
            x, y, z = self.current_pose[0], self.current_pose[1], self.current_pose[2]
            rx, ry, rz = self.current_pose[3], self.current_pose[4], self.current_pose[5]
            # 按键事件处理
            key = cv2.waitKey(1) & 0xFF  # 获取按键的ASCII码
            if key == 27:  # 按下ESC退出
                break
            elif key == ord('w'):
                x += self.step_size  # 移动前进（增加X坐标）
            elif key == ord('s'):
                x -= self.step_size  # 移动后退（减少X坐标）
            elif key == ord('a'):
                y -= self.step_size  # 向左移动（减少Y坐标）
            elif key == ord('d'):
                y += self.step_size  # 向右移动（增加Y坐标）
            elif key == ord('q'):
                z += self.step_size  # 向上移动（增加Z坐标）
            elif key == ord('e'):
                z -= self.step_size  # 向下移动（减少Z坐标）
            elif key == ord('i'):
                rx += self.rotation_step_size  # 绕X轴顺时针旋转（增加Rx）
            elif key == ord('k'):
                rx -= self.rotation_step_size  # 绕X轴逆时针旋转（减少Rx）
            elif key == ord('j'):
                ry += self.rotation_step_size  # 绕Y轴顺时针旋转（增加Ry）
            elif key == ord('l'):
                ry -= self.rotation_step_size  # 绕Y轴逆时针旋转（减少Ry）
            elif key == ord('u'):
                rz += self.rotation_step_size  # 绕Z轴顺时针旋转（增加Rz）
            elif key == ord('o'):
                rz -= self.rotation_step_size  # 绕Z轴逆时针旋转（减少Rz）

            # 更新目标位置并控制机器人
            pose = [x, y, z, rx, ry, rz]
            self.client.HRIF_StartServo(self.box_id, self.rbt_id, dServoTime, dLookaheadTime)
            self.move_arm(pose, dServoTime)
            self.current_pose = self.client.read_pos()
            x, y, z = self.current_pose[0], self.current_pose[1], self.current_pose[2]
            rx, ry, rz = self.current_pose[3], self.current_pose[4], self.current_pose[5]

            # 打印当前坐标
            print(f"Current Pose: X={x}, Y={y}, Z={z}, Rx={rx}, Ry={ry}, Rz={rz}")

        # 销毁窗口
        cv2.destroyAllWindows()


def main():
    controller = RobotController()
    controller.control_arm_with_cv2()  # 启动cv2控制


# 启动程序
if __name__ == "__main__":
    main()
