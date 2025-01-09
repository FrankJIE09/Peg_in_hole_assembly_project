import numpy as np
import time


# 假设这是一个控制机械臂速度的接口函数
def set_arm_velocity(vx, vy, vz, wx, wy, wz):
    """
    设置机械臂的速度命令，单位为米每秒（m/s）和弧度每秒（rad/s）。

    :param vx: x 方向线性速度 (m/s)
    :param vy: y 方向线性速度 (m/s)
    :param vz: z 方向线性速度 (m/s)
    :param wx: x 方向角速度 (rad/s)
    :param wy: y 方向角速度 (rad/s)
    :param wz: z 方向角速度 (rad/s)
    """
    # 这里应该是控制机械臂的实际命令，例如通过 ROS 或其他控制系统
    print(f"设置速度：vx={vx}, vy={vy}, vz={vz}, wx={wx}, wy={wy}, wz={wz}")
    # 真实代码应发送速度命令给机械臂控制器
    pass  # 假设控制命令发送代码


# 将末端执行器坐标系的速度转换到基坐标系
def convert_velocity_to_base(velocity_tool, R_tool_to_base):
    """
    将末端执行器坐标系的速度转换到基坐标系。

    :param velocity_tool: 末端执行器坐标系下的速度 [vx, vy, vz, wx, wy, wz]
    :param R_tool_to_base: 从末端执行器坐标系到基坐标系的旋转矩阵（3x3）
    :return: 基坐标系下的速度
    """
    # 提取速度的线性部分
    linear_velocity_tool = velocity_tool[:3]

    # 使用旋转矩阵转换线性速度
    linear_velocity_base = np.dot(R_tool_to_base, linear_velocity_tool)

    # 角速度部分（如果没有旋转，角速度不变）
    angular_velocity_base = velocity_tool[3:]

    # 返回转换后的速度，线性和角速度
    velocity_base = np.concatenate([linear_velocity_base, angular_velocity_base])
    return velocity_base


def move_arm_z_direction(distance, speed, duration, R_tool_to_base):
    """
    控制机械臂沿末端执行器z轴方向移动特定距离。

    :param distance: 目标位移（米），例如 0.01 代表 10mm
    :param speed: 机械臂的速度（米/秒）
    :param duration: 控制时间（秒），例如 1秒
    :param R_tool_to_base: 末端执行器坐标系到基坐标系的旋转矩阵（3x3）
    """
    # 末端执行器坐标系下的速度命令：沿z轴方向移动
    velocity_tool = np.array([0, 0, speed, 0, 0, 0])  # 末端沿z轴方向速度

    # 将速度从末端执行器坐标系转换到基坐标系
    velocity_base = convert_velocity_to_base(velocity_tool, R_tool_to_base)

    # 发送转换后的速度命令到机械臂控制器
    set_arm_velocity(*velocity_base)

    # 控制机械臂的持续时间
    time.sleep(duration)  # 维持此速度一段时间

    print(f"机械臂已沿末端z轴方向移动 {distance * 1000} mm（{duration}秒内）")


# 假设的旋转矩阵（3x3矩阵），表示末端执行器坐标系相对于基坐标系的旋转
# 这个旋转矩阵会根据实际的末端执行器方向和姿态来确定
R_tool_to_base = np.array([[1, 0, 0],  # 假设末端z轴与基坐标系z轴对齐
                           [0, 1, 0],
                           [0, 0, 1]])

# 设置参数
distance_to_move = 0.01  # 目标位移：10mm = 0.01米
speed = 0.01  # 速度：10mm/s = 0.01米/秒
duration = distance_to_move / speed  # 控制时间：目标位移 / 速度

# 调用函数让机械臂沿末端z轴方向移动
move_arm_z_direction(distance_to_move, speed, duration, R_tool_to_base)
