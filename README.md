# 轴孔装配项目

本项目旨在实现一个机器人装配任务，其中轴（插销）需要插入孔内。系统利用以下技术：
- **力传感器**：实时获取末端受力反馈。
- **导纳控制**：确保机器人运动的柔顺性和平稳性。
- **计算机视觉**：定位孔的位置并引导机器人完成初始对准。

---

## 项目结构

```plaintext
peg_hole_assembly_project/
├── config/                     # 配置文件
│   ├── robot_config.yaml       # 机械臂配置
│   ├── camera_config.yaml      # 相机配置
│   └── control_params.yaml     # 导纳控制参数
├── data/                       # 数据文件
│   ├── calibration/            # 校准数据
│   └── logs/                   # 实验日志
├── src/                        # 源代码
│   ├── vision/                 # 视觉模块
│   │   ├── detect_hole.py      # 孔位检测
│   │   └── visualize.py        # 可视化工具
│   ├── control/                # 控制模块
│   │   ├── admittance_control.py # 导纳控制实现
│   │   ├── move_robot.py       # 机械臂运动控制
│   │   └── force_feedback.py   # 力传感器读取与处理
│   ├── utils/                  # 工具模块
│   │   ├── transformations.py  # 坐标变换工具
│   │   └── logging_utils.py    # 日志工具
│   └── main.py                 # 主程序入口
├── scripts/                    # 辅助脚本
│   ├── calibrate_camera.py     # 相机标定脚本
│   ├── calibrate_force.py      # 力传感器校准脚本
│   └── generate_report.py      # 实验报告生成脚本
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明
