from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading


def detect_circles(image, plot=False):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 HoughCircles 检测图像中的圆，使用指定的参数
    circles = cv2.HoughCircles(
        gray,  # 输入的灰度图像
        cv2.HOUGH_GRADIENT,  # 使用霍夫梯度法来检测圆
        dp=1.2,  # 累加器分辨率与原始图像分辨率的反比
        minDist=60,  # 圆心之间的最小距离
        param1=50,  # Canny 边缘检测的高阈值
        param2=20,  # 圆心检测的累加器阈值，越大越难检测到圆
        minRadius=30,  # 最小圆半径
        maxRadius=30  # 最大圆半径
    )

    # 如果检测到圆，处理结果
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # 将检测到的圆的参数转换为整数

        # 按照圆心的x坐标从小到大排序
        circles = sorted(circles, key=lambda c: c[0])

        # 如果plot为True，绘制圆和圆心
        def show_image(image):
            pil_image = Image.fromarray(image)
            pil_image.show()

        if plot:
            for (x, y, r) in circles:
                # 绘制圆形
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                # 绘制圆心的小矩形
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # 将图像从 BGR 转换为 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 使用线程来显示图像，避免阻塞
            display_thread = threading.Thread(target=show_image, args=(image_rgb,))
            display_thread.start()

        # 返回排序后的圆心坐标
        return [(x, y) for (x, y, r) in circles]

    else:
        print("No circles detected")
        return []


def main(image_path, plot=False):
    # 调用detect_circles函数来检测圆并返回圆心坐标
    image = cv2.imread(image_path)

    circles_coords = detect_circles(image, plot)

    # 输出检测到的圆心坐标
    if circles_coords:
        print("Detected circle centers (sorted by x-coordinate):")
        for coord in circles_coords:
            print(coord)
    else:
        print("No circles were detected.")


# 调用main函数
if __name__ == "__main__":
    image_path = './mask-shrink-0_20240914_091521.png'  # 输入图像路径
    main(image_path, plot=True)  # plot=True时显示检测结果，plot=False时只输出圆心坐标
