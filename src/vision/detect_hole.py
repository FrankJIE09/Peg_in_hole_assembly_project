import cv2
import numpy as np

def detect_hole(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            return int(x), int(y)
    return None

if __name__ == "__main__":
    image = cv2.imread("test_image.jpg")
    hole_center = detect_hole(image)
    print(f"Hole detected at: {hole_center}")