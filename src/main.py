from vision.detect_hole import detect_hole
from control.admittance_control import AdmittanceControl
from utils.transformations import calculate_offset
import cv2
import numpy as np

def main():
    camera = cv2.VideoCapture(0)
    control = AdmittanceControl([1.0]*6, [10.0]*6, [100.0]*6)

    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image")
        return
    hole_center = detect_hole(frame)
    print(f"Hole detected at: {hole_center}")

    axis_pos = np.array([0.0, 0.0, 0.0])
    hole_pos = np.array([hole_center[0], hole_center[1], 0.0])
    offset = calculate_offset(axis_pos, hole_pos)

    print(f"Offset: {offset}")

    for _ in range(100):
        force = np.random.random(6) - 0.5
        position = control.update(force, dt=0.01)
        print(f"New position: {position}")

if __name__ == "__main__":
    main()