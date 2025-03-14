import cv2 as cv
import pytesseract
from pytesseract import Output
import numpy as np

config = r"--psm 11 --oem 3"
COLOR = (0, 255, 0)


def detect_score_board(image):
    try:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, threshold1=50, threshold2=200, apertureSize=3)
        lines = cv.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,  # Angle resolution in radians
            threshold=100,
            minLineLength=150,
            maxLineGap=10,
        )
        accepted_cords = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 > image.shape[0] * 0.75 and y1 < image.shape[0] * 0.90:

                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if abs(angle) < 10 or abs(angle - 180) < 10:
                        # cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        accepted_cords.append((x1, y1, x2, y2))

        accepted_cords = np.array(accepted_cords, dtype=np.int32)
        min_values = np.min(accepted_cords, axis=0)
        max_values = np.max(accepted_cords, axis=0)

        return (min_values[0], min_values[1], max_values[2], max_values[3])
    except Exception as e:
        print(f"Error Processing Frames: {e}")
        return (0, 0, image.shape[0], image.shape[1])


file_path = "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test3.mov"
video = cv.VideoCapture(file_path)

if not video.isOpened():
    print("Could Not Find Video")
    exit()


while video.isOpened():
    ret, frame = video.read()
    frame = cv.resize(frame, (1000, 1000))
    x1, y1, x2, y2 = detect_score_board(frame)
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord("q"):
        break
