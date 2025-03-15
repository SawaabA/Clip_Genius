"""-------------------------------------------------------
Get Score Card: Get Coordinates of Score Card
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,openCV
Version:  1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------
"""

import cv2 as cv
import numpy as np
from ImageProcessingFunctions import *
from test_service2 import *

# Constants
FRAME_SIZE = (1000, 1000)
COLOR = (0, 255, 0)


def is_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    r1 = x1 + w1
    b1 = y1 + h1
    r2 = x2 + w2
    b2 = y2 + h2

    if (r1 <= x2) or (r2 <= x1) or (b1 <= y2) or (b2 <= y1):
        return False
    else:
        return True


def plotscores_on_images(image, scores):
    for s in scores:
        image = cv.rectangle(
            image, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), COLOR, thickness=2
        )
    return image


def convert_to_abs_coordinates(x1, y1, scores):
    if not len(scores) == 2:
        return False
    absolute_coordinates = []
    if not is_overlap(scores[0], scores[1]):
        for s in scores:
            absolute_coordinates.append((s[0] + x1, s[1] + y1, s[2], s[3]))
        return absolute_coordinates
    return None


def process_video(file_path: str):
    """Processes video and detects scoreboard in frames."""
    video = cv.VideoCapture(file_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    while video.isOpened():

        ret, frame = video.read()
        if not ret:
            break

        frame = cv.resize(frame, FRAME_SIZE)
        x1, y1, x2, y2 = get_scoreboard_coordinates(frame)
        # cv.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)

        extracted_image = extract_scoreboard(frame, x1, y1, x2, y2)
        score_cords = find_scores(extracted_image)
        abs_cords = convert_to_abs_coordinates(x1, y1, score_cords)
        if abs_cords:
            frame = plotscores_on_images(frame, abs_cords)

        # TIME STAMP
        timestamp = video.get(cv.CAP_PROP_POS_MSEC) / 1000
        timestamp_text = f"Time: {timestamp:.2f}s"
        cv.putText(
            frame,
            timestamp_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow("Scoreboard Detection", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test1.mov"
    )
    process_video(VIDEO_PATH)
    print("Stream Ended SUc")
