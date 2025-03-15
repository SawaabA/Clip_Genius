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


# Constants
FRAME_SIZE = (1000, 1000)
COLOR = (0, 255, 0)


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
        cv.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        ex = extract_scoreboard(frame, x1, y1, x2, y2)

        # TIME STAMP
        timestamp = video.get(cv.CAP_PROP_POS_MSEC) / 1000
        timestamp_text = f"Time: {timestamp:.2f}s"
        cv.putText(
            ex,
            timestamp_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv.LINE_AA,
        )

        cv.imshow("Scoreboard Detection", ex)

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
