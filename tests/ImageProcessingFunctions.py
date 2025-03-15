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

# Constants
THRESHOLD1 = 50
THRESHOLD2 = 200
LINE_THRESHOLD = 100
MIN_LINE_LENGTH = 150
MAX_LINE_GAP = 10
HORIZONTAL_ANGLE_THRESHOLD = 10
SCOREBOARD_Y_RANGE = (0.75, 0.90)


def get_scoreboard_coordinates(image: np.ndarray) -> tuple[int, int, int, int]:
    try:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(
            gray, threshold1=THRESHOLD1, threshold2=THRESHOLD2, apertureSize=3
        )

        lines = cv.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=LINE_THRESHOLD,
            minLineLength=MIN_LINE_LENGTH,
            maxLineGap=MAX_LINE_GAP,
        )

        accepted_coords = []

        if lines is not None:
            height = image.shape[0]
            min_y, max_y = int(height * SCOREBOARD_Y_RANGE[0]), int(
                height * SCOREBOARD_Y_RANGE[1]
            )

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                if min_y < y1 < max_y and (
                    abs(angle) < HORIZONTAL_ANGLE_THRESHOLD
                    or abs(angle - 180) < HORIZONTAL_ANGLE_THRESHOLD
                ):
                    accepted_coords.append((x1, y1, x2, y2))

        if not accepted_coords:
            return 0, 0, image.shape[1], image.shape[0]  # Full image fallback

        accepted_coords = np.array(accepted_coords, dtype=np.int32)
        x_min, y_min = np.min(accepted_coords, axis=0)[:2]
        x_max, y_max = np.max(accepted_coords, axis=0)[2:]

        if (y_max) in range(y_min - 100, y_min + 100):
            y_max = y_min + 80

        return x_min, y_min, x_max, y_max

    except Exception as e:
        print(f"Error Processing Frames: {e}")
        return 0, 0, image.shape[1], image.shape[0]


def extract_scoreboard(image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    return image[y1:y2, x1:x2]
