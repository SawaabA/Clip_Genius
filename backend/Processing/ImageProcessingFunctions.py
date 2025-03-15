"""-------------------------------------------------------
CLIP GENIUS: Image Processing Functions
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,openCV,pytesseract
Version:  1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------
"""

# IMPORTS
from pytesseract import Output
import pytesseract
import numpy as np
import cv2 as cv

# CONSTANTS
THRESHOLD1 = 50  # for canny edge dector
THRESHOLD2 = 200

LINE_THRESHOLD = 100  # For Hough Line Transform
MIN_LINE_LENGTH = 150
MAX_LINE_GAP = 10
HORIZONTAL_ANGLE_THRESHOLD = 10

SCOREBOARD_Y_RANGE = (0.75, 0.98)  # Expected Area

CONFIG = r"--psm 11 --oem 3"  # for OCR
CONFIDENCE_THRESHOLD = 10


COLOR = (0, 255, 0)


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
        return None, None, None, None


def extract_scoreboard(image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    return image[y1:y2, x1:x2]


def add_timestamp_to_frame(frame, timestamp):
    timestamp_text = f"Time: {timestamp/ 1000:.2f}s"
    cv.putText(
        frame,
        timestamp_text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR,
        2,
        cv.LINE_AA,
    )


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


def find_scores(image: np.ndarray, confidence_threshold: int = CONFIDENCE_THRESHOLD):
    annotated_image = image.copy()

    # Perform OCR
    data = pytesseract.image_to_data(
        annotated_image, config=CONFIG, output_type=Output.DICT
    )
    cords = []
    for i in range(len(data["text"])):
        text, conf = data["text"][i], float(data["conf"][i])
        if text.strip() == "0" and conf > confidence_threshold:
            cords.append(
                (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
            )

    return cords


def extract_scores_location_aux(frame):
    x1, y1, x2, y2 = get_scoreboard_coordinates(frame)
    if x1:
        extracted_image = extract_scoreboard(frame, x1, y1, x2, y2)
        score_cords = find_scores(extracted_image)
        # extracted_image = plotscores_on_images(extracted_image, score_cords)
        # cv.imshow("Scoreboard Detection", extracted_image)
        abs_cords = convert_to_abs_coordinates(x1, y1, score_cords)
        return abs_cords
    return None
