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
import re
import sys

# CONSTANTS
DEBUG = False

SEGMENT_SIZE = 15

COLOR = (0, 255, 0)
FRAME_SIZE = (1000, 1000)

THRESHOLD1 = 50  # for canny edge dector
THRESHOLD2 = 200

LINE_THRESHOLD = 100  # For Hough Line Transform
MIN_LINE_LENGTH = 150
MAX_LINE_GAP = 10
HORIZONTAL_ANGLE_THRESHOLD = 10

SCOREBOARD_Y_RANGE = (0.75, 0.98)  # Expected Area

CONFIG = r"--psm 11 --oem 3"  # for OCR
CONFIDENCE_THRESHOLD = 60

SKIP_FRAME = 3


def get_scoreboard_coordinates(image: np.ndarray) -> tuple[int, int, int, int]:
    """
    -------------------------------------------------------
    Detects and extracts the bounding box coordinates of a scoreboard region in an image.
    Use: x_min, y_min, x_max, y_max = get_scoreboard_coordinates(image)
    -------------------------------------------------------
    Parameters:
        image - input image in BGR format as a NumPy array (np.ndarray)
    Returns:
        x_min - x-coordinate of the top-left corner of the bounding box (int)
        y_min - y-coordinate of the top-left corner of the bounding box (int)
        x_max - x-coordinate of the bottom-right corner of the bounding box (int)
        y_max - y-coordinate of the bottom-right corner of the bounding box (int)
        If no scoreboard is detected, returns the coordinates of the entire image.
        If an error occurs, returns (None, None, None, None).
    -------------------------------------------------------
    """
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
    """
    -------------------------------------------------------
    Extracts a region of interest (ROI) from an image using the provided bounding box coordinates.
    Use: scoreboard_image = extract_scoreboard(image, x1, y1, x2, y2)
    -------------------------------------------------------
    Parameters:
        image - input image as a NumPy array (np.ndarray)
        x1 - x-coordinate of the top-left corner of the bounding box (int)
        y1 - y-coordinate of the top-left corner of the bounding box (int)
        x2 - x-coordinate of the bottom-right corner of the bounding box (int)
        y2 - y-coordinate of the bottom-right corner of the bounding box (int)
    Returns:
        scoreboard_image - the extracted region of interest as a NumPy array (np.ndarray)
    -------------------------------------------------------
    """
    return image[y1:y2, x1:x2]


def add_timestamp_to_frame(frame, timestamp):
    """
    -------------------------------------------------------
    Adds a timestamp to a video frame.
    Use: add_timestamp_to_frame(frame, timestamp)
    -------------------------------------------------------
    Parameters:
        frame - the video frame as a NumPy array (np.ndarray)
        timestamp - the timestamp in milliseconds (int)
    Returns:
        None
    -------------------------------------------------------
    """
    timestamp_text = f"Time: {timestamp:.2f}s"
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
    """
    -------------------------------------------------------
    Checks if two rectangles overlap.
    Use: overlap = is_overlap(rect1, rect2)
    -------------------------------------------------------
    Parameters:
        rect1 - coordinates of the first rectangle as a tuple
        rect2 - coordinates of the second rectangle as a tuple
    Returns:
        overlap - True if the rectangles overlap, False otherwise (bool)
    -------------------------------------------------------
    """
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
    """
    -------------------------------------------------------
    Draws rectangles on an image based on the provided scores.
    Use: annotated_image = plotscores_on_images(image, scores)
    -------------------------------------------------------
    Parameters:
        image - the input image as a NumPy array (np.ndarray)
        scores - a list of rectangles, where each rectangle is represented as a tuple (x, y, width, height)
    Returns:
        annotated_image - the image with rectangles drawn on it (np.ndarray)
    -------------------------------------------------------
    """
    for s in scores:
        image = cv.rectangle(
            image, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), COLOR, thickness=2
        )
    return image


def find_scores(image: np.ndarray, confidence_threshold: int = 10):
    """
    -------------------------------------------------------
    Finds and extracts coordinates of detected scores ("0") in an image using OCR (Optical Character Recognition).
    Use: score_coordinates = find_scores(image, confidence_threshold)
    -------------------------------------------------------
    Parameters:
        image - the input image as a NumPy array (np.ndarray)
        confidence_threshold - the minimum confidence level for OCR detection (int, default = CONFIDENCE_THRESHOLD)
    Returns:
        cords - a list of tuples
    -------------------------------------------------------
    """
    data = pytesseract.image_to_data(image, config=CONFIG, output_type=Output.DICT)
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


def convert_to_abs_coordinates(x1, y1, scores):
    """
    -------------------------------------------------------
    Converts relative coordinates of scores to absolute coordinates based on a reference point (x1, y1).
    Use: abs_coordinates = convert_to_abs_coordinates(x1, y1, scores)
    -------------------------------------------------------
    Parameters:
        x1 - the x-coordinate of the reference point (int)
        y1 - the y-coordinate of the reference point (int)
        scores - a list of two tuples
    Returns:
        absolute_coordinates - a list
        If the input list does not contain exactly two scores or if the scores overlap, returns None.
    -------------------------------------------------------
    """
    if not len(scores) == 2:
        return None
    absolute_coordinates = []
    if not is_overlap(scores[0], scores[1]):
        for s in scores:
            absolute_coordinates.append((s[0] + x1, s[1] + y1, s[2], s[3]))
        return absolute_coordinates
    return None


def extract_scores_location_aux(frame):
    """
    -------------------------------------------------------
    Extracts the absolute coordinates of scores from a frame by detecting the scoreboard region and performing OCR.
    Use: score_locations = extract_scores_location_aux(frame)
    -------------------------------------------------------
    Parameters:
        frame - the input frame as a NumPy array (np.ndarray)
    Returns:
        abs_cords - a list of tuples,
        If the scoreboard region is not detected or no valid scores are found, returns None.
    -------------------------------------------------------
    """
    x1, y1, x2, y2 = get_scoreboard_coordinates(frame)
    if x1:
        extracted_image = extract_scoreboard(frame, x1, y1, x2, y2)
        score_cords = find_scores(extracted_image)
        # extracted_image = plotscores_on_images(extracted_image, score_cords)
        # cv.imshow("Scoreboard Detection", extracted_image)
        abs_cords = convert_to_abs_coordinates(x1, y1, score_cords)
        return abs_cords
    return None


def fetch_score_coords(file_path: str):
    """
    -------------------------------------------------------
    Processes a video file to detect and extract the absolute coordinates of scores by analyzing frames.
    Use: score_coordinates = fetch_score_coords(file_path)
    -------------------------------------------------------
    Parameters:
        file_path - the path to the video file (str)
    Returns:
        abs_cords - a list of tuples, where each tuple contains the absolute coordinates of a score in the format (x, y, width, height) (list[tuple[int, int, int, int]])
        If no scores are detected or the video file cannot be opened, returns None.
    -------------------------------------------------------
    """
    video = cv.VideoCapture(file_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return None
    abs_cords = None
    while video.isOpened() and abs_cords is None:
        ret, frame = video.read()
        if not ret:
            break
        timestamp = video.get(cv.CAP_PROP_POS_MSEC)
        print(f"Searching For Score Box: {timestamp/1000:.3f}", end="\r")
        frame = cv.resize(frame, FRAME_SIZE)
        abs_cords = extract_scores_location_aux(frame)
    video.release()
    return abs_cords


def get_score_value(frame: np.ndarray, coords: list[tuple[int, int, int, int]]):
    """
    -------------------------------------------------------
    Extracts the numeric score value from a frame using OCR (Optical Character Recognition) based on the provided coordinates.
    Use: score_value = get_score_value(frame, coords)
    -------------------------------------------------------
    Parameters:
        frame - the input frame as a NumPy array (np.ndarray)
        coords - a list of two tuples
    Returns:
        score_value - the extracted numeric score value as an integer (int)
    -------------------------------------------------------
    """
    x1, y1, w1, h1 = coords[0]
    x2, y2, w2, h2 = coords[1]
    score_region = frame[y1 - 5 : y2 + h2 + 5, x1 - 5 : x2 + w2 + 5]

    gray = cv.cvtColor(score_region, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite("tempimage.png", binary)
    custom_config = r"--oem 3 --psm 6"
    extracted_text = pytesseract.image_to_string(binary, config=custom_config).strip()
    digits = "".join(re.findall(r"\d", extracted_text))

    if len(digits) == 0:
        return 0
    return int(digits)


def process_frame(frame, score_coords, prev_value, points, timestamp):
    """
    -------------------------------------------------------
    Processes a video frame to detect changes in the score value and records timestamps of score updates.
    Use: updated_value = process_frame(frame, score_coords, prev_value, points, timestamp)
    -------------------------------------------------------
    Parameters:
        frame - the current video frame to process (numpy.ndarray)
        score_coords - coordinates (x, y, w, h)
        prev_value - the previous score value
        points - a list to store timestamps when a score update (basket) is detected (list)
        timestamp - the current timestamp of the frame (float)
    Returns:
        updated_value - the updated score value after processing the current frame (int)
    -------------------------------------------------------
    """
    curr_value = get_score_value(frame, score_coords)
    if prev_value is None:
        prev_value = curr_value
    elif curr_value > prev_value:
        points.append(timestamp)
        print(f"Basket!! @ {timestamp:.3f} \t Total Baskets: {len(points)}")
        prev_value = curr_value

    if DEBUG:
        print(curr_value)
    return prev_value


def analyze_segment(file_path, score_coords, segment_number, debug=False):
    """
    -------------------------------------------------------
    Analyzes a video segment to detect changes in the score value and displays the video with real-time score detection.
    Use: analyze_segment(file_path, score_coords, segment_number, masterfile)
    -------------------------------------------------------
    Parameters:
        file_path - the path to the video file (str)
        score_coords - a list of two tuples,
        segment_number - the segment number being analyzed (int)
    Returns:
        None
    -------------------------------------------------------
    """
    points = []
    video = cv.VideoCapture(file_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    try:
        fps = int(video.get(cv.CAP_PROP_FPS))
        frame_interval = fps * SKIP_FRAME
        frame_count = 0
        prev_value = None

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv.resize(frame, FRAME_SIZE)
            timestamp = segment_number + (video.get(cv.CAP_PROP_POS_MSEC) / 1000)

            if frame_count % frame_interval == 0:
                prev_value = process_frame(
                    frame, score_coords, prev_value, points, timestamp
                )
            if debug:
                add_timestamp_to_frame(frame, timestamp)
                cv.imshow("Analyzing Match Footage", frame)
            frame_count += 1

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

            print(f"Analyzing Video {segment_number} - {timestamp:.2f}", end="\r")
            sys.stdout.flush()

    finally:
        video.release()
        cv.destroyAllWindows()
    return points
