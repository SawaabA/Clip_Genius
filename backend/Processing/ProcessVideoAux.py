"""-------------------------------------------------------
CLIP GENIUS: Process Video
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,openCV,pytesseract
Version:  1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------
"""

from ImageProcessingFunctions import *


def process_video(file_path: str):
    """Processes video and detects scoreboard in frames."""
    video = cv.VideoCapture(file_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    abs_cords = None
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = cv.resize(frame, FRAME_SIZE)

        if not abs_cords:
            abs_cords = extract_scores_location_aux(frame)

        else:
            frame = plotscores_on_images(frame, abs_cords)

        timestamp = video.get(cv.CAP_PROP_POS_MSEC)
        add_timestamp_to_frame(frame, timestamp)
        cv.imshow("Scoreboard Detection", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv.destroyAllWindows()


def get_score_value(frame: np.ndarray, coords: list[tuple[int, int, int, int]]) -> str:
    x1, y1, w1, h1 = coords[0]
    x2, y2, w2, h2 = coords[1]
    score_region = frame[y1 - 5 : y2 + h2 + 5, x1 - 5 : x2 + w2 + 5]

    gray = cv.cvtColor(score_region, cv.COLOR_BGR2GRAY)

    gray = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)

    gray = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    custom_config = r"--oem 3 --psm 6"
    extracted_text = pytesseract.image_to_string(binary, config=custom_config).strip()

    print(extracted_text)
    return binary


def analyze_segment(file_path, score_coords, segment_number, masterfile):
    video = cv.VideoCapture(file_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    curr_value = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = cv.resize(frame, FRAME_SIZE)
        frame = get_score_value(frame, score_coords)
        cv.imshow("Scoreboard Detection", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv.destroyAllWindows()
    return None


if __name__ == "__main__":
    VIDEO_PATH = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test1.mov"
    )
    cords = fetch_score_coords(VIDEO_PATH)
    print(cords)
    print("Stage 2")
    analyze_segment(VIDEO_PATH, cords, 1, "this")
    # cords = process_video(VIDEO_PATH)
    print(cords)
