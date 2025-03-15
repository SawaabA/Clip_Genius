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


FRAME_SKIP_SECONDS = 3


def process_video(file_path: str):
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


def analyze_segment(file_path, score_coords, segment_number, masterfile):
    """
    -------------------------------------------------------
    Analyzes a video segment to detect changes in the score value and displays the video with real-time score detection.
    Use: analyze_segment(file_path, score_coords, segment_number, masterfile)
    -------------------------------------------------------
    Parameters:
        file_path - the path to the video file (str)
        score_coords - a list of two tuples,
        segment_number - the segment number being analyzed (int)
        masterfile - the name or identifier of the master file being processed (str)
    Returns:
        None
    -------------------------------------------------------
    """
    video = cv.VideoCapture(file_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    try:
        fps = int(video.get(cv.CAP_PROP_FPS))
        frame_interval = fps * FRAME_SKIP_SECONDS
        frame_count = 0
        prev_value = None

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv.resize(frame, FRAME_SIZE)

            if frame_count % frame_interval == 0:
                curr_value = get_score_value(frame, score_coords)
                if prev_value is None:
                    prev_value = curr_value
                elif curr_value != prev_value:
                    print(f"Basket!!")
                    prev_value = curr_value
                print(curr_value)
            cv.imshow("Scoreboard Detection", frame)
            frame_count += 1

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test3.mov"
    )
    cords = fetch_score_coords(VIDEO_PATH)
    print(cords)
    print("Stage 2")
    analyze_segment(VIDEO_PATH, cords, 1, "this")
    # cords = process_video(VIDEO_PATH)
