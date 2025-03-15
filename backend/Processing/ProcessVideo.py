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


# Constants
FRAME_SIZE = (1000, 1000)


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


if __name__ == "__main__":
    VIDEO_PATH = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test1.mov"
    )
    cords = process_video(VIDEO_PATH)
    print(cords)
    print("Stream Ended SUc")
