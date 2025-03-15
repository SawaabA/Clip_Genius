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
    video = cv.VideoCapture(file_path)
    points = []
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
            x1, y1, x2, y2 = get_scoreboard_coordinates(frame)
            if x1:
                extracted_image = extract_scoreboard(frame, x1, y1, x2, y2)
                # cv.imshow("Scoreboard Detection", extracted_image)
                score_cords = find_scores(extracted_image)
                abs_cords = convert_to_abs_coordinates(x1, y1, score_cords)
        else:
            frame = plotscores_on_images(frame, abs_cords)
            cv.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        timestamp = video.get(cv.CAP_PROP_POS_MSEC)
        add_timestamp_to_frame(frame, timestamp)
        cv.imshow("Scoreboard Detection", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test2.mov"
    )
    cords = fetch_score_coords(VIDEO_PATH)
    print(cords)
    print("Stage 2")
    results = analyze_segment(VIDEO_PATH, cords, 1, "this")
    print(results)
    # cords = process_video(VIDEO_PATH)
