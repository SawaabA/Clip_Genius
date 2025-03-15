from ImageProcessingFunctions import fetch_score_coords, SEGMENT_SIZE
from MultiProcessing import analayze_segments_with_threads
from PreProcessing import split_video
from time import sleep

TEMPFOLDER = "output_videos"


def PROCESS_FILE(filepath, tempfolder=TEMPFOLDER):
    cords = fetch_score_coords(filepath)
    print(cords)
    split_video(filepath, SEGMENT_SIZE, tempfolder, "segments_%03d.mp4")
    sleep(1)
    results = analayze_segments_with_threads(tempfolder, cords)
    print(f"\nCompleted\n\tTotal Shots Detected {len(results)}")
    return results


if __name__ == "__main__":
    VIDEO_PATH = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/Test1.mov"
    )
    print(PROCESS_FILE(VIDEO_PATH))
