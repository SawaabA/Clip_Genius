import argparse
from Processing.Processor import PROCESS_VIDEO
from Processing.Processor import PROCESS_FILE
from Processing.Processor import PROCESS_FILE_MULTI_THREAD
from datetime import datetime
import os


def main(video_path, function_name, mode):
    start = datetime.now()
    if function_name == "PROCESS_VIDEO":
        PROCESS_VIDEO(video_path)
    elif function_name == "PROCESS_FILE":
        PROCESS_FILE(video_path, mode)
    elif function_name == "PROCESS_FILE_MULTI_THREAD":
        PROCESS_FILE_MULTI_THREAD(video_path)
    else:
        print(f"Error: Unknown function '{function_name}'")
    print(f"TIME ELAPSED : {(datetime.now()-start).total_seconds():.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument(
        "--function",
        type=str,
        choices=["PROCESS_VIDEO", "PROCESS_FILE", "PROCESS_FILE_MULTI_THREAD"],
        required=True,
        help="Function to run: PROCESS_VIDEO, PROCESS_FILE, or PROCESS_FILE_MULTI_THREAD",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    main(args.video_path, args.function, args.debug)
