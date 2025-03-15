"""-------------------------------------------------------
CLIP GENIUS: Image Pre-Processing
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,openCV,pytesseract
Version:  1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------
"""

import subprocess
import os


def split_video(input_file, segment_time, output_folder, output_pattern):
    """
    -------------------------------------------------------
    Splits a video file into smaller segments of a specified duration using FFmpeg.
    Use: split_video(input_file, segment_time, output_folder, output_pattern)
    -------------------------------------------------------
    Parameters:
        input_file - the path to the input video file (str)
        segment_time - the duration of each segment in seconds (int or float)
        output_folder - the directory where the output segments will be saved (str)
        output_pattern - the naming pattern for the output segments (str, e.g., "output_%03d.mp4")
    Returns:
        None
    -------------------------------------------------------
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, output_pattern)

        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-c",
            "copy",
            "-map",
            "0",
            "-segment_time",
            str(segment_time),
            "-f",
            "segment",
            output_path,
        ]

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Video split successfully:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)


if __name__ == "__main__":
    TESTFILE = (
        "/Users/jashan/projects/LaurierAnalitics2025/tests/testImages/FinalOutput.mp4"
    )

    split_video(TESTFILE, 10, "output_videos", "output_%03d.mp4")
