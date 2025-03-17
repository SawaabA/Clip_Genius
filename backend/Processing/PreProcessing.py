"""-------------------------------------------------------
CLIP GENIUS: Image Pre-Processing
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,openCV,pytesseract
Version:  1.0.8
__updated__ = Fri Mar 15 2025
-------------------------------------------------------
"""

import subprocess
import time
import os

# CONSTANTS
TEMPFOLDER = "TEMPFOLDER"
PRE_TIME = 10
POST_TIME = 6
import json


def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def get_video_duration(video_path):
    """
    -------------------------------------------------------
    Retrieves the duration of a video file using FFprobe.
    Use: duration = get_video_duration(video_path)
    -------------------------------------------------------
    Parameters:
        video_path - the path to the video file (str)
    Returns:
        duration - the duration of the video in seconds (float)
    -------------------------------------------------------
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    metadata = json.loads(result.stdout)
    duration = float(metadata["format"]["duration"])
    return duration


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
            "-f",
            "segment",
            "-segment_time",
            str(segment_time),
            "-force_key_frames",
            f"expr:gte(t,n_forced*{segment_time})",
            "-reset_timestamps",
            "1",
            output_path,
        ]

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Video split successfully:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)


def create_clip(
    input_file,
    start_time,
    end_time,
    output_folder,
    output_filename,
):
    """
    -------------------------------------------------------
    Creates a final video clip by extracting a segment from an input video file using FFmpeg.
    Use: create_clip(input_file, start_time, end_time, output_folder, output_pattern)
    -------------------------------------------------------
    Parameters:
        input_file - the path to the input video file (str)
        start_time - the start time of the segment to extract (str, in HH:MM:SS format)
        end_time - the end time of the segment to extract (str, in HH:MM:SS format)
        output_folder - the directory where the output clip will be saved (str)
        output_filename - the naming pattern for the output clip
    Returns:
        None
    -------------------------------------------------------
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, output_filename)

    command = [
        "ffmpeg",
        "-i",
        input_file,
        "-ss",
        start_time,
        "-to",
        end_time,
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(command, check=True)


def process_results(
    source_file,
    results,
    output_folder="OUTPUT",
    pre_time=PRE_TIME,
    post_time=POST_TIME,
):
    """
    -------------------------------------------------------
    Processes a list of timestamps to create video clips around each timestamp using a source video file.
    Use: process_results(source_file, results, output_folder, pre_time, post_time)
    -------------------------------------------------------
    Parameters:
        source_file - the path to the source video file (str)
        results - a list of timestamps (in milliseconds) to create clips around (list of int)
        output_folder - the directory where the output clips will be saved (str, default="OUTPUT")
        pre_time - the time (in seconds) to include before each timestamp (float or int, default=PRE_TIME)
        post_time - the time (in seconds) to include after each timestamp (float or int, default=POST_TIME)
    Returns:
        None
    -------------------------------------------------------
    """
    for i, result in enumerate(results):
        start_time = format_time(max(result - pre_time, 0))
        end_time = format_time(result + post_time)
        create_clip(
            source_file, start_time, end_time, output_folder, f"Final_clip_{i}.mp4"
        )
