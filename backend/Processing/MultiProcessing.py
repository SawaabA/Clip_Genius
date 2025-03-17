"""-------------------------------------------------------
CLIP GENIUS: MultiProcess
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    threading, queue
Version:  1.0.8
__updated__ = Fri Mar 14 2025
-------------------------------------------------------
"""

from .ImageProcessingFunctions import analyze_segment
from .PreProcessing import get_video_duration
import threading
import queue
import os


def thread_wrapper(segment_path, score_coords, duration, result_queue):
    """
    -------------------------------------------------------
    Analyzes a video segment in a separate thread.
    -------------------------------------------------------
    Parameters:
        segment_path - the path of the video segment to analyze (str)
        score_coords - coordinates (x, y, w, h)
        duration - cumulative duration of previous segments (float)
        result_queue - queue to store the results (Queue)
    -------------------------------------------------------
    """
    result = analyze_segment(segment_path, score_coords, duration)
    result_queue.put(result)


def analayze_segments_with_threads(segment_folder, score_coords):
    """
    -------------------------------------------------------
    Analyzes video segments in parallel using multiple threads to detect score changes.
    -------------------------------------------------------
    Parameters:
        segment_folder - the directory containing the video segments to analyze (str)
        score_coords - coordinates (x, y, w, h)
    Returns:
        results - a list of results from analyzing the segments (list)
    -------------------------------------------------------
    """
    result_queue = queue.Queue()
    threads = []
    duration = 0

    segments = os.listdir(segment_folder)
    for i, segment in enumerate(segments):
        segment = os.path.join(segment_folder, segment)
        print(f"{i} - {segment} @ {duration}")
        thread = threading.Thread(
            target=thread_wrapper,
            args=(segment, score_coords, duration, result_queue),
        )
        duration += round(get_video_duration(segment), 2)

        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = []
    while not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

    return results
