import sys
import time
import ffmpeg  # This now correctly refers to `ffmpeg-python`

def extract_highlights(video_path, length):
    """ Extract highlights from a video using FFmpeg. """
    time.sleep(5)  # Simulate a 5-second processing delay

    duration = int(length) * 60  # Convert minutes to seconds
    output_path = "game_highlight.mp4"

    try:
        # Ensure FFmpeg is using the correct syntax
        (
            ffmpeg
            .input(video_path, ss=120, t=duration)
            .output(output_path, vcodec="libx264", acodec="aac")
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())  # Print the full FFmpeg error output

    return output_path

if __name__ == "__main__":
    video_path = sys.argv[1]
    highlight_length = sys.argv[2]
    transcript_lang = sys.argv[3]

    extract_highlights(video_path, highlight_length)
    print("Processing complete.")