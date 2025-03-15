from moviepy.video.io.VideoFileClip import VideoFileClip


def split_video(input_video, output_folder, clip_duration=10):
    # Load the video
    video = VideoFileClip(input_video)
    duration = video.duration

    # Split the video into smaller clips
    for i, start_time in enumerate(range(0, int(duration), clip_duration)):
        end_time = min(start_time + clip_duration, duration)
        clip = video.subclip(start_time, end_time)
        clip.write_videofile(f"{output_folder}/clip_{i}.mp4", codec="libx264")


# Example usage
split_video("game.mp4", "clips", clip_duration=10)
