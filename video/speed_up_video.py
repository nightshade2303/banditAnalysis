from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import speedx

# Path to the input video file
input_video_path = r"C:\Users\dlab\Downloads\20250313_190749_1.mp4"

# Path to the output sped-up video file
output_video_path = r"C:\Users\dlab\Downloads\20250313_190749_1_spedup.mp4"

# Load the video
video = VideoFileClip(input_video_path)

# Speed up the video by 2x
sped_up_video = speedx(video, factor=2)

# Save the sped-up video
sped_up_video.write_videofile(output_video_path, codec="libx264")

# Close the video clips
video.close()
sped_up_video.close()