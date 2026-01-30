
from moviepy.editor import VideoFileClip

# Path to the input video file
input_video_path = r"C:\Users\dlab\rishika_sim\Bayleef_mp4.mp4"

# Path to the output cropped video file
output_video_path = r"C:\Users\dlab\Downloads\Bayleef_20250209_sess_2_trimmed.mp4"

# Load the video
video = VideoFileClip(input_video_path)

# Crop the video until 17 seconds
cropped_video = video.subclip((13*60)+53, (14*60)+5)

# Save the cropped video
cropped_video.write_videofile(output_video_path, codec="libx264")

# Close the video clips
video.close()
cropped_video.close()