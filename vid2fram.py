import os
import cv2


def extract_frames_from_video(video_path, output_folder):
    """
    Extracts frames from a video file and saves them as individual images.

    Parameters:
    video_path (str): Path to the video file.
    output_folder (str): Path to the folder where extracted frames will be saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_path = os.path.join(output_folder, f'{frame_count}.png')
        cv2.imwrite(output_path, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return frame_count


def process_all_videos(video_folder, frame_output_folder):
    """
    Processes all videos in a folder to extract frames.

    Parameters:
    video_folder (str): Path to the folder containing video files.
    frame_output_folder (str): Path to the folder to save extracted frames.
    """
    video_files = os.listdir(video_folder)
    total_frames = 0
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        frames = extract_frames_from_video(video_path, frame_output_folder)
        total_frames += frames
    return total_frames


if __name__ == '__main__':
    VIDEO_FOLDER_PATH = 'data/out/'
    FRAME_SAVE_PATH = 'data/frames/'

    total_extracted_frames = process_all_videos(VIDEO_FOLDER_PATH, FRAME_SAVE_PATH)
    print(f"Total number of frames extracted: {total_extracted_frames}")
