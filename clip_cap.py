import cv2
import os
import xml.etree.ElementTree as ET

# Constants for file paths and extensions
XML_FOLDER_PATH = './data/label/'
VIDEO_FOLDER_PATH = './data/vid/'
OUTPUT_FOLDER_PATH = './data/out/'
VIDEO_EXTENSION = '.avi'
XML_EXTENSION = '.xml'


def parse_xml_behaviors(xml_path):
    """
    Parses the XML file to extract behavior IDs and corresponding timeframes.

    Parameters:
    xml_path (str): Path to the XML file.

    Returns:
    list of tuple: List containing tuples of behavior ID and its timeframe.
    """
    behaviors = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for behavior in root.findall('.//behaviour'):
        b_id = behavior.attrib['id']
        time = [time_elem.text for time_elem in behavior.findall('time')]
        behaviors.append((b_id, time))
    return behaviors


def clip_video(video_path, start_time, end_time, output_path):
    """
    Clips a portion of the video based on start and end time.

    Parameters:
    video_path (str): Path to the input video file.
    start_time (int): Starting time in seconds.
    end_time (int): Ending time in seconds.
    output_path (str): Path to save the clipped video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int((end_time + 1) * fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= start_frame:
            out.write(frame)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def convert_time_string_to_seconds(time_str):
    """
    Converts a time string to seconds.

    Parameters:
    time_str (str): Time string in 'MMSS' or 'SS' format.

    Returns:
    int: Time in seconds.
    """
    if len(time_str) <= 2:
        return int(time_str)
    else:
        minutes = time_str[:-2]
        seconds = time_str[-2:]
        return int(minutes) * 60 + int(seconds)


def process_videos():
    """
    Main function to process each video based on its XML annotations.
    """
    video_files = os.listdir(VIDEO_FOLDER_PATH)

    for video_file in video_files:
        xml_file = video_file.replace(VIDEO_EXTENSION, XML_EXTENSION)
        xml_path = os.path.join(XML_FOLDER_PATH, xml_file)
        behaviors = parse_xml_behaviors(xml_path)

        for behavior_id, times in behaviors:
            start_time = convert_time_string_to_seconds(times[0]) - 3
            end_time = convert_time_string_to_seconds(times[1]) + 3
            output_filename = f"{video_file.replace(VIDEO_EXTENSION, '')}_{behavior_id}{VIDEO_EXTENSION}"
            output_path = os.path.join(OUTPUT_FOLDER_PATH, output_filename)

            video_path = os.path.join(VIDEO_FOLDER_PATH, video_file)
            clip_video(video_path, start_time, end_time, output_path)


if __name__ == '__main__':
    process_videos()
