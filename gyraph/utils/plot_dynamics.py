"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from .utils import cv2, os


def create_video_from_images(
    path_to_misc: str, filename: str, outputname: str, length: int
):
    # Get all image filenames in order
    image_files = [
        os.path.join(path_to_misc, f"{filename}{i}.png") for i in range(length)
    ]
    # Read the first image to get the frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    video_path = os.path.join(path_to_misc, outputname + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, 20, (width, height))

    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            video.write(img)

    video.release()
