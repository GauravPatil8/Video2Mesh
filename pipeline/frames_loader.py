import cv2
import os
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, frame_skip: int = 3) -> dict:
    """
        Extracts frames from video file and stored them in output_dir. 
        "frame_skip" selectes frames after set interval of images.        
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            out_path = os.path.join(output_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    video.release()

def _test_frame_loader():
    vid_file = os.path.join("test_vid", "car.mp4")
    images_path = os.path.join("scene", "images")
    os.makedirs(images_path, exist_ok=True)
    extract_frames(Path(vid_file), Path(images_path))

if __name__ == '__main__':
    _test_frame_loader()