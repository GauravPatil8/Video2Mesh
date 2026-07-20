import cv2
import os
import shutil
from pathlib import Path
import math
from .log_utils import log_execution
from typing import Optional
from .general_utils import resize_image

@log_execution
def load_frames(*, video_path: Optional[Path], images_path: Optional[Path], output_dir: Path, fps: int = 3, data_factor: int = 4):
    """
        Loads frames from video files and Image folders. 
        "fps" extracts fixed number of frames per second.        
    """

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if video_path is not None:
        video_path = Path(video_path)

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = video.get(cv2.CAP_PROP_FPS)
        frame_skip = math.floor(video_fps / fps)

        frame_idx = 0
        saved = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame = resize_image(frame=frame, data_factor=data_factor)
                file_path = os.path.join(output_dir, f"frame_{saved:06d}.png")
                cv2.imwrite(file_path, frame)
                saved += 1
            frame_idx += 1

        video.release()
    else:
        for i, images in enumerate(os.listdir(images_path)):
            image_path = os.path.join(images_path, images)
            dest_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            if data_factor > 1:
                image = cv2.imread(image_path)
                image = resize_image(frame=image, data_factor=data_factor)
                cv2.imwrite(dest_path, image)
            else:
                shutil.move(image_path, dest_path)


def _test_frame_loader():
    vid_file = "path/to/you/video"
    out_path = os.path.join("scene", "images")
    os.makedirs(out_path, exist_ok=True)
    load_frames(video_path=Path(vid_file), output_dir=Path(out_path))

if __name__ == '__main__':
    _test_frame_loader()