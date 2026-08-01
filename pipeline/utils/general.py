import torch
import numpy as np
import subprocess
from pathlib import Path
import cv2
import os

def clone_repo(repo_url, destination):
    destination = Path(destination)
    os.makedirs(destination, exist_ok=True)
    subprocess.run(
        ["git", "clone", repo_url, str(destination)],
        check=True,
    )

def resize_image(frame, data_factor):
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (w // data_factor, h // data_factor),
        interpolation=cv2.INTER_AREA
    )
