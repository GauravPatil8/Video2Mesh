# Video2Mesh

A minimal pipeline that turns a video, image set, or existing COLMAP scene into a 3D mesh.

## What it does

The workflow is:

1. Load input frames from video or images.
2. Run COLMAP structure-from-motion.
3. Train a 3D Gaussian Splatting model.
4. Extract a mesh with Poisson reconstruction.

## Requirements

- Python 3.10–3.12

## Installation
Clone Repository:

```bash
git clone https://github.com/GauravPatil8/Video2Mesh.git
%cd Video2Mesh
```
Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python orchestrator.py --video path/to/video.mp4 --output_dir output
```

## Inputs

You can provide exactly one of these:

- `--video path/to/video.mp4`
- `--images path/to/images/`
- `--scene path/to/colmap_scene/`

For a complete list of command-line arguments and their descriptions, run:
```bash
python orchestrator.py -h
```
