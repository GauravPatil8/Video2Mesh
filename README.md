# Video2Mesh

A minimal pipeline that turns a video, image set, or an existing COLMAP scene into a 3D mesh.

## What it does

The workflow is:

1. Load input frames from video or images.
2. Run COLMAP structure-from-motion.
3. Train a 3D Gaussian Splatting model.
4. Extract a mesh with Poisson reconstruction.

## Requirements

- Docker with NVIDIA Container Toolkit (recommended)
- Python 3.10–3.12 for local setup
- NVIDIA GPU with CUDA support for the full pipeline

## Docker setup (recommended)

Build the container image from the repository root:

```bash
docker build -t video2mesh .
```

Run the container with GPU access and mount your input/output folders:

```bash
docker run --gpus all -it \
  --name video2mesh \
  video2mesh
```

Inside the container, follow these steps:

1. Go to the project directory:

```bash
cd /workspace/Video2Mesh
```

2. Activate the conda environment:

```bash
conda activate sugar
```

3. Install the required Gaussian Splatting submodules:

```bash
pip install -e libs/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization
pip install -e libs/SuGaR/gaussian_splatting/submodules/simple-knn
```

4. Run the pipeline:

```bash
python orchestrator.py \
  --video /workspace/data/video.mp4 \
  --fps 3 \
  --data_factor 4 \
  --prompt "oxford book" \
  --output_dir /workspace/output
```

Run the pipeline:

```bash
python orchestrator.py --video path/to/video.mp4 --fps 3 --data_factor 4 --prompt "oxford book"
```

## Inputs

You can provide exactly one of these input sources:

- `--video path/to/video.mp4`
- `--images path/to/images/`
- `--scene path/to/colmap_scene/`

## CLI arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--video` | `None` | Path to the input video file. |
| `--images` | `None` | Path to the input images folder. |
| `--scene` | `None` | Path to the input COLMAP scene folder. |
| `--output_dir` | `./output` | Root directory for all pipeline outputs. |
| `--fps` | `5` | Number of frames to extract per second from the video. |
| `--data_factor` | `1` | Downsampling factor applied to input images. |
| `--segment` | `True` | Enable FastSAM segmentation on extracted frames. |
| `--no-segment` | `None` | Disable FastSAM segmentation on extracted frames. |
| `--prompt` | `toy` | Text prompt used by FastSAM during segmentation. |
| `--gpu` | `0` | CUDA device index to use. |

### Example commands

Use a video input:

```bash
python orchestrator.py --video path/to/video.mp4 --fps 3 --data_factor 4
```

Use an image directory:

```bash
python orchestrator.py --images path/to/images/ --fps 3 --data_factor 2
```

Use an existing COLMAP scene:

```bash
python orchestrator.py --scene path/to/colmap_scene/ --output_dir ./my_outputs
```

To see the full help text for the CLI, run:

```bash
python orchestrator.py -h
```
