from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.frames_loader import extract_frames
from pipeline.colmap_sfm import run_sfm
from pipeline.gaussian_splatting import train as train_3dgs
from pipeline.mesh_extraction import run_mesh_extraction
from pipeline.config import PipelineConfig

def orchestrate(config: PipelineConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.frames_dir.mkdir(parents=True, exist_ok=True)
    config.scene_dir.mkdir(parents=True, exist_ok=True)
    config.gs_result_dir.mkdir(parents=True, exist_ok=True)
    config.sugar_output_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(
        video_path=config.video,
        output_dir=config.frames_dir,
        fps=config.fps,
    )

    run_sfm(
        frames_dir=config.frames_dir,
        scene_dir=config.scene_dir,
    )

    train_3dgs(
        scene_dir=config.scene_dir,
        result_dir=config.gs_result_dir,
        max_steps=config.max_steps,
        data_factor=config.data_factor
    )
    
    run_mesh_extraction(
        scene_dir=config.scene_dir,
        gs_output_dir=config.gs_result_dir,
        mesh_output_dir=config.sugar_output_dir,
        poisson_depth = config.poisson_depth,
        density_quantile = config.density_quantile,
        voxel_size = config.voxel_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the video-to-mesh pipeline.",
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./output"),
        help="Root directory for all pipeline outputs.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Extract N frames per second from the video.",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Number of gsplat training iterations.",
    )
    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=9,
        help="Octree depth for Poisson surface reconstruction. Higher values capture finer detail but use more memory (default: 9).",
    )
    parser.add_argument(
        "--density_quantile",
        type=float,
        default=0.01,
        help="Fraction (0-1) of lowest-density vertices to trim from the reconstructed mesh. Removes spurious surfaces at the boundary (default: 0.01).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="Voxel size for point cloud downsampling before reconstruction. 0 disables downsampling (default: 0.0).",
    )

    parser.add_argument(
        "--data_factor",
        type=int,
        default=4,
        help="Downsample factor for input images.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(
        video=args.video.resolve(),
        output_dir=args.output_dir.resolve(),
        fps=args.fps,
        max_steps=args.max_steps,
        data_factor=args.data_factor,
        gpu_id=args.gpu,
        poisson_depth = args.poisson_depth,
        density_quantile = args.density_quantile,
        voxel_size = args.voxel_size
    )

    orchestrate(config)


if __name__ == "__main__":
    main()

