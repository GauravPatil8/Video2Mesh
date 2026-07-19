from __future__ import annotations
import os
import argparse
from pathlib import Path

from .pipeline.frames_loader import extract_frames
from .pipeline.colmap_sfm import run_sfm
from .pipeline.gaussian_splatting import train as train_3dgs
from .pipeline.mesh_extraction import run_mesh_extraction
from .pipeline.config import PipelineConfig

def orchestrate(config: PipelineConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.frames_dir.mkdir(parents=True, exist_ok=True)
    config.scene_dir.mkdir(parents=True, exist_ok=True)
    config.gs_result_dir.mkdir(parents=True, exist_ok=True)
    config.sugar_output_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(
        video_path=config.video,
        output_dir=config.frames_dir,
        frame_skip=config.frame_skip,
    )

    run_sfm(
        frames_dir=config.frames_dir,
        scene_dir=config.scene_dir,
    )

    train_3dgs(
        scene_dir=config.scene_dir,
        result_dir=config.gs_result_dir,
        repo_dest=config.gsplat_repo,
        max_steps=config.max_steps,
        data_factor=config.data_factor,
        disable_viewer=True,
    )
    
    run_mesh_extraction(
        scene_dir=config.scene_dir,
        gs_output_dir=config.gs_result_dir,
        sugar_output_dir=config.sugar_output_dir,
        sugar_repo=config.sugar_repo,
        gpu_id=config.gpu_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dirac video-to-mesh pipeline.",
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
        "--gsplat_repo",
        type=Path,
        default=Path("./lib"),
        help="Path to the local clone of https://github.com/nerfstudio-project/gsplat.",
    )
    parser.add_argument(
        "--sugar_repo",
        type=Path,
        default=Path("./lib"),
        help="Path to the local clone of https://github.com/Anttwo/SuGaR.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30_000,
        help="Number of gsplat training iterations.",
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=1,
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
        gsplat_repo=args.gsplat_repo.resolve(),
        sugar_repo=args.sugar_repo.resolve(),
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        data_factor=args.data_factor,
        gpu_id=args.gpu,
    )

    orchestrate(config)


if __name__ == "__main__":
    main()

