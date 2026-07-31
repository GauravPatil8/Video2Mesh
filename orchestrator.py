from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_args import parse_args

from pipeline.stages.frames_loader import load_frames
from pipeline.stages.colmap_sfm import run_sfm
from pipeline.stages.mesh_extraction import run_mesh_extraction
from pipeline.config import PipelineConfig

def orchestrate(config: PipelineConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.frames_dir.mkdir(parents=True, exist_ok=True)
    config.scene_dir.mkdir(parents=True, exist_ok=True)
    config.mesh_output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [config.video, config.images, config.scene]
    num_inputs = sum(x is not None for x in inputs)

    if num_inputs == 0:
        raise ValueError(
            "One of --video, --images, or --scene must be provided."
        )

    if num_inputs > 1:
        raise ValueError(
            "Only one of --video, --images, or --scene may be provided."
        )
    
    if config.scene is None:
        load_frames(
            video_path=config.video,
            images_path=config.images,
            output_dir=config.frames_dir,
            fps=config.fps,
            data_factor=config.data_factor,
            segment=config.segment,
            prompt=config.prompt,
        )

        run_sfm(
            frames_dir=config.frames_dir,
            scene_dir=config.scene_dir,
        )

    run_mesh_extraction(
        scene_dir=config.scene_dir,
        mesh_output_dir=config.mesh_output_dir,
    )

def main() -> int:
    args = parse_args()
    config = PipelineConfig(
        video=args.video.resolve() if args.video else None,
        images=args.images.resolve() if args.images else None,
        scene=args.scene.resolve() if args.scene else None,
        output_dir=args.output_dir.resolve(),
        fps=args.fps,
        data_factor=args.data_factor,
        gpu_id=args.gpu,
        segment=args.segment,
        prompt=args.prompt,
    )

    orchestrate(config)


if __name__ == "__main__":
    main()

