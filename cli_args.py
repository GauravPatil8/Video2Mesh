import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the video-to-mesh pipeline.",
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Path to the input images folder.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Path to the input COLMAP scene folder.",
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
