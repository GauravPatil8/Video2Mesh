from pathlib import Path
from .log_utils import log_execution
from easy_3dgs.pipeline import GaussianSplattingPipeline

@log_execution
def train(
    scene_dir: Path,
    result_dir: Path,
    max_steps: int = 30_000,
    data_factor: int = 1,
):
    """Train a 3D Gaussian Splatting model with gsplat.

    Parameters:
        scene_dir : Path
            COLMAP scene directory.
        result_dir : Path
            Directory where gsplat will write checkpoints and the .ply file.
        max_steps : int, optional
            Total training iterations (default 30_000).
        data_factor : int, optional
            Down-sample factor for input images (default 1 = full res).
    """
    scene_dir = Path(scene_dir)
    result_dir = Path(result_dir)

    gaussian_splatting_pipeline = GaussianSplattingPipeline(
        data_factor=data_factor,
        result_dir=result_dir,
        strategy_type="mcmc",  
        max_steps = max_steps,
        disable_viewer=True
    )

    gaussian_splatting_pipeline.train(scene_dir)
