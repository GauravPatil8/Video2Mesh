from pathlib import Path
from ..utils.logs import log_execution
from easy_3dgs.pipeline import GaussianSplattingPipeline

@log_execution
def train(
    scene_dir: Path,
    result_dir: Path,
    max_steps: int = 30_000,
    strategy_type: str = "mcmc" 
):
    """Train a 3D Gaussian Splatting model with gsplat.

    Parameters:
        scene_dir : Path
            COLMAP scene directory.
        result_dir : Path
            Directory where gsplat will write checkpoints and the .ply file.
        max_steps : int, optional
            Total training iterations (default 30_000).
        strategy_type: str ["default", "mcmc"]
            strategy for pruning and addition of GSs.
    """
    scene_dir = Path(scene_dir)
    result_dir = Path(result_dir)

    gaussian_splatting_pipeline = GaussianSplattingPipeline(
        result_dir=result_dir,
        strategy_type=strategy_type,  
        max_steps = max_steps,
        disable_viewer=True,
        data_factor=1,
    )

    gaussian_splatting_pipeline.train(scene_dir)
