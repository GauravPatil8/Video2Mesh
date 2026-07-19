import subprocess
import sys
from pathlib import Path
from utils import clone_repo
from utils import log_execution

@log_execution
def run_mesh_extraction(
    scene_dir: Path,
    gs_output_dir: Path,
    sugar_output_dir: Path,
    sugar_repo: Path,
    gpu_id: int = 0,
):
    """Run SuGaR to extract a mesh from a trained 3DGS model.

    Parameters
    ----------
    scene_dir : Path
        COLMAP scene directory (``images/`` + ``sparse/0/``).
    gs_output_dir : Path
        Directory containing the trained 3DGS ``.ply`` checkpoint
        (output of :func:`pipeline.gaussian_splatting.train_3dgs`).
    sugar_output_dir : Path
        Directory where SuGaR will write its outputs (coarse model,
        extracted mesh, refined model).
    sugar_repo : Path
        Path to a local clone of `Anttwo/SuGaR
        <https://github.com/Anttwo/SuGaR>`_.
    gpu_id : int, optional
        CUDA device index (default ``0``).
    """

    repo_url = "https://github.com/Anttwo/SuGaR.git"
    scene_dir = Path(scene_dir)
    gs_output_dir = Path(gs_output_dir)
    sugar_output_dir = Path(sugar_output_dir)
    sugar_repo = Path(sugar_repo)

    clone_repo(repo_url=repo_url, destination=sugar_repo)
    train_script = sugar_repo / "train_full_pipeline.py"

    if not train_script.exists():
        raise FileNotFoundError(
            f"Cannot find SuGaR script at {train_script}.\n"
            f"Make sure you have cloned https://github.com/Anttwo/SuGaR "
            f"into {sugar_repo}"
        )

    sugar_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        "-s", str(scene_dir),
        "-c", str(sugar_output_dir),
        "-r", "default",
        "--gpu", str(gpu_id),
        "--gs_output_dir", str(gs_output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(sugar_repo),
        stdout=None,
        stderr=None,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"SuGaR mesh extraction failed with return code {result.returncode}. "
            f"Check the output above for details."
        )