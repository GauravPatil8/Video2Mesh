import subprocess
import sys
from pathlib import Path
from .utils import clone_repo
from .utils import log_execution

@log_execution
def train(
    scene_dir: Path,
    result_dir: Path,
    repo_dest: Path,
    max_steps: int = 30_000,
    data_factor: int = 1,
    disable_viewer: bool = True,
):
    """Train a 3D Gaussian Splatting model with gsplat.

    Parameters->
    scene_dir : Path
        COLMAP scene directory (must contain ``images/`` and ``sparse/0/``).
    result_dir : Path
        Directory where gsplat will write checkpoints and the ``.ply`` file.
    gsplat_repo : Path
        Path to a local clone of `nerfstudio-project/gsplat
        <https://github.com/nerfstudio-project/gsplat>`_.
    max_steps : int, optional
        Total training iterations (default ``30_000``).
    data_factor : int, optional
        Down-sample factor for input images (default ``1`` = full res).
    disable_viewer : bool, optional
        Disable the Viser real-time viewer (default ``True``).
    """
    scene_dir = Path(scene_dir)
    result_dir = Path(result_dir)
    gsplat_repo = Path(repo_dest)

    clone_repo(repo_url="https://github.com/nerfstudio-project/gsplat.git", destination=repo_dest)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=str(repo_dest),
        check=True,
    )
    
    trainer_script = gsplat_repo / "examples" / "simple_trainer.py"
    if not trainer_script.exists():
        raise FileNotFoundError(
            f"Cannot find gsplat trainer at {trainer_script}.\n"
            f"Make sure you have cloned https://github.com/nerfstudio-project/gsplat "
            f"into {gsplat_repo}"
        )

    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(trainer_script),
        "default",
        "--data_dir", str(scene_dir),
        "--result_dir", str(result_dir),
        "--data_factor", str(data_factor),
        "--save_ply",
        "--ply_steps", str(max_steps),
    ]

    if disable_viewer:
        cmd.append("--disable_viewer")


    result = subprocess.run(
        cmd,
        cwd=str(gsplat_repo),
        stdout=None,  
        stderr=None,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"gsplat training failed with return code {result.returncode}. "
            f"Check the output above for details."
        )

    ply_files = sorted(result_dir.rglob("*.ply"))
    if ply_files:
        print(f"[gsplat] Training complete — PLY saved at {ply_files[-1]}")
    else:
        print("[gsplat] Training complete — no .ply found (check --save_ply flag)")
