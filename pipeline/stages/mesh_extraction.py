"""Poisson mesh extraction from a 3D Gaussian Splatting scene.

Loads the Gaussian PLY, builds an oriented point cloud, and calls
``poisson_mesh`` from ``utils.general_utils`` (pymeshlab + scipy)
to reconstruct and export a cleaned OBJ mesh.
"""

from __future__ import annotations
import os
from pathlib import Path
import subprocess
import shutil

from ..utils.general import clone_repo
from ..utils.logs import log_execution, logger

@log_execution
def run_mesh_extraction(
    scene_dir: Path,
    mesh_output_dir: Path,
) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    sugar_root = Path(os.path.join(project_root, "libs", "SuGaR"))

    if not os.path.exists(sugar_root):
        clone_repo("https://github.com/Anttwo/SuGaR.git", os.path.join(project_root, "libs"))

    sugaR_script = sugar_root / "train_full_pipeline.py"

    if not sugaR_script.exists():
        raise FileNotFoundError(
            f"Expected SuGaR pipeline script at {sugaR_script}. "
            "Clone it into libs/SuGaR before running mesh extraction."
        )

    subprocess.run(
        [
            "python",
            str(sugaR_script),
            "-s",
            str(scene_dir),
            "-r",
            "dn_consistency",
            "--low_poly",
            "True",
            "--export_obj",
            "True",
            "--refinement_time",
            "short"
        ],
        cwd=sugar_root,
        check=True,
    )

    output_folder = sugar_root / 'output'
    shutil.copytree(output_folder, mesh_output_dir, dirs_exist_ok=True)