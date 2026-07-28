"""Poisson mesh extraction from a 3D Gaussian Splatting scene.

Loads the Gaussian PLY, builds an oriented point cloud, and calls
``poisson_mesh`` from ``utils.general_utils`` (pymeshlab + scipy)
to reconstruct and export a cleaned OBJ mesh.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import open3d as o3d
import pymeshlab
import torch
from plyfile import PlyData

from ..utils.logs import log_execution, logger
from ..utils.general import poisson_mesh


def _find_ply(gs_output_dir: Path) -> Path:
    """Recursively find the first ``*.ply`` under *gs_output_dir*,
    preferring files whose name contains ``point_cloud``."""
    candidates = sorted(
        glob.glob(str(gs_output_dir / "**" / "*.ply"), recursive=True)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .ply file found under {gs_output_dir}. "
            "Make sure 3DGS training completed successfully."
        )
    for c in candidates:
        if "point_cloud" in Path(c).stem.lower():
            return Path(c)
    return Path(candidates[0])


def _load_gaussian_ply(ply_path: Path):
    """Load a 3DGS PLY and return (points, normals, colors) as torch tensors.

    Returns
    -------
    points : Tensor (N, 3)
    normals : Tensor | None (N, 3)
    colors : Tensor | None (N, 3) — values in [0, 1]
    """
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    
    # Positions
    points = np.column_stack([
        np.array(vertex["x"], dtype=np.float32),
        np.array(vertex["y"], dtype=np.float32),
        np.array(vertex["z"], dtype=np.float32),
    ])

    # Convert to torch tensors (poisson_mesh expects them)
    points_t = torch.from_numpy(points)

    return points_t


def _prepare_normals(
    points: torch.Tensor,
    voxel_size: float = 0.0,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean the point cloud and ensure normals exist.

    Uses Open3D for statistical outlier removal and normal estimation,
    then returns cleaned (points, normals) as torch tensors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy().astype(np.float64))
    
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        logger.info(f"After voxel downsampling ({voxel_size}): {len(pcd.points)} pts")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    logger.info(f"After outlier removal: {len(pcd.points)} pts")

    if not pcd.has_normals():
        logger.info("Estimating normals (KNN=30) …")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

    # Orient normals outward using centroid (avoids Qhull crash)
    centroid = pcd.get_center()
    pcd.orient_normals_towards_camera_location(camera_location=centroid)
    # orient_towards_camera points inward → flip to outward
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

    pts_out = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
    nrm_out = torch.from_numpy(np.asarray(pcd.normals).astype(np.float32))
    return pts_out, nrm_out


@log_execution
def run_mesh_extraction(
    gs_output_dir: Path,
    mesh_output_dir: Path,
    *,
    poisson_depth: int = 9,
    voxel_size: float = 0.0,
) -> Path:
    """Extract a triangle mesh from a trained 3DGS scene.

    Parameters
    ----------
    gs_output_dir : Path
        Directory containing the gsplat ``*.ply`` point cloud.
    mesh_output_dir : Path
        Output directory for the mesh files.
    poisson_depth : int
        Octree depth for Poisson reconstruction (default 9).
    voxel_size : float
        Voxel size for point-cloud downsampling (0 = disabled).

    Returns
    -------
    Path
        Absolute path to the saved OBJ file.
    """
    gs_output_dir = Path(gs_output_dir)
    mesh_output_dir = Path(mesh_output_dir)
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = _find_ply(gs_output_dir)
    logger.info(f"Loading Gaussian PLY: {ply_path}")
    points = _load_gaussian_ply(ply_path)
    logger.info(f"Loaded {len(points)} Gaussians")

    points, normals = _prepare_normals(points, voxel_size=voxel_size)

    mesh_prefix = str(mesh_output_dir / f"poisson_mesh_{poisson_depth}")
    poisson_mesh(
        path=mesh_prefix,
        vtx=points,
        normal=normals,
        depth=poisson_depth,
        thrsh=0,
    )

    pruned_ply = mesh_prefix + "_pruned.ply"
    plain_ply = mesh_prefix + "_plain.ply"
    pruned_obj_path = mesh_output_dir / "pruned_mesh.obj"
    plain_obj_path = mesh_output_dir / "plain_mesh.obj"

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(pruned_ply)
    ms.save_current_mesh(str(pruned_obj_path))

    ms.load_new_mesh(plain_ply)
    ms.save_current_mesh(str(plain_obj_path))

    logger.info(
        f"Mesh saved to {pruned_obj_path} & {plain_obj_path}"
    )
