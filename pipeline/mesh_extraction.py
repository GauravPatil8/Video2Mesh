from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

from .utils import log_execution, logger

def _find_ply(gs_output_dir: Path) -> Path:
    """Return the path to the first `*.ply` file found in *gs_output_dir*.

    The gsplat / easy-3dgs pipeline typically writes a single PLY file
    (e.g. ``point_cloud.ply``) into the result directory or a
    sub-directory.  We search recursively so the caller does not need to
    know the exact layout.
    """
    candidates = sorted(glob.glob(str(gs_output_dir / "**" / "*.ply"), recursive=True))
    if not candidates:
        raise FileNotFoundError(
            f"No .ply file found under {gs_output_dir}. "
            "Make sure 3DGS training completed successfully."
        )
    for c in candidates:
        if "point_cloud" in Path(c).stem.lower():
            return Path(c)
    return Path(candidates[0])


def _load_gaussian_ply(ply_path: Path) -> o3d.geometry.PointCloud:
    """Load a 3DGS PLY and return an Open3D PointCloud.

    Gaussian splat PLY files store per-Gaussian attributes (position,
    covariance / scale + rotation, SH coefficients, opacity, …).  For
    Poisson reconstruction we only need the **positions** – and normals
    when available.  Colours (from SH DC component) are carried along so
    they can be baked into the mesh vertices.
    """
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]

    x = np.array(vertex["x"], dtype=np.float64)
    y = np.array(vertex["y"], dtype=np.float64)
    z = np.array(vertex["z"], dtype=np.float64)
    points = np.column_stack((x, y, z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    normal_names = {p.name for p in vertex.properties}
    if {"nx", "ny", "nz"}.issubset(normal_names):
        nx = np.array(vertex["nx"], dtype=np.float64)
        ny = np.array(vertex["ny"], dtype=np.float64)
        nz = np.array(vertex["nz"], dtype=np.float64)
        normals = np.column_stack((nx, ny, nz))
        # Only keep normals that are non-zero (some exporters write zeros).
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = (norms > 1e-8).squeeze()
        if valid.sum() > 0.5 * len(valid):
            normals[~valid] = [0, 0, 1]  # placeholder for degenerate ones
            normals = normals / np.maximum(norms, 1e-8)
            pcd.normals = o3d.utility.Vector3dVector(normals)

    sh_names = {p.name for p in vertex.properties}
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(sh_names):
        # SH DC to linear RGB: c = 0.5 + C0 * sh_dc  (C0 ≈ 0.28209)
        C0 = 0.28209479177387814
        r = 0.5 + C0 * np.array(vertex["f_dc_0"], dtype=np.float64)
        g = 0.5 + C0 * np.array(vertex["f_dc_1"], dtype=np.float64)
        b = 0.5 + C0 * np.array(vertex["f_dc_2"], dtype=np.float64)
        colors = np.clip(np.column_stack((r, g, b)), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif {"red", "green", "blue"}.issubset(sh_names):
        r = np.array(vertex["red"], dtype=np.float64) / 255.0
        g = np.array(vertex["green"], dtype=np.float64) / 255.0
        b = np.array(vertex["blue"], dtype=np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack((r, g, b)))

    if "opacity" in {p.name for p in vertex.properties}:
        opacity = np.array(vertex["opacity"], dtype=np.float64)
        # Gaussian opacity is stored as logit; apply sigmoid.
        sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacity))
        mask = sigmoid_opacity > 0.05
        pcd = pcd.select_by_index(np.where(mask)[0])
        logger.info(
            f"Opacity filter: kept {mask.sum()}/{len(mask)} Gaussians "
            f"(threshold 0.05)"
        )

    return pcd


def _prepare_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.0,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Down-sample, remove outliers, and ensure consistent normals."""

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        logger.info(f"After voxel downsampling ({voxel_size}): {len(pcd.points)} pts")

    pcd, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    logger.info(f"After outlier removal: {len(pcd.points)} pts")

    if not pcd.has_normals():
        logger.info("Estimating normals (KNN=30) …")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

    pcd.orient_normals_consistent_tangent_plane(k=15)

    return pcd


def _poisson_reconstruct(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    width: float = 0.0,
    scale: float = 1.1,
    linear_fit: bool = False,
    density_quantile: float = 0.01,
) -> o3d.geometry.TriangleMesh:
    """Run Screened Poisson Surface Reconstruction and clean the result.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Oriented point cloud (must have normals).
    depth : int
        Octree depth – controls the resolution of the reconstruction.
        Higher values capture finer detail but need more memory.
    width : float
        Target width of the finest octree cells (overrides *depth* when > 0).
    scale : float
        Ratio between the diameter of the cube used for reconstruction
        and the diameter of the samples' bounding cube.
    linear_fit : bool
        If True, use a linear interpolation for the iso-surface extraction
        instead of the default quadratic.  Faster but less smooth.
    density_quantile : float
        Fraction (0-1) of lowest-density vertices to trim.  Removes
        spurious surface at the boundary of the reconstruction volume.
    """
    logger.info(
        f"Running Poisson reconstruction (depth={depth}, "
        f"density_quantile={density_quantile}) …"
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    logger.info(
        f"Raw mesh: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )

 
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Clean up degenerate geometry.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    logger.info(
        f"Cleaned mesh: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )

    return mesh

@log_execution
def run_mesh_extraction(
    scene_dir: Path,
    gs_output_dir: Path,
    mesh_output_dir: Path,
    *,
    poisson_depth: int = 9,
    density_quantile: float = 0.01,
    voxel_size: float = 0.0,
) -> Path:
    """Extract a triangle mesh from a trained 3DGS scene via Poisson
    reconstruction and save it as an OBJ file.

    Parameters
    ----------
    scene_dir : Path
        COLMAP scene directory (used for context / camera poses, but not
        directly consumed by Poisson reconstruction).
    gs_output_dir : Path
        Directory containing the gsplat output (must have a `*.ply`
        point cloud).
    mesh_output_dir : Path
        Directory where the extracted mesh (`mesh.obj`) will be written.
    poisson_depth : int
        Octree depth for Poisson reconstruction (default 9).
    density_quantile : float
        Fraction of lowest-density vertices to trim (default 0.01).
    voxel_size : float
        Voxel size for down-sampling (0 = no down-sampling).

    Returns
    -------
    Path
        Absolute path to the saved OBJ file.
    """
    scene_dir = Path(scene_dir)
    gs_output_dir = Path(gs_output_dir)
    mesh_output_dir = Path(mesh_output_dir)
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    ply_path = _find_ply(gs_output_dir)
    logger.info(f"Loading Gaussian PLY: {ply_path}")
    pcd = _load_gaussian_ply(ply_path)
    logger.info(f"Loaded {len(pcd.points)} Gaussians")

    pcd = _prepare_point_cloud(pcd, voxel_size=voxel_size)

    mesh = _poisson_reconstruct(
        pcd,
        depth=poisson_depth,
        density_quantile=density_quantile,
    )

    obj_path = mesh_output_dir / "mesh.obj"
    o3d.io.write_triangle_mesh(
        str(obj_path),
        mesh,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    logger.info(f"Mesh saved to {obj_path}")

    return obj_path
