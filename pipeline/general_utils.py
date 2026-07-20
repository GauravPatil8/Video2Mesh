import torch
import numpy as np
import pymeshlab
from scipy.spatial import KDTree
from tqdm import tqdm
import subprocess
from pathlib import Path

def clone_repo(repo_url: str, destination: str | Path):
    destination = Path(destination)
    subprocess.run(
        ["git", "clone", repo_url, str(destination)],
        check=True,
    )

def poisson_mesh(path, vtx, normal, color, depth, thrsh):

    pbar = tqdm(total=4)
    pbar.update(1)
    pbar.set_description('Poisson meshing')

    # create pcl with normal from sampled points
    ms = pymeshlab.MeshSet()
    pts = pymeshlab.Mesh(vtx.cpu().numpy(), [], normal.cpu().numpy())
    ms.add_mesh(pts)

    # poisson reconstruction
    ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True, samplespernode=1.5)
    vert = ms.current_mesh().vertex_matrix()
    face = ms.current_mesh().face_matrix()
    print(f"[poisson_mesh] Poisson output: {len(vert)} verts, {len(face)} faces")
    ms.save_current_mesh(path + '_plain.ply')

    if len(face) == 0:
        print("[poisson_mesh] WARNING: Poisson produced 0 faces. Saving as-is.")
        ms.save_current_mesh(path + '_pruned.ply')
        pbar.update(3)
        pbar.close()
        return

    pbar.update(1)
    pbar.set_description('Mesh refining')
    # knn to compute distance and color of poisson-meshed points to sampled points
    tree = KDTree(vtx.cpu().numpy())
    nn_dist_np, nn_idx_np = tree.query(vert, k=4)
    nn_dist = torch.from_numpy(nn_dist_np).to(torch.float32)
    nn_idx_t = torch.from_numpy(nn_idx_np).to(torch.long)
    nn_color = torch.mean(color[nn_idx_t], axis=1)

    # create mesh with color and quality (distance to the closest sampled points)
    vert_color = nn_color.clip(0, 1).cpu().numpy()
    vert_color = np.concatenate([vert_color, np.ones_like(vert_color[:, :1])], 1)
    quality = nn_dist[:, 0].cpu().numpy()
    print(f"[poisson_mesh] Quality stats: min={quality.min():.6f}, "
          f"median={np.median(quality):.6f}, "
          f"p90={np.percentile(quality, 90):.6f}, "
          f"max={quality.max():.6f}")
    ms.add_mesh(pymeshlab.Mesh(vert, face, v_color_matrix=vert_color, v_scalar_array=quality))

    pbar.update(1)
    pbar.set_description('Mesh cleaning')

    # Compute pruning threshold
    if thrsh <= 0:
        thrsh = float(np.percentile(quality, 90))
    # Safety: never prune if it would leave fewer than 100 vertices
    kept = (quality <= thrsh).sum()
    print(f"[poisson_mesh] Pruning threshold={thrsh:.6f}, vertices kept={kept}/{len(quality)}")
    if kept >= 100:
        ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
        ms.meshing_remove_selected_vertices()
    else:
        print("[poisson_mesh] Skipping pruning (would remove too many vertices)")

    # fill holes and smooth only if we still have faces
    if ms.current_mesh().face_number() > 0:
        ms.meshing_close_holes(maxholesize=300)
        ms.save_current_mesh(path + '_pruned.ply')

        ms.load_new_mesh(path + '_pruned.ply')
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)
        ms.save_current_mesh(path + '_pruned.ply')
    else:
        print("[poisson_mesh] WARNING: No faces after pruning, saving plain mesh")
        ms.load_new_mesh(path + '_plain.ply')
        ms.save_current_mesh(path + '_pruned.ply')

    print(f"[poisson_mesh] Final: {ms.current_mesh().vertex_number()} verts, "
          f"{ms.current_mesh().face_number()} faces")
    pbar.update(1)
    pbar.close()
