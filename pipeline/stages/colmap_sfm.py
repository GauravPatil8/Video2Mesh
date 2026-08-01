import shutil
from pathlib import Path

import pycolmap

from ..utils.logs import log_execution


def _has_cuda() -> bool:
    try:
        return pycolmap.has_cuda
    except AttributeError:
        try:
            return pycolmap.Device.cuda is not None
        except Exception:
            return False


@log_execution
def run_sfm(frames_dir: Path, scene_dir: Path):

    image_dir = Path(frames_dir)

    database_path = scene_dir / "database.db"
    sparse_root = scene_dir / "sparse"
    sparse_dir = sparse_root / "0"

    if database_path.exists():
        database_path.unlink()

    if sparse_root.exists():
        shutil.rmtree(sparse_root)

    sparse_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = _has_cuda()
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    print(f"Using {'GPU' if use_gpu else 'CPU'} for COLMAP")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = use_gpu

    if use_gpu:
        extraction_options.gpu_index = "0"

    reader_options = pycolmap.ImageReaderOptions()

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),

        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",

        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = use_gpu

    if use_gpu:
        matching_options.gpu_index = "0"

    pycolmap.match_exhaustive(
        database_path=str(database_path),
        matching_options=matching_options,
        device=device,
    )

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(sparse_root),
    )

    if len(reconstructions) == 0:
        raise RuntimeError("COLMAP mapping failed.")

    best = max(
        reconstructions.values(),
        key=lambda r: r.num_reg_images(),
    )

    best.write(str(sparse_dir))

    print(f"Registered {best.num_reg_images()} images.")

    recon = pycolmap.Reconstruction(str(sparse_dir))

    for cam in recon.cameras.values():
        print("Camera model:", cam.model)