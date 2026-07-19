import os
from pathlib import Path
import pycolmap
from .utils import log_execution

def _has_cuda() -> bool:
    """Return True if pycolmap has CUDA support."""
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
    sparse_root = scene_dir / "sparse"
    sparse_dir = sparse_root / "0"
    database_path = scene_dir / "database.db"

    sparse_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = _has_cuda()
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    print(f"Using {'GPU' if use_gpu else 'CPU'} for COLMAP")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = use_gpu

    if use_gpu:
        extraction_options.gpu_index = "0"

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        extraction_options=extraction_options,
        device=device,
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = use_gpu

    if use_gpu:
        matching_options.gpu_index = "0"

    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = 15

    pairing_options.loop_detection = False

    pycolmap.match_sequential(
        database_path=database_path,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=device,
    )

    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_root,
    )

    if not reconstructions:
        raise RuntimeError(
            "COLMAP incremental mapping failed. "
            "Try extracting more frames or increasing overlap."
        )

    best = max(
        reconstructions.values(),
        key=lambda r: r.num_reg_images(),
    )

    best.write(sparse_dir)