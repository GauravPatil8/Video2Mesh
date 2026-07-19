import os
from pathlib import Path
import pycolmap
from .utils import log_execution

@log_execution
def run_sfm(frames_dir: Path, scene_dir: Path):

    image_dir = frames_dir
    sparse_dir = scene_dir / "sparse" / "0"
    database_path = scene_dir / "database.db"

    os.makedirs(sparse_dir, exist_ok=True)

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = True  
    extraction_options.gpu_index = "0"

    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        extraction_options=extraction_options,
        device=pycolmap.Device.auto,
    )

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = True
    matching_options.gpu_index = "0"

    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = 15
    pairing_options.loop_detection = True

    pycolmap.match_sequential(
        database_path=database_path,
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.auto,
    )

    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=scene_dir / "sparse",
    )

    if len(reconstructions) == 0:
        raise RuntimeError(
            "COLMAP incremental mapping failed. "
            "Try extracting more frames or increasing overlap."
        )
    best = max(
        reconstructions.values(),
        key=lambda r: r.num_reg_images(),
    )

    best.write(sparse_dir)

def _test_sfm():
    frames_dir = Path("extracted_frames")
    scene_dir = Path("scene")
    os.makedirs(scene_dir, exist_ok=True)
    run_sfm(frames_dir, scene_dir)

if __name__ == "__main__":
    _test_sfm()