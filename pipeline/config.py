from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    video: Path
    output_dir: Path
    fps: int = 5
    max_steps: int = 30_000
    data_factor: int = 1
    gpu_id: int = 0
    poisson_depth: int = 9,
    density_quantile: float = 0.01,
    voxel_size:float = 0.0

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "scene" / "images"

    @property
    def scene_dir(self) -> Path:
        return self.output_dir / "scene"

    @property
    def gs_result_dir(self) -> Path:
        return self.output_dir / "3dgs"

    @property
    def sugar_output_dir(self) -> Path:
        return self.output_dir / "extracted_mesh"