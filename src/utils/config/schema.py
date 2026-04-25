from dataclasses import dataclass
from pathlib import Path

@dataclass
class HrNetConfig:
    model_cfg: str
    model_ckpt: str

@dataclass
class YOLOConfig:
    model_path: Path

@dataclass
class PathsConfig:
    dataset: str
    output_videos: Path
    models: Path

@dataclass
class AppConfig:
    project_root: Path
    device: str
    paths: PathsConfig
    yolo: YOLOConfig
    hrnet: HrNetConfig