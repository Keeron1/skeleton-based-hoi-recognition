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
class DeepSORTConfig:
    max_age : int
    n_init : int
    nn_budget : int
    max_cosine_distance : float
    nms_max_overlap : float
    max_iou_distance : float

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
    deepsort: DeepSORTConfig
    hrnet: HrNetConfig