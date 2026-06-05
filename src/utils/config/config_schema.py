from dataclasses import dataclass
from pathlib import Path

@dataclass
class HrNetConfig:
    model_cfg: str
    model_ckpt: str

@dataclass
class YOLOConfig:
    model_path: Path
    imgsz: int

@dataclass
class DeepSORTConfig:
    max_age: int
    n_init: int
    nn_budget: int
    max_cosine_distance: float
    nms_max_overlap: float
    max_iou_distance: float

@dataclass
class GNNConfig:
    model_ckpt: Path
    seq_len: int
    stride: int
    hidden: int
    heads: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    hflip_p: float
    jitter_std: float
    obj_dropout_p: float

@dataclass
class LSTMConfig:
    seq_len: int
    stride: int
    hidden: int
    num_layers: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int

@dataclass
class MLPConfig:
    hidden: int
    dropout: float
    sample_every: int
    lr: float
    batch_size: int
    epochs: int

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
    gnn: GNNConfig
    lstm: LSTMConfig
    mlp: MLPConfig
