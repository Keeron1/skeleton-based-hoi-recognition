import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
import torch
from .config_schema import AppConfig, PathsConfig, YOLOConfig, DeepSORTConfig, HrNetConfig

# Load .env file
load_dotenv()

# Load yaml file
def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

class Config:
    def __init__(self):
        # Set project root
        self.project_root = Path(__file__).resolve().parents[3]
        
        # Load env variables
        self.env = os.getenv("ENV", "local") # Get ENV or default to local
        self.data_root = os.getenv("DATA_ROOT", "data") # Get environment dataset path

        # Load yaml files
        self.configs = {
            "paths": load_yaml(self.project_root / "configs/paths.yaml"), # Where things are located
            "model": load_yaml(self.project_root / "configs/model.yaml") # How models are configured
        }
    
    def get(self, config_name, key_path):
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")

        config = self.configs[config_name]
            
        try:
            keys = key_path.split(".")
            value = config

            # Get through all keys
            for k in keys:
                value = value[k]

            # Return dataset path (env path + dataset path)
            if config_name == "paths" and keys[0] == "dataset":
                return os.path.join(self.data_root, value)

            return value
        except KeyError:
            raise KeyError(f"Invalid config path: {key_path}")
       
    # Loads the whole config that the prototype needs 
    def load_config(self) -> AppConfig:     
        # nvidia gpu or cpu
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset_path = self.get("paths", "dataset.processed")

        output_videos_path = self.project_root / self.get("paths", "outputs.videos")
        models_dir = self.project_root / self.get("paths", "models")

        # Gets the path where the model will load or download from
        yolo_model_type = models_dir / self.get("model", "yolo.model_type") 
        yolo_imgsz = self.get("model", "yolo.imgsz")

        deepsort_max_age = self.get("model", "deepsort.max_age")
        deepsort_n_init = self.get("model", "deepsort.n_init")
        deepsort_nn_budget = self.get("model", "deepsort.nn_budget")
        deepsort_max_cosine_distance = self.get("model", "deepsort.max_cosine_distance")
        deepsort_nms_max_overlap = self.get("model", "deepsort.nms_max_overlap")
        deepsort_max_iou_distance = self.get("model", "deepsort.max_iou_distance")

        # should be able to download these and store in models folder instead of online
        hrnet_model_cfg = self.get("model", "hrnet.w32_256x192_coco.model_cfg")
        hrnet_model_ckpt = self.get("model", "hrnet.w32_256x192_coco.model_ckpt")
        
        return AppConfig(
            project_root=self.project_root,
            device=device,
            paths=PathsConfig(
                dataset=dataset_path,
                output_videos=output_videos_path,
                models=models_dir
            ),
            yolo=YOLOConfig(
                model_path=yolo_model_type,
                imgsz=yolo_imgsz
            ),
            deepsort=DeepSORTConfig(
                max_age = deepsort_max_age,
                n_init = deepsort_n_init,
                nn_budget = deepsort_nn_budget,
                max_cosine_distance = deepsort_max_cosine_distance,
                nms_max_overlap = deepsort_nms_max_overlap,
                max_iou_distance = deepsort_max_iou_distance
            ),
            hrnet=HrNetConfig(
                model_cfg=hrnet_model_cfg,
                model_ckpt=hrnet_model_ckpt
            )
        )
        