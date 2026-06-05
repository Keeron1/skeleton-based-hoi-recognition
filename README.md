# Skeleton-Based HOI Recognition

Real-time Human-Object Interaction recognition prototype on the CAFE dataset.

Pipeline:
**YOLOv8 (detection) > DeepSORT (tracking) > HRNet (pose) > ST-GATv2 (action classifier)**

Built as the prototype for a BSc dissertation titled ***Skeleton-Based Human-Object Interaction Recognition in Surveillance Environments Using a Spatio-Temporal Graph Attention Network***.

## Setup

GPU:

```powershell
./setup_main.ps1
.\venv\Scripts\activate
```

CPU-only (may fail on OpenMMLab deps):

```powershell
./setup_cpu.ps1
```

Python is pinned to **3.10.11**, PyTorch to **2.1.2 + cu118**, NumPy to `>=1.26,<2` (mmcv / mmpose / mmdet break on NumPy 2).

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/main.ipynb` | End-to-end live inference pipeline. Picks a held-out fold checkpoint per clip or uses `final.pt`. |
| `notebooks/yolo.ipynb` | Train and evaluate YOLOv8m on the CAFE dataset. |
| `notebooks/deepsort.ipynb` | Per-class MOT evaluation of DeepSORT. |
| `notebooks/hrnet.ipynb` | HRNet pose estimation sanity check. |
| `notebooks/cache_keypoints.ipynb` | One-time HRNet keypoint cache used by the classifier. |
| `notebooks/gnn.ipynb` | 5-fold training and evaluation of the ST-GATv2 classifier. |
| `notebooks/lstm.ipynb` | 5-fold LSTM baseline. |
| `notebooks/single_frame.ipynb` | 5-fold single-frame MLP baseline. |
| `notebooks/comparison.ipynb` | Cross-classifier comparison (F1 tables, confusion matrices). |
| `notebooks/train_final.ipynb` | Train one ST-GATv2 on all clips for live pipeline `final.pt`. |
| `src/utils/dataset/*.ipynb` | Dataset preparation and annotation. |

## Folder layout

```
src/                source modules (detector, tracker, pose, classifier, utils)
notebooks/          all training, evaluation, and inference notebooks
configs/            paths.yaml and model.yaml
models/             trained weights (YOLO, HRNet checkpoint, GNN)
runs/               training outputs (per-fold GNN checkpoints, YOLO training logs)
outputs/            generated videos
```

## Dataset

The CAFE dataset is not included in the repository. Set `DATA_ROOT` as an environment variable to point at the dataset root. Clips are expected at `<DATA_ROOT>/processed/CafeV1/Clips/<viewpoint>/<clip>/`.

## License

Research prototype, academic use only.