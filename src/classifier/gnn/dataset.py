import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.classifier.common.constants import PERSON_CAT_ID
from src.classifier.common.features import build_node_features, NODE_CHANNELS
from src.classifier.common.normalize import normalize_window


# Sliding window dataset for the GNN.
# One sample is a (T, N, C) tensor of normalized node features and a class label.
# N = 18: 17 COCO keypoints + 1 object node positioned at the object bbox center.
# C = 8: x, y, conf, w, h, then a 3-dim object class one-hot.
#
# w, h are bbox dimensions. They're zero for human keypoints (since each keypoint
# is a point, not a region) and the object's size for the object node. This lets
# the model distinguish a phone from a laptop by size alone, the same information
# the LSTM / MLP get from the full object bbox.
#
# The one-hot is non-zero only on the object node and only when the object is
# visible in that frame.
class GNNHOIDataset(Dataset):
    def __init__(self, clips_root, cache_root, clip_keys, action_names,
                 seq_len=16, stride=8, augment=None):
        self.clips_root = Path(clips_root)
        self.cache_root = Path(cache_root)
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.action_names = action_names
        self.name_to_idx = {n: i for i, n in enumerate(action_names)}

        with open(self.clips_root.parent / "action_classes.json") as f:
            name_to_id = json.load(f)
        self.id_to_name = {int(v): k for k, v in name_to_id.items()}

        self.samples = []
        self._diag = defaultdict(int)
        self._build(clip_keys)

    def _build(self, clip_keys):
        for key in clip_keys:
            vp, cid = key.split("/")
            clip_dir = self.clips_root / vp / cid

            ann_path = clip_dir / "anns.json"
            hoi_path = clip_dir / "hoi-anns.json"
            if not ann_path.exists() or not hoi_path.exists():
                self._diag["missing_files"] += 1
                continue

            with open(ann_path) as f:
                coco = json.load(f)
            with open(hoi_path) as f:
                hoi = json.load(f)

            frame_kpts = self._load_keypoints(vp, cid)
            if frame_kpts is None:
                self._diag["missing_cache"] += 1
                continue

            obj_lookup = self._index_objects(coco)

            for seg in hoi.get("annotations", []):
                self._add_segment(seg, frame_kpts, obj_lookup)

    def _load_keypoints(self, vp, clip):
        path = self.cache_root / str(vp) / f"{clip}.json"
        if not path.exists():
            return None
        with open(path) as f:
            raw = json.load(f)
        return {
            int(fid): {int(tid): kpts for tid, kpts in by_track.items()}
            for fid, by_track in raw.items()
        }

    # track_id -> frame_id -> (x_center, y_center, w, h, category_id)
    def _index_objects(self, coco):
        out = defaultdict(dict)
        for ann in coco.get("annotations", []):
            cat = ann["category_id"]
            if cat == PERSON_CAT_ID:
                continue
            tid = _parse_track_id(ann)
            if tid is None:
                continue
            x, y, w, h = ann["bbox"]
            out[tid][ann["image_id"]] = (x + w / 2.0, y + h / 2.0, w, h, cat)
        return out

    def _add_segment(self, seg, frame_kpts, obj_lookup):
        self._diag["segments_total"] += 1

        label_name = self.id_to_name.get(seg["action_id"])
        if label_name not in self.name_to_idx:
            self._diag["dropped_unknown_label"] += 1
            return
        label = self.name_to_idx[label_name]

        person_track = int(seg["person_track_id"])
        object_track = int(seg.get("object_track_id", -1))
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])

        frames = []
        for fid in range(start, end + 1):
            kpts = frame_kpts.get(fid, {}).get(person_track)
            if kpts is None:
                continue

            obj_info = None
            if object_track != -1:
                obj_info = obj_lookup.get(object_track, {}).get(fid)

            frames.append(build_node_features(np.asarray(kpts, dtype=np.float32), obj_info))

        if len(frames) < self.seq_len:
            self._diag["dropped_too_short"] += 1
            return

        for i in range(0, len(frames) - self.seq_len + 1, self.stride):
            window = np.stack(frames[i:i + self.seq_len])
            window = normalize_window(window)
            self.samples.append((window, label))

    def label_counts(self):
        counts = [0] * len(self.action_names)
        for _, y in self.samples:
            counts[y] += 1
        return counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def _parse_track_id(ann):
    tid = ann.get("attributes", {}).get("track_id")
    if tid is None or tid == "":
        return None
    try:
        return int(tid)
    except (TypeError, ValueError):
        return None
