import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.classifier.common.feature_extractor import FeatureExtractor
from src.classifier.common.constants import OBJECT_CAT_IDS
from src.classifier.lstm.dataset import (
    _collect_frames, _load_action_id_to_name, _build_object_track_to_category
)

class HOIFrameDataset(Dataset):
    def __init__(self,
                 clips_root,
                 class_names,
                 sample_every=1,
                 frame_width=1920,
                 frame_height=1080,
                 action_classes_path=None,
                 clip_dirs=None):
        self.clips_root = Path(clips_root)
        self.clip_dirs = clip_dirs
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.sample_every = sample_every

        if action_classes_path is None:
            action_classes_path = self.clips_root.parent / "action_classes.json"
        self.action_id_to_name = _load_action_id_to_name(action_classes_path)

        self.feature_extractor = FeatureExtractor(
            num_object_classes=len(OBJECT_CAT_IDS),
            frame_width=frame_width,
            frame_height=frame_height
        )

        # list of (feature_vector, label_idx)
        self.samples = []
        self._build()

    def _build(self):
        if self.clip_dirs is not None:
            clip_dirs = [Path(p) for p in self.clip_dirs]
        else:
            clip_dirs = [p.parent for p in self.clips_root.rglob("hoi-anns.json")]
        if not clip_dirs:
            print(f"No hoi-anns.json files found under {self.clips_root}")
            return

        skipped = 0
        for clip_dir in clip_dirs:
            try:
                with open(clip_dir / "anns.json", "r", encoding="utf-8") as f:
                    ann_data = json.load(f, strict=False)
                with open(clip_dir / "hoi-anns.json", "r", encoding="utf-8") as f:
                    hoi_data = json.load(f, strict=False)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Skipping {clip_dir}: {e}")
                skipped += 1
                continue

            segments = hoi_data.get("annotations", [])
            if not segments:
                continue

            self._process_clip(ann_data, segments)

        print(f"Built {len(self.samples)} frame samples from {len(clip_dirs) - skipped} clips (skipped {skipped})")

    def _process_clip(self, ann_data, segments):
        anns_by_frame = {}
        for ann in ann_data.get("annotations", []):
            anns_by_frame.setdefault(ann["image_id"], []).append(ann)

        track_to_cat = _build_object_track_to_category(ann_data)

        for seg in segments:
            label = self.action_id_to_name.get(seg.get("action_id"))
            if label not in self.class_to_idx:
                continue

            human_track = seg["person_track_id"]
            object_track = seg.get("object_track_id", -1)
            start = seg["start_frame"]
            end = seg["end_frame"]

            if object_track == -1:
                object_track = None
                object_cat = None
            else:
                object_cat = track_to_cat.get(object_track)
                if object_cat is None:
                    continue

            frames_data = _collect_frames(
                anns_by_frame, human_track, object_track, object_cat, start, end
            )

            label_idx = self.class_to_idx[label]
            for i in range(0, len(frames_data), self.sample_every):
                f = frames_data[i]
                feat = self.feature_extractor.extract(
                    f["human_bbox"], f["object_bbox"], f["object_class_id"]
                )
                self.samples.append((feat, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        x = torch.tensor(feat, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
