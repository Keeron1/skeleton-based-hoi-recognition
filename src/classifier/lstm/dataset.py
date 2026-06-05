import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.classifier.common.feature_extractor import FeatureExtractor
from src.classifier.common.constants import (
    PERSON_CAT_ID, OBJECT_CAT_IDS, object_class_to_index
)

class HOISequenceDataset(Dataset):
    def __init__(self,
                 clips_root,
                 class_names,
                 seq_len=16,
                 stride=8,
                 frame_width=1920,
                 frame_height=1080,
                 action_classes_path=None,
                 clip_dirs=None):
        # If clip_dirs is provided, use exactly those clips (for clip-level splits)
        # If None, auto-discover all hoi-anns.json under clips_root (legacy behaviour)
        self.clips_root = Path(clips_root)
        self.clip_dirs = clip_dirs
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.seq_len = seq_len
        self.stride = stride

        # Load action name mapping
        if action_classes_path is None:
            action_classes_path = self.clips_root.parent / "action_classes.json"
        self.action_id_to_name = _load_action_id_to_name(action_classes_path)

        self.feature_extractor = FeatureExtractor(
            num_object_classes=len(OBJECT_CAT_IDS),
            frame_width=frame_width,
            frame_height=frame_height
        )

        # list of (sequence_features, label_idx)
        self.samples = []
        self._build()

    def _build(self):
        if self.clip_dirs is not None:
            clip_dirs = [Path(p) for p in self.clip_dirs]
        else:
            # Each clip dir has both anns.json (bboxes) and hoi-anns.json (segments)
            clip_dirs = [p.parent for p in self.clips_root.rglob("hoi-anns.json")]
        if not clip_dirs:
            print(f"No hoi-anns.json files found under {self.clips_root}")
            return

        skipped = 0
        no_segments = 0
        self._diag = {"segments_total": 0, "label_unknown": 0, "too_short": 0,
                      "labels_seen": set(), "no_object_category": 0}

        for clip_dir in clip_dirs:
            ann_path = clip_dir / "anns.json"
            hoi_path = clip_dir / "hoi-anns.json"

            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann_data = json.load(f, strict=False)
                with open(hoi_path, "r", encoding="utf-8") as f:
                    hoi_data = json.load(f, strict=False)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Skipping {clip_dir}: {e}")
                skipped += 1
                continue

            segments = hoi_data.get("annotations", [])
            if not segments:
                no_segments += 1
                continue

            self._process_clip(ann_data, segments)

        print(f"Built {len(self.samples)} sequences from {len(clip_dirs) - skipped} clips (skipped {skipped})")
        print(f"  Clips with no segments: {no_segments}")
        print(f"  Total segments seen: {self._diag['segments_total']}")
        print(f"  Dropped - unknown label: {self._diag['label_unknown']}")
        print(f"  Dropped - too short (< {self.seq_len}): {self._diag['too_short']}")
        print(f"  Dropped - object category not findable: {self._diag['no_object_category']}")
        print(f"  Action labels seen: {sorted(self._diag['labels_seen'])}")
        print(f"  Labels expected: {self.class_names}")

    def _process_clip(self, ann_data, segments):
        # Group bbox annotations by frame for fast lookup
        anns_by_frame = {}
        for ann in ann_data.get("annotations", []):
            anns_by_frame.setdefault(ann["image_id"], []).append(ann)

        # Build a track_id -> object_category_id index for objects only
        # (humans don't need this since their category is always PERSON_CAT_ID)
        track_to_cat = _build_object_track_to_category(ann_data)

        for seg in segments:
            self._diag["segments_total"] += 1

            action_id = seg.get("action_id")
            label = self.action_id_to_name.get(action_id)
            self._diag["labels_seen"].add(label)

            if label not in self.class_to_idx:
                self._diag["label_unknown"] += 1
                continue

            human_track = seg["person_track_id"]
            object_track = seg.get("object_track_id", -1)
            start = seg["start_frame"]
            end = seg["end_frame"]

            # -1 means no associated object (idle)
            if object_track == -1:
                object_track = None
                object_cat = None
            else:
                object_cat = track_to_cat.get(object_track)
                if object_cat is None:
                    self._diag["no_object_category"] += 1
                    continue

            frames_data = _collect_frames(
                anns_by_frame, human_track, object_track, object_cat, start, end
            )

            if len(frames_data) < self.seq_len:
                self._diag["too_short"] += 1
                continue

            label_idx = self.class_to_idx[label]
            for i in range(0, len(frames_data) - self.seq_len + 1, self.stride):
                window = frames_data[i:i + self.seq_len]
                feats = self.feature_extractor.extract_sequence(window)
                self.samples.append((feats, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        x = torch.tensor(feats, dtype=torch.float32) # (seq_len, feature_dim)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# Loads action_classes.json (action_name -> action_id) and inverts it
def _load_action_id_to_name(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"action_classes.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    return {int(v): k for k, v in name_to_id.items()}


# Walk the bbox annotations once, mapping object track_id -> first seen category_id
def _build_object_track_to_category(ann_data):
    track_to_cat = {}
    for ann in ann_data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id == PERSON_CAT_ID:
            continue
        track_id = ann.get("attributes", {}).get("track_id")
        if track_id is None:
            continue
        # Only set the first time we see this track_id
        if track_id not in track_to_cat:
            track_to_cat[track_id] = cat_id
    return track_to_cat


# Shared helpers - also used by the single-frame dataset
def _collect_frames(anns_by_frame, human_track, object_track, object_cat, start, end):
    frames_data = []
    for frame_id in range(start, end + 1):
        anns = anns_by_frame.get(frame_id, [])

        human_bbox = _find_bbox_by_track(anns, human_track, PERSON_CAT_ID)
        if human_bbox is None:
            continue

        object_bbox = None
        object_class_id = None
        if object_track is not None and object_cat is not None:
            object_bbox = _find_bbox_by_track(anns, object_track, object_cat)
            if object_bbox is not None:
                object_class_id = object_class_to_index(object_cat)

        frames_data.append({
            "human_bbox": human_bbox,
            "object_bbox": object_bbox,
            "object_class_id": object_class_id
        })
    return frames_data


# Bboxes in anns.json are stored in COCO xywh, this returns xyxy
def _find_bbox_by_track(anns, track_id, category_id):
    for ann in anns:
        if ann.get("category_id") != category_id:
            continue
        attrs = ann.get("attributes", {})
        if attrs.get("track_id") != track_id:
            continue
        x, y, w, h = ann["bbox"]
        return [x, y, x + w, y + h]
    return None
