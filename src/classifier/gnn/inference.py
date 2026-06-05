import glob
import time
from pathlib import Path
from contextlib import contextmanager
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.classifier.common.constants import ACTION_NAMES, NAME_TO_CAT_ID
from src.classifier.common.features import build_node_features
from src.classifier.common.normalize import normalize_window
from src.classifier.common.graph import COCO_EDGES
from src.utils.draw_boxes import DrawBoxes, compute_color_for_labels


# Rolling per-track history. Person buffer holds (frame_id, kpts), object buffer
# holds (frame_id, cx, cy, w, h, cat_id). The GNN needs seq_len frames per pair
# so we accumulate one buffer per track and slice the last seq_len at query time.
class TrackBuffers:
    def __init__(self, maxlen):
        self.person = defaultdict(lambda: deque(maxlen=maxlen))
        self.object = defaultdict(lambda: deque(maxlen=maxlen))
        self.person_last_bbox = {}

    def add_person(self, track_id, frame_id, bbox, kpts):
        self.person[track_id].append((frame_id, kpts))
        self.person_last_bbox[track_id] = bbox

    def add_object(self, track_id, frame_id, bbox, cat_id):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        self.object[track_id].append((frame_id, cx, cy, w, h, cat_id))


# Yields (frame_id, image) from a directory of frames or an .mp4 file.
def frame_iterator(source):
    source = Path(source)
    if source.is_dir():
        paths = sorted(glob.glob(str(source / "*.jpg")) + glob.glob(str(source / "*.png")))
        for i, p in enumerate(paths):
            frame = cv2.imread(p)
            if frame is not None:
                yield i, frame
    else:
        cap = cv2.VideoCapture(str(source))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
        cap.release()


# Object center inside the person bbox expanded by margin pixels.
def is_nearby(person_bbox, obj_bbox, margin):
    px1, py1, px2, py2 = person_bbox
    ox1, oy1, ox2, oy2 = obj_bbox
    ocx = (ox1 + ox2) / 2.0
    ocy = (oy1 + oy2) / 2.0
    return (px1 - margin <= ocx <= px2 + margin and
            py1 - margin <= ocy <= py2 + margin)


# Assembles a (T, N, C) window using the person's last seq_len visible frames.
# Calls the same build_node_features that the training dataset uses, so the
# input layout to the GNN is identical at train and inference.
def build_window(buffers, person_id, object_id, seq_len):
    person_history = list(buffers.person[person_id])
    if len(person_history) < seq_len:
        return None

    recent = person_history[-seq_len:]

    obj_lookup = {}
    if object_id is not None:
        for fid, cx, cy, w, h, cat in buffers.object[object_id]:
            obj_lookup[fid] = (cx, cy, w, h, cat)

    window = np.stack([
        build_node_features(kpts, obj_lookup.get(fid))
        for fid, kpts in recent
    ])
    return normalize_window(window)


# Per-stage timing accumulator. Calls torch.cuda.synchronize() around each
# block when on GPU so async kernel launches don't get under-counted.
class StageTimer:
    def __init__(self, device):
        self.device = device
        self.records = defaultdict(list)

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.records[name].append(time.perf_counter() - t0)

    def summary(self):
        out = {}
        for name, samples in self.records.items():
            if not samples:
                continue
            mean_ms = (sum(samples) / len(samples)) * 1000
            out[name] = round(mean_ms, 2)
        return out


# Full end-to-end inference pipeline. Holds the stateless models (detector,
# pose, gnn). A fresh tracker is created per run() call because DeepSORT is
# stateful and shouldn't carry context across clips.
class HOIPipeline:
    def __init__(self, detector, tracker_factory, pose_estimator, gnn, adj, device, seq_len):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.pose_estimator = pose_estimator
        self.gnn = gnn
        self.adj = adj
        self.device = device
        self.seq_len = seq_len
        self.drawer = DrawBoxes()

    def set_gnn(self, gnn):
        self.gnn = gnn

    @torch.no_grad()
    def _predict(self, window):
        x = torch.from_numpy(window).unsqueeze(0).to(self.device)
        logits = self.gnn(x, self.adj)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        return ACTION_NAMES[idx.item()], conf.item()

    def run(self, source, output_path,
            fps=5, pair_margin_px=60, min_conf=0.5):
        tracker = self.tracker_factory()
        buffers = TrackBuffers(maxlen=self.seq_len * 4)
        timer = StageTimer(self.device)
        writer = None
        stats = {"n_frames": 0, "n_predictions": 0, "action_counts": defaultdict(int)}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for frame_id, frame in frame_iterator(source):
            stats["n_frames"] += 1
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            with timer.stage("total"):
                self._process_frame(frame, frame_id, tracker, buffers,
                                    pair_margin_px, min_conf, stats, timer)
            writer.write(frame)

        if writer is not None:
            writer.release()

        stage_ms = timer.summary()
        total_ms = stage_ms.get("total", 0.0)

        stats["action_counts"] = dict(stats["action_counts"])
        stats["output"] = str(output_path)
        stats["stage_ms"] = stage_ms
        stats["fps"] = round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0
        return stats

    def _process_frame(self, frame, frame_id, tracker, buffers,
                       pair_margin_px, min_conf, stats, timer):

        with timer.stage("yolo"):
            yolo_results = self.detector.predict(frame, save=False, verbose=False)

        with timer.stage("tracker"):
            person_dets, object_dets = tracker.yolo_to_deepsort_split(yolo_results)
            persons, objects = tracker.run_split(person_dets + object_dets, frame)

        with timer.stage("pose"):
            if persons:
                person_bboxes = [bbox for _, bbox in persons]
                pose_results = self.pose_estimator.infer(frame, person_bboxes)
                for (pid, bbox), sample in zip(persons, pose_results):
                    kpts_xy = sample.pred_instances.keypoints[0]
                    kpts_conf = sample.pred_instances.keypoint_scores[0]
                    kpts = np.concatenate([kpts_xy, kpts_conf[:, None]], axis=1).astype(np.float32)
                    buffers.add_person(pid, frame_id, bbox, kpts)

        objects_with_cat = []
        for oid, obox, oname in objects:
            cat_id = NAME_TO_CAT_ID.get(oname)
            if cat_id is None:
                continue
            buffers.add_object(oid, frame_id, obox, cat_id)
            objects_with_cat.append((oid, obox, oname))

        # For each person, run the baseline + each nearby-object pair
        candidates = defaultdict(list)
        with timer.stage("gnn"):
            for pid, pbbox in persons:
                self._query_pair(pid, None, None, buffers, candidates, min_conf, stats)
                for oid, obox, oname in objects_with_cat:
                    if is_nearby(pbbox, obox, pair_margin_px):
                        self._query_pair(pid, oid, oname, buffers, candidates, min_conf, stats)

        with timer.stage("render"):
            for pid, pbbox in persons:
                color = compute_color_for_labels(pid)
                action = _best_action(candidates[pid])
                self.drawer.draw_box(frame, pbbox, "person", track_id=pid, action=action)
                history = buffers.person[pid]
                if history:
                    self.drawer.draw_skeleton(frame, history[-1][1], COCO_EDGES, color)

            for oid, obox, oname in objects_with_cat:
                self.drawer.draw_box(frame, obox, oname, track_id=oid)

    def _query_pair(self, person_id, object_id, object_name,
                    buffers, candidates, min_conf, stats):
        window = build_window(buffers, person_id, object_id, self.seq_len)
        if window is None:
            return

        name, conf = self._predict(window)
        if conf < min_conf:
            return

        # Drop idle for pair queries. Many nearby objects -> many idles -> noise.
        # The baseline (no object) query still surfaces idle when nothing fires.
        if object_id is not None and name == "idle":
            return

        label = f"{name} {conf:.2f}"
        candidates[person_id].append((name, conf, label))
        stats["action_counts"][name] += 1
        stats["n_predictions"] += 1


# Prefers non-idle actions if any exist, falling back to baseline idle.
# Returns the label text or None when no candidate passed the confidence filter.
def _best_action(candidates):
    if not candidates:
        return None
    non_idle = [c for c in candidates if c[0] != "idle"]
    pool = non_idle if non_idle else candidates
    return max(pool, key=lambda c: c[1])[2]
