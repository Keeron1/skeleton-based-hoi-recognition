import json
from pathlib import Path

def cache_path(cache_root, vp, clip):
    return Path(cache_root) / str(vp) / f"{clip}.json"

def save(cache_root, vp, clip, frame_kpts):
    path = cache_path(cache_root, vp, clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(frame_kpts, f)

def load(cache_root, vp, clip):
    path = cache_path(cache_root, vp, clip)
    with open(path) as f:
        raw = json.load(f)
    return {
        int(frame): {int(tid): kpts for tid, kpts in by_track.items()}
        for frame, by_track in raw.items()
    }

def exists(cache_root, vp, clip):
    return cache_path(cache_root, vp, clip).exists()
