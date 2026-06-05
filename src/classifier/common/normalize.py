import numpy as np

LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


# Pelvis from the first frame is the origin so motion across the window
# is preserved (idle vs reading is mostly about how much things move).
# Scale by the median torso length so close / far people look the same size.
#
# Channels: [0]=x, [1]=y, [2]=conf, [3]=w, [4]=h, [5:]=untouched extras.
# x, y get pelvis-centered AND scaled.
# w, h get scaled only (size is invariant to translation).
def normalize_window(window):
    coords = window[..., :2].astype(np.float32).copy()
    conf   = window[..., 2:3].astype(np.float32).copy()
    size   = window[..., 3:5].astype(np.float32).copy()
    extras = window[..., 5:].astype(np.float32).copy()

    pelvis = (coords[0, LEFT_HIP] + coords[0, RIGHT_HIP]) / 2.0

    hips      = (coords[:, LEFT_HIP]      + coords[:, RIGHT_HIP])      / 2.0
    shoulders = (coords[:, LEFT_SHOULDER] + coords[:, RIGHT_SHOULDER]) / 2.0
    torso = np.linalg.norm(shoulders - hips, axis=-1)
    scale = float(np.median(torso)) + 1e-6

    coords = (coords - pelvis) / scale
    size = size / scale

    # Zero-conf nodes carry no meaningful position. Wipe them so the
    # post-normalize values don't pretend to encode a real location.
    mask = (conf > 0).astype(np.float32)
    coords = coords * mask
    size = size * mask

    return np.concatenate([coords, conf, size, extras], axis=-1)
