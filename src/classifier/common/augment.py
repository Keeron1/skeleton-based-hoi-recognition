import numpy as np

# COCO 17 left-right pairs for horizontal flip
LR_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]

class SkeletonAugment:
    def __init__(self, hflip_p=0.5, jitter_std=0.02, obj_dropout_p=0.1,
                 num_keypoints=17, object_node_idx=17):
        self.hflip_p = hflip_p
        self.jitter_std = jitter_std
        self.obj_dropout_p = obj_dropout_p
        self.num_keypoints = num_keypoints
        self.object_node_idx = object_node_idx

    def __call__(self, x):
        x = x.copy()

        if np.random.rand() < self.hflip_p:
            x = self._hflip(x)

        if self.jitter_std > 0:
            # Jitter x and y on every node, including the object node.
            # Confidence and one-hot channels are untouched.
            noise = np.random.normal(0, self.jitter_std, x[..., :2].shape).astype(np.float32)
            x[..., :2] += noise

        if np.random.rand() < self.obj_dropout_p:
            x[:, self.object_node_idx, :] = 0.0

        return x

    def _hflip(self, x):
        # Negate x for every node, then swap left / right keypoint slots
        x[..., 0] = -x[..., 0]
        for a, b in LR_PAIRS:
            tmp = x[:, a, :].copy()
            x[:, a, :] = x[:, b, :]
            x[:, b, :] = tmp
        return x
