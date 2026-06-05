import numpy as np

from src.classifier.common.constants import (
    OBJECT_CAT_IDS, NUM_KEYPOINTS, NUM_NODES, OBJECT_NODE_IDX,
    object_class_to_index,
)

# Single source of truth for the node feature layout.
# Channels: [0]=x, [1]=y, [2]=conf, [3]=w, [4]=h, [5:]=object class one-hot
NUM_OBJECT_CLASSES = len(OBJECT_CAT_IDS)
NODE_CHANNELS = 5 + NUM_OBJECT_CLASSES


# Per-frame node features for one (person, object_or_none) pair.
# kpts: (17, 3) with (x, y, conf)
# object_info: (cx, cy, w, h, cat_id) for the paired object in this frame, or None
# Used by both GNNHOIDataset at train time and the live pipeline at inference.
def build_node_features(kpts, object_info=None):
    nodes = np.zeros((NUM_NODES, NODE_CHANNELS), dtype=np.float32)
    nodes[:NUM_KEYPOINTS, :3] = kpts

    if object_info is not None:
        cx, cy, w, h, cat = object_info
        nodes[OBJECT_NODE_IDX, 0] = cx
        nodes[OBJECT_NODE_IDX, 1] = cy
        nodes[OBJECT_NODE_IDX, 2] = 1.0
        nodes[OBJECT_NODE_IDX, 3] = w
        nodes[OBJECT_NODE_IDX, 4] = h
        cls_idx = object_class_to_index(cat)
        if cls_idx is not None:
            nodes[OBJECT_NODE_IDX, 5 + cls_idx] = 1.0

    return nodes
