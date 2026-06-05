import torch

# COCO 17 keypoint indices for reference:
#  0 nose         1 l_eye       2 r_eye      3 l_ear     4 r_ear
#  5 l_shoulder   6 r_shoulder  7 l_elbow    8 r_elbow
#  9 l_wrist     10 r_wrist    11 l_hip     12 r_hip
# 13 l_knee      14 r_knee     15 l_ankle   16 r_ankle

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
]

OBJECT_NODE = 17
LEFT_WRIST = 9
RIGHT_WRIST = 10


# Symmetric (N, N) bool tensor. True means an edge between i and j.
# Self loops included so each node attends to itself inside GATv2.
def build_adjacency(num_nodes=18):
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)

    for i, j in COCO_EDGES:
        adj[i, j] = True
        adj[j, i] = True

    # The object node hooks into both wrists. Where the action signal lives.
    adj[OBJECT_NODE, LEFT_WRIST] = True
    adj[LEFT_WRIST, OBJECT_NODE] = True
    adj[OBJECT_NODE, RIGHT_WRIST] = True
    adj[RIGHT_WRIST, OBJECT_NODE] = True

    for i in range(num_nodes):
        adj[i, i] = True

    return adj
