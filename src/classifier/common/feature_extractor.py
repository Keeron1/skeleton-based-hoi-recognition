from src.utils.bbox_utils import BboxUtils

# Builds per-frame feature vectors that describe a (human, object) pair
# These vectors are the SAME for the LSTM and the single-frame baseline
# - LSTM stacks T of these into a sequence
# - Single-frame uses one at a time
# Inspired by Cob-Parro et al., where lightweight bbox features replace
# raw image input
class FeatureExtractor:
    def __init__(self, num_object_classes, frame_width=1920, frame_height=1080):
        self.num_object_classes = num_object_classes
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.bbox_utils = BboxUtils()

    # Normalize bbox coords to [0, 1] using frame dimensions
    def _normalize_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return [
            x1 / self.frame_width,
            y1 / self.frame_height,
            x2 / self.frame_width,
            y2 / self.frame_height
        ]

    # Get center point of a bbox
    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # One-hot encode the object class id (0-indexed)
    def _one_hot(self, object_class_id):
        vec = [0.0] * self.num_object_classes
        if object_class_id is not None and 0 <= object_class_id < self.num_object_classes:
            vec[object_class_id] = 1.0
        return vec

    # Build a single per-frame feature vector
    # human_bbox / object_bbox are in xyxy format
    # If object_bbox is None (no object present), zero-fill those slots
    def extract(self, human_bbox, object_bbox=None, object_class_id=None):
        # Normalized human bbox (4)
        h_norm = self._normalize_bbox(human_bbox)

        if object_bbox is None:
            o_norm = [0.0, 0.0, 0.0, 0.0]
            dx, dy = 0.0, 0.0
            iou = 0.0
            obj_one_hot = [0.0] * self.num_object_classes
        else:
            o_norm = self._normalize_bbox(object_bbox)

            # Relative center distance, normalized
            hc = self._bbox_center(human_bbox)
            oc = self._bbox_center(object_bbox)
            dx = (oc[0] - hc[0]) / self.frame_width
            dy = (oc[1] - hc[1]) / self.frame_height

            # IoU between human and object bboxes
            iou = self.bbox_utils.get_iou(human_bbox, object_bbox)

            # One-hot of object class
            obj_one_hot = self._one_hot(object_class_id)

        # Final feature vector: 4 + 4 + 2 + 1 + num_object_classes
        return h_norm + o_norm + [dx, dy, iou] + obj_one_hot

    # Build a sequence of feature vectors for one (human, object) pair across frames
    # frames_data is a list of dicts: { "human_bbox", "object_bbox" or None, "object_class_id" or None }
    def extract_sequence(self, frames_data):
        return [
            self.extract(
                f["human_bbox"],
                f.get("object_bbox"),
                f.get("object_class_id")
            )
            for f in frames_data
        ]
