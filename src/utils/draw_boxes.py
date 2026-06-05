import cv2

# pulled from deep_sort_pytorch
def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

class DrawBoxes:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1

    # x1, y1, x2, y2. action is an optional string appended to the label so the
    # caller can show e.g. "person ID 1 | using_phone 0.92" in one box.
    def draw_box(self, frame, bbox, class_name, class_id=None, track_id=None, action=None):
        x1, y1, x2, y2 = map(int, bbox)
        label_color = (255, 0, 0)

        label = f'{class_name}'

        if track_id is not None:
            label += f' ID {track_id}'
            label_color = compute_color_for_labels(track_id)
        elif class_id is not None:
            label_color = compute_color_for_labels(class_id)

        if action is not None:
            label += f' | {action}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)

        (text_width, text_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width, y1), label_color, -1)

        cv2.putText(frame, label, (x1, y1 - 4), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame

    # Draws lines between connected keypoints and dots at each visible one.
    # edges is the list of (i, j) joint index pairs (caller passes COCO_EDGES).
    def draw_skeleton(self, frame, kpts, edges, color, conf_thresh=0.3):
        for a, b in edges:
            ax, ay, ac = kpts[a]
            bx, by, bc = kpts[b]
            if ac < conf_thresh or bc < conf_thresh:
                continue
            cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, 2)
        for x, y, c in kpts:
            if c >= conf_thresh:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        return frame

