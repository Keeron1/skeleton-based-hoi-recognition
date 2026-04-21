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
        
    # Change to select label color depending on class id    
    def draw_detection(self, frame, bbox, class_name):
        x1, y1, x2, y2 = map(int, bbox)
        label_color = (255, 0, 0)

        # Label text
        label = f"{class_name}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
        
        # Draw label background
        (text_width, text_height),_ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width, y1), label_color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 4), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame
        
    def draw_track(self, frame, bbox, class_name, track_id):
        x1, y1, x2, y2 = map(int, bbox)
        label_color = compute_color_for_labels(track_id)

        # Label text
        label = f"{class_name} ID {track_id}"

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
        
        # Draw label background
        (text_width, text_height),_ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width, y1), label_color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 4), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame
    
    # Frame, Bboxes, Class_names, track_ids
    def draw_boxes_tracking(self, frame, bboxes, names, identities=None):
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(i) for i in box] # bbox coords
            track_id = int(identities[i]) if identities is not None else 0
            label_color = compute_color_for_labels(track_id)

            # Format label
            label = f'{names[i]} '
            if not identities is None:
                label += f'{track_id}'

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)

            # Draw label background
            (text_width, text_height),_ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width, y1), label_color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 4), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame