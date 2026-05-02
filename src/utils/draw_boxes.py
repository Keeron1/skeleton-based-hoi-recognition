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
    
    # x1, y1, x2, y2
    def draw_box(self, frame, bbox, class_name, class_id=None, track_id=None):
        x1, y1, x2, y2 = map(int, bbox) # bbox coords
        label_color = (255, 0, 0) # default label color

        # Format label
        label = f'{class_name} '
        
        # if track id exists (human)
        if track_id is not None:
            label += f'ID {track_id} '
            label_color = compute_color_for_labels(track_id)  
        # if track id doesn't exist and class id does (object)
        elif not class_id is None:
            label_color = compute_color_for_labels(class_id)  

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)

        # Draw label background
        (text_width, text_height),_ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width, y1), label_color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 4), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame