from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self, project_root = None):
        self.tracker = DeepSort(
            # Read values from cfg
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0
        )
        self.project_root = project_root

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
    
    def yolo_to_deepsort(self, results):
        detections = []

        boxes = results[0].boxes
        if boxes is None:
            return detections

        bbox_xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        names = results[0].names

        for i in range(len(bbox_xywh)):
            x_c, y_c, w, h = bbox_xywh[i]

            x1 = x_c - w / 2
            y1 = y_c - h / 2
            
            class_name = names[int(cls_ids[i])]

            detections.append((
                [x1, y1, w, h],
                float(conf[i]),
                # int(cls_ids[i])
                class_name
            ))

        return detections