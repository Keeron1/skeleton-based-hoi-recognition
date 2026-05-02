from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self, 
                 max_age=30, 
                 n_init=3, 
                 nn_budget=None,
                 max_cosine_distance=0.2,
                 max_iou_distance=0.7,
                 nms_max_overlap=1.0):
        self.tracker = DeepSort(
            max_age=max_age, # How long a track survives without detections
            n_init=n_init, # Frames required before a track is "confirmed"
            nn_budget=nn_budget, # Size of feature gallery per track
            max_cosine_distance=max_cosine_distance, # Threshold for appearance matching (ReID embeddings)
            max_iou_distance=max_iou_distance, # Fallback matching using IoU when appearance fails
            nms_max_overlap=nms_max_overlap # Removes duplicates based on IoU overlap
        )

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
    
    # this is used for testing
    def yolo_to_deepsort_split(self, results):
        # DeepSORT expects tuples of ( [left,top,w,h], confidence, detection_class )
        human_detections = []
        object_detections = []

        boxes = results[0].boxes
        if boxes is None:
            return [], []

        bbox_xyxy = boxes.xyxy.cpu().numpy()
        bbox_xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        names = results[0].names

        for i in range(len(bbox_xywh)):
            cls_id = int(cls_ids[i])
            class_name = names[cls_id]
            
            # Sort human and non-human
            if class_name != "person":
                x1, y1, x2, y2 = bbox_xyxy[i]
                object_detections.append((
                    [x1, y1, x2, y2],
                    float(conf[i]),
                    class_name
                ))
                continue

            x_c, y_c, w, h = bbox_xywh[i]
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            
            human_detections.append((
                [x1, y1, w, h],
                float(conf[i]),
                # int(cls_ids[i])
                class_name,
                cls_id
            ))

        return human_detections, object_detections
    
    def yolo_to_deepsort(self, detections):
        # DeepSORT expects tuples of ( [left,top,w,h], confidence, detection_class )
        results = []

        boxes = detections[0].boxes
        if boxes is None:
            return []

        bbox_xyxy = boxes.xyxy.cpu().numpy()
        bbox_xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        # names = detections[0].names

        for i in range(len(bbox_xywh)):
            cls_id = int(cls_ids[i])
            # class_name = names[cls_id]

            x1, y1, x2, y2 = bbox_xyxy[i]
            x_c, y_c, w, h = bbox_xywh[i]
            
            results.append((
                [x1, y1, w, h],
                float(conf[i]),
                cls_id
            ))

        return results
    
    
    def extract_deepsort_results(self, deepsort_results):
        if len(deepsort_results) == 0:
            return {
                "bboxes": [],
                "track_ids": [],
                "class_names": []
            }
        
        bboxes = []
        track_ids = []
        class_names = []
        
        # If the track is confirmed then extract its bbox, id, and class name
        for track in deepsort_results:
            if not track.is_confirmed():
                continue

            bbox = track.to_ltrb()
            track_id = int(track.track_id)
            class_name = track.get_det_class()
            
            bboxes.append(bbox)
            track_ids.append(track_id)
            class_names.append(class_name)
  
        return {
            "bboxes": bboxes,
            "track_ids": track_ids,
            "class_names": class_names
        }
    
    def run_deepsort(self, detections, frame):
        deepsort_results = self.tracker.update(detections, frame)
        return self.extract_deepsort_results(deepsort_results)
        
        