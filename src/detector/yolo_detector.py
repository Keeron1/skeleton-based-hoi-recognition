from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_type_path, project_root = None):
        self.model = YOLO(model_type_path)
        self.project_root = project_root
        
        # Display model information
        self.model.info()

    def predict(self, frame):
        results = self.model(
            source=frame, 
            save=True, 
            verbose=True
        )
        return results
    
    # Trains on format: class x_center y_center width height
    def train(self, dataset):
        results = self.model.train(
            project=str(self.project_root / "runs"), # Set name of project directory where training outputs are saved
            data=dataset,
            pretrained=True,
            epochs=100,
            batch=16,
            imgsz=640
        )
        return results