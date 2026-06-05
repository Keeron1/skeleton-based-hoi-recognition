from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_type_path, project_root):
        # YOLO() loads pretrained weights only when given a .pt file
        # (e.g. yolov8m.pt). A .yaml path builds the architecture from
        # scratch with random init.
        self.model = YOLO(model_type_path)
        self.project_root = project_root

        # Display model information
        self.model.info()

    def predict(self, frame, save=False, verbose=True):
        results = self.model(
            source=frame,
            save=save,
            verbose=verbose
        )
        return results
    
    # Trains on format: class x_center y_center width height
    def train(self, dataset, imgsz):
        results = self.model.train(
            project=str(self.project_root / "runs"), # Set name of project directory where training outputs are saved
            data=dataset,
            pretrained=True,
            epochs=100,
            batch=8,
            patience=20, # stop training if the model starts overfitting
            imgsz=imgsz,

            # Geometric — fixed CCTV camera, no rotation/shear/perspective
            degrees=0.0,      # default 0.0
            shear=0.0,        # default 0.0
            perspective=0.0,  # default 0.0
            flipud=0.0,       # default 0.0
            fliplr=0.0,       # default 0.5
            scale=0.25,       # default 0.5
            translate=0.1,    # default 0.1

            # Color — hsv_v handles brightness variation across lighting conditions
            hsv_h=0.015,      # default 0.015
            hsv_s=0.7,        # default 0.7
            hsv_v=0.4,        # default 0.4

            # Composition — mosaic still useful for small datasets, off in last 10 epochs
            mosaic=1,       # default 1.0 - combine 4 images together
            close_mosaic=10,  # default 10
            mixup=0.0,        # default 0.0
            copy_paste=0.0,   # default 0.0
        )
        return results