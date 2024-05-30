from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("teacher_model.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data="mot17.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)