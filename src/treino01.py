from ultralytics import YOLOWorld

# Load a model
model = YOLOWorld("yolov8s-worldv2.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="lvis.yaml", epochs=30, imgsz=640)