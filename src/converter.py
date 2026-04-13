from ultralytics import YOLO
model= YOLO('modelos/best.pt')
model.export(format ='onnx', imgsz=320, half=False, simplify=True)
model2=YOLO('modelos/yolov8n.pt')
model2.export(format='onnx', imgsz=320, half=False, simplify=True)
0