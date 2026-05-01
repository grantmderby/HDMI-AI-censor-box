from ultralytics import YOLO
model = YOLO('yolo26n-seg.pt')
model.export(format='openvino', imgsz=640, half=True)