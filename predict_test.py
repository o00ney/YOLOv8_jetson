from ultralytics import YOLO

model = YOLO("yolov8s.pt", task="detect")

for _ in model.predict(
    source="test.mp4",
    stream=True,
    save=True,
    name="pt",
):
    pass
