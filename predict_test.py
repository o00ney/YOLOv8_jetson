from ultralytics import YOLO

model = YOLO("yolov8s.engine", task="detect")

for _ in model.predict(
    source="test.mp4",
    stream=True,
    save=True,
    name="test",
):
    pass
