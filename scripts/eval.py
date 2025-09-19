from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

metrics = model.val(
    data="yolo_params.yaml",
    imgsz=1280,
    augment=True,
    workers=0  # <- avoid multiprocessing on Windows
)

print(metrics)