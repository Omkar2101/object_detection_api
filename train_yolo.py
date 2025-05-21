from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolov8n.pt")

# Train on custom data
results = model.train(
    data="dataset/data.yaml",  # path to YAML
    epochs=20,
    imgsz=640,
    project="runs/train",      # save dir
    name="exp3",
    exist_ok=True              # overwrite if exists
)

# Evaluate on the validation set
metrics = model.val()
# metrics = results.metrics  # This is a named tuple or object
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

