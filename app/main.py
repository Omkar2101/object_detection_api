from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List

app = FastAPI()

class DetectionResult(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]


# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

@app.post("/detect", response_model=DetectionResult)
async def detect_objects(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)  # Use np.frombuffer
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference
    results = model.predict(img)  # Returns a Results object

    # Extract predictions
    predictions = results[0]  # First image's results
    boxes = predictions.boxes.xyxy.cpu().numpy().tolist()  # Bounding boxes
    scores = predictions.boxes.conf.cpu().numpy().tolist()  # Confidence scores /probality score
    labels = [model.names[int(cls)] for cls in predictions.boxes.cls.cpu().numpy()]  # Class labels

    # Draw bounding boxes on the image
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{labels[i]}: {scores[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the output image locally
    output_image_path = "output.jpg"
    cv2.imwrite(output_image_path, img)

    # Return detection results
    return DetectionResult(boxes=boxes, labels=labels, scores=scores)
