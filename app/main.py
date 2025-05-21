
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List

app = FastAPI()

class DetectionResult(BaseModel):
    labels: List[str]
    scores: List[float]

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

@app.post("/detect", response_model=DetectionResult)
async def detect_objects(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference
    results = model.predict(img)

    # Extract predictions
    predictions = results[0]
    scores = predictions.boxes.conf.cpu().numpy().tolist()
    labels = [model.names[int(cls)] for cls in predictions.boxes.cls.cpu().numpy()]

    # Return only labels and scores
    return DetectionResult(labels=labels, scores=scores)
