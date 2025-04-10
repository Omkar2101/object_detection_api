from fastapi.testclient import TestClient
import sys
import os

from pathlib import Path

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app  # adjust import path as needed

client = TestClient(app)

def test_detect_endpoint():
    image_path = Path("test_images/dog_cat.jpg")
    with open(image_path, "rb") as img_file:
        files = {"file": ("dog_cat.jpg", img_file, "image/jpeg")}
        response = client.post("/detect", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "boxes" in data
    assert isinstance(data["boxes"], list)
