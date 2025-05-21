# from fastapi.testclient import TestClient
# import sys
# import os

# from pathlib import Path

# # Add the project root to PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from app.main import app  # adjust import path as needed

# client = TestClient(app)

# def test_detect_endpoint():
#     image_path = Path("test_images/dog_cat.jpg")
#     with open(image_path, "rb") as img_file:
#         files = {"file": ("dog_cat.jpg", img_file, "image/jpeg")}
#         response = client.post("/detect", files=files)

#     assert response.status_code == 200
#     data = response.json()
#     assert "boxes" in data
#     assert isinstance(data["boxes"], list)
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
    
    # Check for labels and scores instead of boxes
    assert "labels" in data
    assert "scores" in data
    assert isinstance(data["labels"], list)
    assert isinstance(data["scores"], list)
    
    # Make sure the lists are not empty (assuming the test image has detectable objects)
    assert len(data["labels"]) > 0
    assert len(data["scores"]) > 0
    
    # Optionally, verify that both lists have the same length
    assert len(data["labels"]) == len(data["scores"])