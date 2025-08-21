# detector.py
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
from collections import Counter
from io import BytesIO

def run_detection(uploaded_file, model_path="yolov8n.pt"):
    # Load YOLO once (you can optimize by caching globally if needed)
    model = YOLO(model_path)

    # Read bytes -> PIL
    raw_bytes = uploaded_file.read()
    img = Image.open(BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Run detection
    results = model(np.array(img))

    summary = {}
    annotated_img = None
    for r in results:
        annotated_img = Image.fromarray(r.plot())
        labels = [r.names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
        summary = dict(Counter(labels))

    return results, annotated_img, summary
