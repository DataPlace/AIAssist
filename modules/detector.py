# modules/detector.py
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import io
import numpy as np

@st.cache_resource
def load_model(model_name: str = "yolov8n") -> YOLO:
    model = YOLO(model_name)
    return model

def detect_from_bytes(image_bytes: bytes, model: YOLO, conf: float = 0.25):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(img, conf=conf, imgsz=640)
    r = results[0]
    annotated = r.plot()
    detections = []
    if hasattr(r, "boxes"):
        for box in r.boxes:
            try:
                cls = int(box.cls[0].item())
                name = r.names.get(cls, str(cls))
                confscore = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                detections.append({"class_id": cls, "label": name, "score": confscore, "xyxy": xyxy})
            except Exception:
                continue
    return annotated, detections
