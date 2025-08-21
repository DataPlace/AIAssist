# pages/2_Object_Detection.py
import streamlit as st
from modules.detector import load_model, detect_from_bytes
from PIL import Image
import io

st.set_page_config(page_title="Object Detection", layout="centered")
st.title("Object Detection — YOLO (Advanced)")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    model_choice = st.selectbox("Model", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25)
    if uploaded:
        img_bytes = uploaded.getvalue()
        st.image(img_bytes, caption="Uploaded image", use_column_width=True)
        if st.button("Run Detection"):
            with st.spinner("Loading model and running detection..."):
                model = load_model(model_choice)
                annotated, detections = detect_from_bytes(img_bytes, model, conf=conf)
                st.image(annotated, caption="Annotated", use_column_width=True)
                st.markdown("**Detections**")
                if detections:
                    for d in detections:
                        st.write(f"{d['label']} — {d['score']:.2f} — bbox: {d['xyxy']}")
                else:
                    st.info("No objects detected above threshold.")

with col2:
    st.markdown("**Model info**")
    st.write("Ultralytics YOLOv8 — choose a model size. Smaller models are faster.")
    st.write("Weights will download automatically on first run (may take time).")
    st.markdown("**Download results**")
    if st.button("Export last annotated image"):
        try:
            buf = io.BytesIO()
            pil = Image.fromarray(annotated)
            pil.save(buf, format="PNG")
            st.download_button("Download Annotated PNG", data=buf.getvalue(), file_name="annotated.png", mime="image/png")
        except Exception as e:
            st.error("No annotated image available: " + str(e))
