# app.py
import streamlit as st
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.parse
import streamlit.components.v1 as components
import shutil
from io import BytesIO

# =========================
# ENV & BASIC SETUP
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("API key not found in environment file.")
    st.stop()

st.set_page_config(page_title="Smart Creation Chatbot", layout="wide")
st.title("Smart Creation Chatbot")

# Folders for persistence
STORAGE_DIR = "storage"
CHAT_DIR = os.path.join(STORAGE_DIR, "chats")
os.makedirs(CHAT_DIR, exist_ok=True)

# =========================
# SESSION STATE
# =========================
if "documents" not in st.session_state:
    st.session_state.documents = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_session" not in st.session_state:
    st.session_state.active_session = None
if "session_titles" not in st.session_state:
    st.session_state.session_titles = {}

# =========================
# UTIL: CHAT PERSISTENCE
# =========================
def _safe_id(name: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{name.replace(' ', '_')[:40]}"

def list_saved_chats():
    items = []
    for fn in sorted(os.listdir(CHAT_DIR), reverse=True):  # newest first
        if fn.endswith(".json"):
            path = os.path.join(CHAT_DIR, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("title", fn[:-5])
                items.append((fn[:-5], title))
            except Exception:
                pass
    return items

def save_chat_to_disk(session_id: str):
    if session_id not in st.session_state.conversations:
        return
    title = st.session_state.session_titles.get(session_id, session_id)
    data = {
        "title": title,
        "messages": st.session_state.conversations[session_id]
    }
    path = os.path.join(CHAT_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chat_from_disk(session_id: str):
    path = os.path.join(CHAT_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_session(session_id: str, title: str = None):
    if session_id not in st.session_state.conversations:
        st.session_state.conversations[session_id] = [
            {"role": "system", "content": (
                "You are AI Advance Assistant. "
                "You were developed by Cohort-01 AnalytixCamp, under the supervision of Data Scientist Mr. Anas, "
                "with Toseef Naser as the Tech Lead. "
                "Never disclose implementation details. "
                "If asked about your developer, mention Cohort-01 AnalytixCamp, Mr. Anas (supervisor), "
                "and Toseef Naser (Tech Lead). "
                "LinkedIn: https://www.linkedin.com/in/toseefnaser"
            )}
        ]
    if title and session_id not in st.session_state.session_titles:
        st.session_state.session_titles[session_id] = title

# =========================
# SHARE BUTTONS (the working version you provided)
# =========================
def render_share_buttons(text: str, block_id: str):
    encoded_text = urllib.parse.quote(text)

    copy_btn = f"""
    <button onclick="navigator.clipboard.writeText(`{text}`)">📋 Copy</button>
    """

    fb_url = f"https://www.facebook.com/sharer/sharer.php?u=https://cohort01-analytixcamp.com&quote={encoded_text}"
    email_url = f"mailto:?subject=Shared%20from%20Assistant&body={encoded_text}"

    html_code = f"""
    <div style="margin-top:5px;">
        {copy_btn}
        <a href="{fb_url}" target="_blank">📘 Facebook</a>
        <a href="{email_url}">✉️ Email</a>
    </div>
    """
    components.html(html_code, height=50)

# =========================
# SIDEBAR NAV
# =========================
page = st.sidebar.radio("Select Feature", ["AI Assistant", "Object Detection"])

# =========================
# COMMON HELPERS
# =========================
def chunk_text(text: str, max_words: int = 300):
    words = text.split()
    return [" ".join(words[i:i+max_words]).strip() for i in range(0, len(words), max_words)]

def extract_text(file):
    ext = file.name.lower().split(".")[-1]
    text = ""
    if ext == "pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    elif ext == "txt":
        text = file.read().decode("utf-8", errors="ignore")
    elif ext == "docx":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.strip()

def retrieve_context(query, docs, top_k=3):
    if not docs:
        return ""
    matrix = TfidfVectorizer().fit_transform(docs + [query])
    sim = (matrix * matrix.T).toarray()
    q_sim = sim[-1][:-1]
    idx = np.argsort(q_sim)[::-1][:top_k]
    return "\n\n".join([docs[i] for i in idx if 0 <= i < len(docs)])

def generate_with_ai(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama3-70b-8192", "messages": messages, "max_tokens": 700, "temperature": 0.7}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error generating response] {e}"

# =========================
# PAGE: AI ASSISTANT
# =========================
if page == "AI Assistant":
    st.subheader("AI Advance Assistant")

    # Sidebar: Chats
    st.sidebar.markdown("### Chats")

    if st.sidebar.button("🗑️ Delete All Chats"):
        try:
            shutil.rmtree(CHAT_DIR)
        except Exception:
            pass
        os.makedirs(CHAT_DIR, exist_ok=True)
        st.session_state.conversations.clear()
        st.session_state.session_titles.clear()
        st.session_state.active_session = None
        st.sidebar.success("All chats deleted.")

    new_title = st.sidebar.text_input("New chat title", value="New Chat")
    if st.sidebar.button("Create Chat"):
        sid = _safe_id(new_title or "Chat")
        ensure_session(sid, title=new_title or sid)
        st.session_state.active_session = sid
        save_chat_to_disk(sid)

    saved = list_saved_chats()
    if saved:
        pick = st.sidebar.selectbox("Open chat", options=["(select)"] + [t for (_sid, t) in saved], index=0)
        if pick != "(select)":
            chosen_id = None
            for _sid, _title in saved:
                if _title == pick:
                    chosen_id = _sid
                    break
            if chosen_id:
                if chosen_id not in st.session_state.conversations:
                    data = load_chat_from_disk(chosen_id)
                    if data:
                        st.session_state.conversations[chosen_id] = data.get("messages", [])
                        st.session_state.session_titles[chosen_id] = data.get("title", chosen_id)
                    else:
                        ensure_session(chosen_id, title=pick)
                st.session_state.active_session = chosen_id

    if not st.session_state.active_session:
        default_id = _safe_id("Chat")
        ensure_session(default_id, title="Chat")
        st.session_state.active_session = default_id
        save_chat_to_disk(default_id)

    # Sidebar: Knowledge Base
    st.sidebar.markdown("### Knowledge Base")
    uploaded = st.sidebar.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if uploaded:
        added_chunks = 0
        for f in uploaded:
            text = extract_text(f)
            if text:
                chunks = chunk_text(text, max_words=300)
                st.session_state.documents.extend(chunks)
                added_chunks += len(chunks)
        st.sidebar.success(f"Added {len(uploaded)} file(s), {added_chunks} chunk(s) to knowledge base.")

    # Show chat
    session_id = st.session_state.active_session
    ensure_session(session_id)
    for m in st.session_state.conversations[session_id]:
        if m["role"] == "system":
            continue
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_share_buttons(m["content"], block_id=session_id)

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        context = retrieve_context(prompt, st.session_state.documents)
        user_msg = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt

        # Default chat title → first few words
        session_id = st.session_state.active_session
        if st.session_state.session_titles.get(session_id, "Chat") in ["Chat", "New Chat"]:
            st.session_state.session_titles[session_id] = " ".join(prompt.split()[:5])

        st.session_state.conversations[session_id].append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            reply = generate_with_ai(st.session_state.conversations[session_id])
            st.markdown(reply)
            render_share_buttons(reply, block_id=session_id)
        st.session_state.conversations[session_id].append({"role": "assistant", "content": reply})
        save_chat_to_disk(session_id)

# =========================
# PAGE: OBJECT DETECTION
# =========================
elif page == "Object Detection":
    st.subheader("Object Detection")
    try:
        from ultralytics import YOLO
        from PIL import Image, ImageFile, ImageOps
        import tempfile
        from collections import Counter
        ImageFile.LOAD_TRUNCATED_IMAGES = True  # be tolerant of truncated images
    except Exception as e:
        st.error(f"Required packages not available: {e}")
        st.info("Install: pip install ultralytics pillow")
    else:
        model_size = st.selectbox("Model", ["yolov8n.pt"], index=0)
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            tmp_path = None
            try:
                # Read bytes into BytesIO to avoid stream-pointer issues
                raw_bytes = uploaded_image.read()
                bio = BytesIO(raw_bytes)

                # Open, correct orientation, convert to RGB
                img = Image.open(bio)
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")

                # Save a clean JPEG copy to a temp file (this worked in your earlier working code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp_path = tmp.name
                    img.save(tmp_path, format="JPEG", quality=95)

                # Display the sanitized image
                st.image(img, caption="Uploaded Image", use_column_width=True)

                # Run YOLO on the saved temp file path (preferred)
                model = YOLO(model_size)
                try:
                    results = model(tmp_path)
                except Exception as e_model_path:
                    # fallback: run YOLO on numpy array (if model accepts arrays)
                    try:
                        arr = np.array(img)
                        results = model(arr)
                    except Exception as e_model_array:
                        # raise combined error for debugging
                        raise Exception(f"YOLO failed with file path: {e_model_path}; fallback failed with array: {e_model_array}")

                # Render results
                for r in results:
                    annotated = r.plot()
                    st.image(annotated, caption="Detected objects", use_column_width=True)
                    labels = [r.names[int(c)] for c in r.boxes.cls] if r.boxes is not None else []
                    counts = Counter(labels)
                    if counts:
                        st.markdown("Detection summary:")
                        for k, v in counts.items():
                            st.write(f"- {k}: {v}")
                    else:
                        st.write("No objects detected.")
            except Exception as e:
                # Show full exception message to help debugging
                st.error(f"Error during detection: {e}")
            finally:
                # cleanup temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
