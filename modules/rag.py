import os
import tempfile
from PyPDF2 import PdfReader
import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_FILE = "vector_store.pkl"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def process_uploaded_files(files):
    """Extract text, embed, and save FAISS index."""
    texts = []
    for file in files:
        if file.type == "application/pdf":
            pdf = PdfReader(file)
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        elif file.type in ["text/plain"]:
            texts.append(file.read().decode("utf-8"))
        elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            from docx import Document
            doc = Document(file)
            for para in doc.paragraphs:
                texts.append(para.text)

    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, texts), f)

def retrieve_context(query, top_k=3):
    """Retrieve top_k relevant chunks for a query."""
    if not os.path.exists(INDEX_FILE):
        return ""
    with open(INDEX_FILE, "rb") as f:
        index, texts = pickle.load(f)
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, top_k)
    return "\n\n".join([texts[i] for i in indices[0] if i < len(texts)])
