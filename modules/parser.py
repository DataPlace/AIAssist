# modules/parser.py
import io
from typing import List, Dict
import pdfplumber
from docx import Document

def parse_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text()
            if ptext:
                text.append(ptext)
    return "\n".join(text)

def parse_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras)

def parse_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def parse_file(file_name: str, file_bytes: bytes) -> str:
    name = file_name.lower()
    if name.endswith(".pdf"):
        return parse_pdf(file_bytes)
    if name.endswith(".docx"):
        return parse_docx(file_bytes)
    if name.endswith(".txt"):
        return parse_txt(file_bytes)
    raise ValueError("Unsupported file type: " + file_name)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i:i + chunk_size]
        chunks.append({"text": " ".join(chunk), "start_word": i})
        i += chunk_size - overlap
    return chunks
