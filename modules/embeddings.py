# modules/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings

    def embed_single(self, text: str):
        return self.embed([text])[0]
