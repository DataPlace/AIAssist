# modules/vectorstore.py
import faiss
import numpy as np
import os
import pickle
from typing import List

class FaissIndex:
    def __init__(self, dim: int = None, index_path: str = None):
        self.dim = dim
        self.index_path = index_path
        if dim:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
        self.metadatas = []

    def add(self, embeddings: np.ndarray, metadatas: List[dict]):
        faiss.normalize_L2(embeddings)
        embeddings = embeddings.astype('float32')
        if self.index is None:
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.metadatas.extend(metadatas)

    def search(self, query_emb: np.ndarray, k: int = 5):
        if self.index is None:
            return [[]]
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb.astype('float32'), k)
        results = []
        for row_i in range(I.shape[0]):
            items = []
            for idx, score in zip(I[row_i], D[row_i]):
                if idx < 0 or idx >= len(self.metadatas):
                    continue
                items.append({"metadata": self.metadatas[idx], "score": float(score)})
            results.append(items)
        return results

    def save(self, base_path: str):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        if self.index is None:
            return
        faiss.write_index(self.index, base_path + ".index")
        with open(base_path + ".meta", "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self, base_path: str):
        self.index = faiss.read_index(base_path + ".index")
        with open(base_path + ".meta", "rb") as f:
            self.metadatas = pickle.load(f)
        self.dim = self.index.d
