import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalRAGIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []          # chunks
        self.metadatas = []      # source, page, filename, etc.       
    
    def add_documents(self, docs):
        """
        docs: list[dict] -> {"text": "...", "meta": {...}}
        """
        new_texts = [d["text"] for d in docs]
        embeddings = self.model.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors

        self.index.add(embeddings)
        self.texts.extend(new_texts)
        self.metadatas.extend([d.get("meta", {}) for d in docs])

    def search(self, query: str, k: int = 5):
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "text": self.texts[idx],
                "meta": self.metadatas[idx],
            })
        return results
