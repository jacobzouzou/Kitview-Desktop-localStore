import pickle
import faiss

from faissClass import LocalRAGIndex

def save_index(rag: LocalRAGIndex, faiss_path="index.faiss", meta_path="index_meta.pkl"):
    faiss.write_index(rag.index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": rag.texts, "metadatas": rag.metadatas}, f)

def load_index(rag: LocalRAGIndex, faiss_path="index.faiss", meta_path="index_meta.pkl"):
    rag.index = faiss.read_index(faiss_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    rag.texts = data["texts"]
    rag.metadatas = data["metadatas"]
