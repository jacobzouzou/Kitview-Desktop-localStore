import os, json
from typing import Dict, List, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sqlite3

KB_DIR = "./knowledge_base"
DB_PATH = os.path.join(KB_DIR, "store.sqlite")
TEXT_INDEX_PATH = os.path.join(KB_DIR, "faiss_text.index")
IMAGE_INDEX_PATH = os.path.join(KB_DIR, "faiss_image.index")
KB_META_JSON = os.path.join(KB_DIR, "kb_meta.json")

_CACHE = {
    "loaded": False,
    "text_index": None,
    "img_index": None,
    "text_model": None,
    "img_model": None,
}

def load_resources_once():
    if _CACHE["loaded"]:
        return

    meta = load_meta()
    _CACHE["text_index"] = faiss.read_index(TEXT_INDEX_PATH)
    _CACHE["img_index"] = faiss.read_index(IMAGE_INDEX_PATH)

    _CACHE["text_model"] = SentenceTransformer(meta["text_embedding_model"])
    _CACHE["img_model"] = SentenceTransformer(meta["image_embedding_model"])

    _CACHE["loaded"] = True

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_get_chunk(kind: str, faiss_id: int) -> Optional[Dict]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
    SELECT kind, source_file, page, text, image_path, faiss_id
    FROM chunks
    WHERE kind = ? AND faiss_id = ?
    LIMIT 1;
    """, (kind, faiss_id))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "kind": row[0],
        "source_file": row[1],
        "page": row[2],
        "text": row[3],
        "image_path": row[4],
        "faiss_id": row[5],
    }


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / norms).astype("float32")


def load_meta() -> Dict:
    with open(KB_META_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_indices():
    text_index = faiss.read_index(TEXT_INDEX_PATH)
    img_index = faiss.read_index(IMAGE_INDEX_PATH)
    meta = load_meta()
    text_model = SentenceTransformer(meta["text_embedding_model"])
    img_model = SentenceTransformer(meta["image_embedding_model"])
    return text_index, img_index, text_model, img_model


def retrieve_text(query: str, top_k: int = 8) -> List[Dict]:
    load_resources_once()
    text_index = _CACHE["text_index"]
    text_model = _CACHE["text_model"]

    qv = text_model.encode([query], convert_to_numpy=True).astype("float32")
    qv = l2_normalize(qv)
    scores, ids = text_index.search(qv, top_k)

    out = []
    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        row = db_get_chunk("text", int(fid))
        if not row:
            continue
        row["score"] = float(score)
        out.append(row)
    return out


def retrieve_images(query: str, top_k: int = 4) -> List[Dict]:
    load_resources_once()
    img_index = _CACHE["img_index"]
    img_model = _CACHE["img_model"]

    qv = img_model.encode([query], convert_to_numpy=True).astype("float32")
    qv = l2_normalize(qv)
    scores, ids = img_index.search(qv, top_k)

    out = []
    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        row = db_get_chunk("image", int(fid))
        if not row:
            continue
        row["score"] = float(score)
        out.append(row)
    return out

def build_prompt(query: str, contexts: list, images: list) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source_file']} | Page {c['page']} | Score {c['score']:.2f}]\n{c['text']}"
        for c in contexts
    )

    image_block = "\n\n".join(
        f"[Image: {i['source_file']} | Page {i['page']} | Score {i['score']:.2f}]\nOCR: {i['ocr_text'] or 'N/A'}"
        for i in images
    )

    return f"""
        Tu es un assistant expert. Tu dois répondre uniquement à partir du contexte fourni.

        QUESTION:
        {query}

        CONTEXTE TEXTUEL:
        {context_block}

        CONTEXTE IMAGES (OCR):
        {image_block}

        INSTRUCTIONS:
        - Réponds de manière professionnelle, claire et structurée
        - Si l'information est absente du contexte, dis-le explicitement
        - Ne fais aucune supposition

        RÉPONSE:
        """

def answer(query: str,top_k_text: int =4, top_k_images: int = 1) -> Dict:
    text_hits = retrieve_text(query, top_k=top_k_text)
    image_hits = retrieve_images(query, top_k=top_k_images)

    return {
        "query": query,
        "contexts": [
            {
                "score": h["score"],
                "source_file": h["source_file"],
                "page": h["page"],
                "text": h["text"],
            }
            for h in text_hits
        ],
        "images": [
            {
                "score": h["score"],
                "source_file": h["source_file"],
                "page": h["page"],
                "image_path": h["image_path"],
                # h["text"] contient l’OCR si présent
                "ocr_text": h["text"],
            }
            for h in image_hits
        ],
        "prompt": None,
        "llm": {
        "answer": None,
        "model": None,
        "latency_ms": None,
        "error": None
        },
        "status": "retrieved_only"
    }

if __name__ == "__main__":
    q = "Comment activer le mode expert ?"
    res = answer(q)
    print(json.dumps(res, indent=2, ensure_ascii=False))
