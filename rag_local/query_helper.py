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
    
    _CACHE["text_metric"] = detect_faiss_metric(_CACHE["text_index"])
    _CACHE["img_metric"] = detect_faiss_metric(_CACHE["img_index"])

    _CACHE["text_model"] = SentenceTransformer(meta["text_embedding_model"])
    _CACHE["img_model"] = SentenceTransformer(meta["image_embedding_model"])

    _CACHE["loaded"] = True

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_get_chunks(kind: str, faiss_ids: List[int]) -> Dict[int, Dict]:
    if not faiss_ids:
        return {}

    conn = db_connect()
    cur = conn.cursor()

    placeholders = ",".join("?" for _ in faiss_ids)
    cur.execute(f"""
        SELECT kind, source_file, page, text, image_path, faiss_id
        FROM chunks
        WHERE kind = ? AND faiss_id IN ({placeholders});
    """, [kind, *faiss_ids])

    rows = cur.fetchall()
    conn.close()

    out = {}
    for row in rows:
        out[int(row[5])] = {
            "kind": row[0],
            "source_file": row[1],
            "page": row[2],
            "text": row[3],
            "image_path": row[4],
            "faiss_id": row[5],
        }
    return out



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
    faiss_ids = [int(fid) for fid in ids[0] if fid != -1]

    rows_by_id = db_get_chunks("text", faiss_ids)

    out = []
    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        row = rows_by_id.get(int(fid))
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
    faiss_ids = [int(fid) for fid in ids[0] if fid != -1]

    rows_by_id = db_get_chunks("image", faiss_ids)

    out = []
    for score, fid in zip(scores[0], ids[0]):
        if fid == -1:
            continue
        row = rows_by_id.get(int(fid))
        if not row:
            continue
        row["score"] = float(score)
        out.append(row)
    return out

def detect_faiss_metric(index) -> str:
    # FAISS expose index.metric_type pour les IndexFlat / certains index
    # Sinon, on fait une heuristique minimale
    mt = getattr(index, "metric_type", None)
    if mt is None:
        return "unknown"
    # 0: METRIC_INNER_PRODUCT, 1: METRIC_L2 (dans faiss)
    if int(mt) == int(faiss.METRIC_INNER_PRODUCT):
        return "ip"
    if int(mt) == int(faiss.METRIC_L2):
        return "l2"
    return "unknown"

def build_prompt(query: str, contexts: list, images: list) -> str:
    context_block = "\n\n".join(
        f"[Source: {c.get('source_file')} | Page {c.get('page')} | Score {c.get('score', 0.0):.2f}]\n{c.get('text','')}"
        for c in contexts
    ) or "(Aucun contexte textuel)"

    image_block = "\n\n".join(
        f"[Image: {i.get('source_file')} | Page {i.get('page')} | Score {i.get('score', 0.0):.2f}]\nOCR: {(i.get('ocr_text') or 'N/A')}"
        for i in images
    ) or "(Aucune image)"

    return f"""Tu es un assistant expert. Tu dois répondre uniquement à partir du contexte fourni.

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


def answer(query: str, top_k_text: int = 4, top_k_images: int = 1) -> Dict:
    text_hits = retrieve_text(query, top_k=top_k_text)
    image_hits = retrieve_images(query, top_k=top_k_images)

    return {
        "query": query,
        "score_type": {
            "text": _CACHE.get("text_metric", "unknown"),
            "image": _CACHE.get("img_metric", "unknown"),
        },
        "contexts": [
            {
                "faiss_id": h.get("faiss_id"),
                "score": h["score"],
                "source_file": h["source_file"],
                "page": h["page"],
                "text": h["text"],
            }
            for h in text_hits
        ],
        "images": [
            {
                "faiss_id": h.get("faiss_id"),
                "score": h["score"],
                "source_file": h["source_file"],
                "page": h["page"],
                "image_path": h["image_path"],
                "ocr_text": h["text"],
            }
            for h in image_hits
        ],
        "prompt": None,
        "llm": {"answer": None, "model": None, "latency_ms": None, "error": None},
        "status": "retrieved_only",
    }


if __name__ == "__main__":
    q = "Comment activer le mode expert ?"
    res = answer(q)
    print(json.dumps(res, indent=2, ensure_ascii=False))
