import os, io, re, json, time, uuid, hashlib
import sqlite3
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import fitz  # pymupdf
from PIL import Image
import pytesseract

from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from collections.abc import Iterable


# -----------------------------
# Configuration
# -----------------------------
KB_DIR = "./knowledge_base"
RAW_DIR = os.path.join(KB_DIR, "raw")
CACHE_DIR = os.path.join(KB_DIR, "cache")

DB_PATH = os.path.join(KB_DIR, "store.sqlite")
TEXT_INDEX_PATH = os.path.join(KB_DIR, "faiss_text.index")
IMAGE_INDEX_PATH = os.path.join(KB_DIR, "faiss_image.index")

TEXT_CHUNKS_JSONL = os.path.join(KB_DIR, "text_chunks.jsonl")
IMAGES_JSONL = os.path.join(KB_DIR, "images.jsonl")
KB_META_JSON = os.path.join(KB_DIR, "kb_meta.json")

# Modèles (local)
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


# OCR: si besoin, décommentez et mettez votre chemin local
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

SUPPORTED_EXT = {".pdf", ".docx", ".csv", ".txt"}


# -----------------------------
# Utilitaires
# -----------------------------
def ensure_dirs():
    os.makedirs(KB_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def now_ts() -> int:
    return int(time.time())


# -----------------------------
# SQLite
# -----------------------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def db_init():
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_hash TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        filename TEXT NOT NULL,
        ext TEXT NOT NULL,
        added_at INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,                 -- 'text' ou 'image'
        doc_hash TEXT NOT NULL,
        source_file TEXT NOT NULL,
        page INTEGER,                       -- page PDF si applicable
        text TEXT,                          -- chunk texte ou OCR image
        image_path TEXT,                    -- chemin vers image extraite
        faiss_id INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (doc_hash) REFERENCES documents(doc_hash)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash ON chunks(doc_hash);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(kind, faiss_id);")

    conn.commit()
    conn.close()


def db_document_exists(doc_hash: str) -> bool:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM documents WHERE doc_hash = ? LIMIT 1;", (doc_hash,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def db_insert_document(doc_hash: str, path: str, filename: str, ext: str):
    conn = db_connect()
    conn.execute(
        "INSERT INTO documents (doc_hash, path, filename, ext, added_at) VALUES (?, ?, ?, ?, ?);",
        (doc_hash, path, filename, ext, now_ts()),
    )
    conn.commit()
    conn.close()


def db_insert_chunks(rows: List[Tuple]):
    """
    rows: (kind, doc_hash, source_file, page, text, image_path, faiss_id, created_at)
    """
    conn = db_connect()
    conn.executemany("""
    INSERT INTO chunks (kind, doc_hash, source_file, page, text, image_path, faiss_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, rows)
    conn.commit()
    conn.close()


def db_get_chunk_by_faiss_id(kind: str, faiss_id: int) -> Optional[Dict]:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
    SELECT kind, doc_hash, source_file, page, text, image_path, faiss_id
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
        "doc_hash": row[1],
        "source_file": row[2],
        "page": row[3],
        "text": row[4],
        "image_path": row[5],
        "faiss_id": row[6],
    }

# -----------------------------
# FAISS
# -----------------------------
def load_or_create_index(path: str, dim):
    # Validation dimension
    if dim is None:
        raise ValueError(f"dim is None for index: {path}")
    try:
        dim_int = int(dim)
    except Exception:
        raise ValueError(f"dim must be int-like, got {type(dim)}={dim} for index: {path}")

    if dim_int <= 0:
        raise ValueError(f"dim must be > 0, got {dim_int} for index: {path}")

    # Lecture si possible
    if os.path.exists(path):
        try:
            if os.path.getsize(path) == 0:
                os.remove(path)
            else:
                idx = faiss.read_index(path)
                # Optionnel : vérifier cohérence de dimension
                if hasattr(idx, "d") and idx.d != dim_int:
                    corrupt_name = f"{path}.dim_mismatch_{idx.d}_vs_{dim_int}_{int(time.time())}"
                    os.replace(path, corrupt_name)
                else:
                    return idx
        except Exception:
            corrupt_name = f"{path}.corrupt_{int(time.time())}"
            try:
                os.replace(path, corrupt_name)
            except Exception:
                try: os.remove(path)
                except Exception: pass

    # Création index
    index = faiss.IndexFlatIP(dim_int)
    faiss.write_index(index, path)
    return index

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    # v: (n, d) float32
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / norms).astype("float32")

# -----------------------------
# Extraction: PDF
# -----------------------------
def pdf_extract_text_and_images(pdf_path: str, cache_subdir: str) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Retourne :
      - liste (page_index_1based, texte_page)
      - liste (page_index_1based, chemin_image_extraite)
    """
    doc = fitz.open(pdf_path)
    texts = []
    images = []

    os.makedirs(cache_subdir, exist_ok=True)

    for i in range(len(doc)):
        page = doc[i]
        page_no = i + 1

        # Texte natif
        text = page.get_text("text") or ""
        text = text.strip()
        texts.append((page_no, text))

        # Images intégrées
        img_list = page.get_images(full=True)
        if img_list:
            for img_idx, img in enumerate(img_list):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base.get("image")
                ext = base.get("ext", "png")
                out_path = os.path.join(cache_subdir, f"pdf_p{page_no}_img{img_idx}.{ext}")
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                images.append((page_no, out_path))

    doc.close()
    return texts, images


def ocr_pdf_if_needed(pdf_path: str, cache_subdir: str, texts_by_page: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    OCR fallback: si une page a très peu de texte, on rasterise la page et OCR.
    """
    doc = fitz.open(pdf_path)
    os.makedirs(cache_subdir, exist_ok=True)

    out = []
    for (page_no, text) in texts_by_page:
        # Heuristique : si page quasi vide => OCR
        if len(normalize_spaces(text)) >= 30:
            out.append((page_no, text))
            continue

        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=200)  # compromis qualité/perf
        img_path = os.path.join(cache_subdir, f"pdf_p{page_no}_ocr.png")
        pix.save(img_path)

        img = Image.open(img_path)
        ocr_text = pytesseract.image_to_string(img) or ""
        ocr_text = ocr_text.strip()

        merged = (text + "\n" + ocr_text).strip() if text else ocr_text
        out.append((page_no, merged))

    doc.close()
    return out


# -----------------------------
# Extraction: DOCX
# -----------------------------
def docx_extract_text_and_images(docx_path: str, cache_subdir: str) -> Tuple[str, List[str]]:
    """
    Retourne texte complet et liste de chemins images extraites.
    """
    os.makedirs(cache_subdir, exist_ok=True)

    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    full_text = "\n".join(paragraphs)

    # Extraction images via relations
    image_paths = []
    rels = doc.part.rels
    idx = 0
    for rel in rels.values():
        if "image" in rel.reltype:
            img_bytes = rel.target_part.blob
            # extension souvent inconnue: on dérive via content_type
            ct = (rel.target_part.content_type or "").lower()
            ext = "png"
            if "jpeg" in ct or "jpg" in ct:
                ext = "jpg"
            elif "png" in ct:
                ext = "png"
            elif "gif" in ct:
                ext = "gif"

            out_path = os.path.join(cache_subdir, f"docx_img{idx}.{ext}")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            image_paths.append(out_path)
            idx += 1

    return full_text, image_paths


# -----------------------------
# Extraction: CSV / TXT
# -----------------------------
def csv_to_text_rows(csv_path: str, max_rows: Optional[int] = None) -> List[str]:
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)
    rows = []
    cols = list(df.columns)
    for _, r in df.iterrows():
        parts = [f"{c}: {r[c]}" for c in cols]
        rows.append(" | ".join(parts))
    return rows


def txt_read(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -----------------------------
# Embeddings
# -----------------------------
def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    v = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    v = v.astype("float32")
    return l2_normalize(v)

def embed_images(clip_model, img_paths):
    # img_paths doit être itérable
    if isinstance(img_paths, (str, bytes)):
        img_paths = [img_paths]

    images = []
    for x in img_paths:
        if isinstance(x, Image.Image):
            img = x.convert("RGB")
        else:
            img = Image.open(x).convert("RGB")

        images.append(np.array(img, dtype=np.uint8))  # <-- point clé

    return clip_model.encode(images, convert_to_numpy=True, show_progress_bar=False)
# -----------------------------
# Écriture JSONL (append)
# -----------------------------
def append_jsonl(path: str, records: List[Dict]):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Ingestion principale
# -----------------------------
def ingest():
    ensure_dirs()
    db_init()

    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    clip_model = SentenceTransformer(CLIP_MODEL_NAME)

    text_dim = text_model.get_sentence_embedding_dimension()
    img_dim = clip_model.get_sentence_embedding_dimension()
    print("CLIP dim:", clip_model.get_sentence_embedding_dimension())
    print("CLIP device:", clip_model.device)
    print("CLIP modules:", clip_model._modules.keys())
    text_index = load_or_create_index(TEXT_INDEX_PATH, text_dim)
    img_index = load_or_create_index(IMAGE_INDEX_PATH, img_dim)

    # scan raw
    files = []
    for name in os.listdir(RAW_DIR):
        p = os.path.join(RAW_DIR, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXT:
            files.append((p, name, ext))

    if not files:
        print("Aucun fichier trouvé dans knowledge_base/raw/")
        return

    new_docs = 0
    new_text_chunks = 0
    new_images = 0

    for path, filename, ext in files:
        doc_hash = sha256_file(path)
        if db_document_exists(doc_hash):
            continue  # déjà indexé (incrémental)

        cache_subdir = os.path.join(CACHE_DIR, doc_hash)
        os.makedirs(cache_subdir, exist_ok=True)

        # ------------------ Extraction ------------------
        text_records = []  # for jsonl
        image_records = []  # for jsonl

        chunks_meta_rows = []  # for sqlite insert
        images_meta_rows = []  # for sqlite insert

        if ext == ".pdf":
            texts_by_page, extracted_images = pdf_extract_text_and_images(path, cache_subdir)
            # OCR fallback pages
            texts_by_page = ocr_pdf_if_needed(path, cache_subdir, texts_by_page)

            # chunks texte page par page
            page_chunks = []
            page_chunk_meta = []
            for page_no, page_text in texts_by_page:
                for ch in chunk_text(page_text):
                    page_chunks.append(ch)
                    page_chunk_meta.append((page_no, ch))

            # embeddings texte
            tv = embed_texts(text_model, page_chunks)
            start_id = int(text_index.ntotal)
            if tv.shape[0] > 0:
                text_index.add(tv)

            for i, (page_no, ch) in enumerate(page_chunk_meta):
                faiss_id = start_id + i
                rec = {
                    "kind": "text",
                    "doc_hash": doc_hash,
                    "source_file": filename,
                    "page": page_no,
                    "text": ch,
                    "faiss_id": faiss_id,
                }
                text_records.append(rec)
                chunks_meta_rows.append((
                    "text", doc_hash, filename, page_no, ch, None, faiss_id, now_ts()
                ))

            # images extraites
            img_paths = [p for (_, p) in extracted_images]
            if img_paths:
                iv = embed_images(clip_model, img_paths)
                img_start_id = int(img_index.ntotal)
                img_index.add(iv)

                for j, (page_no, img_path) in enumerate(extracted_images):
                    faiss_id = img_start_id + j

                    # OCR image (option C)
                    try:
                        ocr_txt = pytesseract.image_to_string(Image.open(img_path).convert("RGB")) or ""
                        ocr_txt = normalize_spaces(ocr_txt)
                    except Exception:
                        ocr_txt = ""

                    rec = {
                        "kind": "image",
                        "doc_hash": doc_hash,
                        "source_file": filename,
                        "page": page_no,
                        "image_path": img_path,
                        "ocr_text": ocr_txt,
                        "faiss_id": faiss_id,
                    }
                    image_records.append(rec)
                    images_meta_rows.append((
                        "image", doc_hash, filename, page_no, ocr_txt or None, img_path, faiss_id, now_ts()
                    ))

        elif ext == ".docx":
            full_text, img_paths = docx_extract_text_and_images(path, cache_subdir)

            # chunks texte
            chunks = chunk_text(full_text)
            tv = embed_texts(text_model, chunks)
            start_id = int(text_index.ntotal)
            if tv.shape[0] > 0:
                text_index.add(tv)

            for i, ch in enumerate(chunks):
                faiss_id = start_id + i
                rec = {
                    "kind": "text",
                    "doc_hash": doc_hash,
                    "source_file": filename,
                    "page": None,
                    "text": ch,
                    "faiss_id": faiss_id,
                }
                text_records.append(rec)
                chunks_meta_rows.append((
                    "text", doc_hash, filename, None, ch, None, faiss_id, now_ts()
                ))

            # images + OCR
            if img_paths:
                iv = embed_images(clip_model, img_paths)
                img_start_id = int(img_index.ntotal)
                img_index.add(iv)

                for j, img_path in enumerate(img_paths):
                    faiss_id = img_start_id + j
                    try:
                        ocr_txt = pytesseract.image_to_string(Image.open(img_path).convert("RGB")) or ""
                        ocr_txt = normalize_spaces(ocr_txt)
                    except Exception:
                        ocr_txt = ""

                    rec = {
                        "kind": "image",
                        "doc_hash": doc_hash,
                        "source_file": filename,
                        "page": None,
                        "image_path": img_path,
                        "ocr_text": ocr_txt,
                        "faiss_id": faiss_id,
                    }
                    image_records.append(rec)
                    images_meta_rows.append((
                        "image", doc_hash, filename, None, ocr_txt or None, img_path, faiss_id, now_ts()
                    ))

        elif ext == ".csv":
            rows = csv_to_text_rows(path)
            # on chunk aussi, car des lignes peuvent être longues
            chunks = []
            for r in rows:
                chunks.extend(chunk_text(r, chunk_size=700, overlap=100))

            tv = embed_texts(text_model, chunks)
            start_id = int(text_index.ntotal)
            if tv.shape[0] > 0:
                text_index.add(tv)

            for i, ch in enumerate(chunks):
                faiss_id = start_id + i
                rec = {
                    "kind": "text",
                    "doc_hash": doc_hash,
                    "source_file": filename,
                    "page": None,
                    "text": ch,
                    "faiss_id": faiss_id,
                }
                text_records.append(rec)
                chunks_meta_rows.append((
                    "text", doc_hash, filename, None, ch, None, faiss_id, now_ts()
                ))

        elif ext == ".txt":
            full_text = txt_read(path)
            chunks = chunk_text(full_text)

            tv = embed_texts(text_model, chunks)
            start_id = int(text_index.ntotal)
            if tv.shape[0] > 0:
                text_index.add(tv)

            for i, ch in enumerate(chunks):
                faiss_id = start_id + i
                rec = {
                    "kind": "text",
                    "doc_hash": doc_hash,
                    "source_file": filename,
                    "page": None,
                    "text": ch,
                    "faiss_id": faiss_id,
                }
                text_records.append(rec)
                chunks_meta_rows.append((
                    "text", doc_hash, filename, None, ch, None, faiss_id, now_ts()
                ))

        # ------------------ Persist ------------------
        db_insert_document(doc_hash, path, filename, ext)
        if chunks_meta_rows:
            db_insert_chunks(chunks_meta_rows)
        if images_meta_rows:
            db_insert_chunks(images_meta_rows)

        append_jsonl(TEXT_CHUNKS_JSONL, text_records)
        append_jsonl(IMAGES_JSONL, image_records)

        new_docs += 1
        new_text_chunks += len(text_records)
        new_images += len(image_records)

        print(f"Indexé: {filename} | chunks texte: {len(text_records)} | images: {len(image_records)}")

    # Sauvegarde FAISS
    save_index(text_index, TEXT_INDEX_PATH)
    save_index(img_index, IMAGE_INDEX_PATH)

    # Meta
    meta = {
        "text_embedding_model": TEXT_MODEL_NAME,
        "image_embedding_model": CLIP_MODEL_NAME,
        "text_dim": int(text_index.d),
        "image_dim": int(img_index.d),
        "text_vectors": int(text_index.ntotal),
        "image_vectors": int(img_index.ntotal),
        "updated_at": now_ts(),
    }
    with open(KB_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("----")
    print(f"Nouveaux documents: {new_docs}")
    print(f"Nouveaux chunks texte: {new_text_chunks}")
    print(f"Nouvelles images indexées: {new_images}")
    print("KB prête.")


if __name__ == "__main__":
    ingest()
