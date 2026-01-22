import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {p['meta'].get('source','?')}] {p['text']}"
        for p in passages
    )
    return f"""Vous êtes un assistant technique. Répondez uniquement à partir du contexte.
    Si le contexte est insuffisant, dites-le explicitement.

    Contexte:
    {context}

    Question:
    {question}
    """

