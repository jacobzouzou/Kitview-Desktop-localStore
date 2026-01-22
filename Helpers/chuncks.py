def chunk_docs(docs, chunk_size=900, overlap=150):
    def chunk_text(t):
        t = " ".join((t or "").split())
        out, i = [], 0
        while i < len(t):
            out.append(t[i:i+chunk_size])
            i += max(1, chunk_size - overlap)
        return out

    chunked = []
    for d in docs:
        for ch in chunk_text(d["text"]):
            chunked.append({"text": ch, "meta": d["meta"]})
    return chunked
