import os,re
import html as html_lib

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _where_label(src, page):
    src = src or "Source inconnue"
    if page not in (None, "", 0, "0"):
        return f"{src} — page {page}"
    return f"{src}"

def render_sources_html(query_response: dict) -> str:
    contexts = sorted(
        query_response.get("contexts") or [],
        key=lambda x: x.get("score", 0),
        reverse=True
    )
    out = []

    out.append("<b>Sources</b><br>")

    if not contexts:
        out.append("<div style='margin:8px 0; color:gray;'>Aucun passage pertinent.</div>")
        return "".join(out)

    out.append(f"<div style='margin:6px 0; color:gray;'>{len(contexts)} passage(s) pertinent(s).</div>")
    out.append("<b>Passages pertinents</b><br>")

    for i, c in enumerate(contexts, start=1):
        where = _where_label(c.get("source_file"), c.get("page"))
        score = _safe_float(c.get("score", 0.0))
        text = (c.get("text") or "").strip()
        snippet = text[:900]

        out.append(
            "<div style='margin:10px 0; padding:8px 10px; border-left:3px solid #ddd;'>"
            f"<div style='margin-bottom:4px;'>"
            f"<span style='color:gray;'>#{i}</span> "
            f"<i>{html_lib.escape(where)}</i> "
            f"<span style='color:gray;'>(score {score:.3f})</span>"
            f"</div>"
            f"<div style='line-height:1.35;'>"
            f"{html_lib.escape(snippet).replace(chr(10), '<br>')}"
            f"</div>"
            "</div>"
        )

    return "".join(out)

def render_images_html(query_response: dict) -> str:
    images = sorted(
        query_response.get("images") or [],
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    if not images:
        return ""

    out = []
    out.append("<br><b>Captures / images pertinentes</b><br>")
    out.append(f"<div style='margin:6px 0; color:gray;'>{len(images)} image(s).</div>")

    for i, im in enumerate(images, start=1):
        where = _where_label(im.get("source_file"), im.get("page"))
        score = _safe_float(im.get("score", 0.0))

        img_path = (im.get("image_path") or "").strip()
        abs_path = os.path.abspath(img_path).replace("\\", "/")
        file_url = f"file:///{abs_path}"
        exists = os.path.exists(abs_path)

        ocr = (im.get("ocr_text") or "").strip()
        ocr_snippet = ocr[:600]

        out.append(
            "<div style='margin:12px 0; padding:8px 10px; border-left:3px solid #ddd;'>"
            f"<div style='margin-bottom:6px;'>"
            f"<span style='color:gray;'>#{i}</span> "
            f"<i>{html_lib.escape(where)}</i> "
            f"<span style='color:gray;'>(score {score:.3f})</span>"
            "</div>"
        )

        if exists:
            out.append(
                f"<img src='{html_lib.escape(file_url)}' width='420' "
                "style='border:1px solid #eee; border-radius:6px;'><br>"
            )
        else:
            out.append(
                "<div style='color:#b00020; margin:6px 0;'>Image introuvable.</div>"
            )

        # Chemin (utile pour debug)
        out.append(
            f"<div style='color:gray; font-size:12px; margin-top:6px;'>"
            f"{html_lib.escape(abs_path)}"
            f"</div>"
        )

        # OCR si présent
        if ocr_snippet:
            out.append(
                "<div style='margin-top:8px;'>"
                "<b>OCR</b><br>"
                f"<div style='line-height:1.35;'>"
                f"{html_lib.escape(ocr_snippet).replace(chr(10), '<br>')}"
                f"</div>"
                "</div>"
            )

        out.append("</div>")

    return "".join(out)
def is_api_error(text: str) -> bool:
    return "Error code" in text or "'error':" in text or '"error":' in text

def extract_message(raw_error):
    match = re.search(r"'message':\s*'([^']+)'", raw_error)
    return match.group(1) if match else "Erreur inconnue"

def format_final_answer_html(query_response: dict, final_answer: str) -> str:
    
    try:
        # 6) HTML propre
        out = []
        if is_api_error(final_answer):
            clean_message = extract_message(final_answer)
            out.append(
                "<span style='margin:8px 0; line-height:1.35;color:red;'>"
                f"{html_lib.escape(clean_message)}"
                "</span><br>"
            )
        else:
            clean_message = final_answer
            out.append(
                "<span style='margin:8px 0; line-height:1.35;'>"
                f"{html_lib.escape(clean_message).replace(chr(10), '<br>')}"
                "</span><br>"
            )
            out.append(render_sources_html(query_response))
            out.append(render_images_html(query_response))
            
        bot_reply_html = "".join(out)        
        return bot_reply_html

    except Exception as e:
        bot_reply_html = f"<span style='color: red;'>Erreur : {str(e)}</span>"
        return bot_reply_html