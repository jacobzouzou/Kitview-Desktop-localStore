import os
import re
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader

def load_documents(self, docs): # Load documents from external source
        loaders = [
            PyPDFLoader("../Data/botanical.pdf"),
        ]

        docs = []

        for loader in loaders:
            docs.extend(loader.load())
        return docs
def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def ocr_image_pil(img: Image.Image, lang: str = "fra") -> str:
    # Ajustements simples pour OCR sur captures UI
    img = img.convert("L")  # grayscale
    return clean_text(pytesseract.image_to_string(img, lang=lang))

def extract_pdf_with_selective_ocr(
    pdf_path: str,
    out_images_dir: str,
    min_native_text_chars: int = 200,
    min_image_area_ratio: float = 0.02,
    ocr_lang: str = "fra",
    ocr_full_page_fallback: bool = True,
    dpi_full_page: int = 200,
):
    """
    Retourne une liste de documents (chunks bruts) sous forme:
    [{"text": "...", "meta": {...}}, ...]

    - Extrait texte natif page par page
    - Extrait images de la page et fait OCR sélectif
    - Optionnel: OCR du rendu page si page sans texte (scans)
    """
    os.makedirs(out_images_dir, exist_ok=True)

    docs = []
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_no = page_index + 1

        native_text = clean_text(page.get_text("text"))
        page_rect = page.rect
        page_area = float(page_rect.width * page_rect.height)

        # Collecter images de la page
        images = page.get_images(full=True)

        extracted_image_paths = []
        total_img_area_est = 0.0

        for img_i, img_info in enumerate(images, start=1):
            xref = img_info[0]
            pix = fitz.Pixmap(doc, xref)

            # Convertir en PNG (gérer CMYK/alpha)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_name = f"{base}_p{page_no}_img{img_i}.png"
            img_path = os.path.join(out_images_dir, img_name)
            pix.save(img_path)
            extracted_image_paths.append(img_path)

            # estimation grossière de l'importance (si pixmap grand)
            total_img_area_est += float(pix.width * pix.height)

        # Heuristique : décider OCR
        native_text_is_low = len(native_text) < min_native_text_chars
        images_are_significant = False
        if extracted_image_paths:
            # Estimation relative (pixmap area vs page area; pas parfait mais utile)
            # On ramène l'aire des pixmaps à une ratio approximatif
            # (on considère que de grandes captures d'écran dominent)
            images_are_significant = (total_img_area_est / max(page_area, 1.0)) > min_image_area_ratio

        should_ocr_images = extracted_image_paths and (native_text_is_low or images_are_significant)

        # 1) Ajouter le texte natif (même si vide, pour garder la trace page)
        if native_text:
            docs.append({
                "text": native_text,
                "meta": {
                    "source": pdf_path,
                    "page": page_no,
                    "type": "native_text",
                    "images": extracted_image_paths,
                }
            })

        # 2) OCR des images (captures)
        if should_ocr_images:
            for img_path in extracted_image_paths:
                try:
                    img = Image.open(img_path)
                    ocr_text = ocr_image_pil(img, lang=ocr_lang)
                    if ocr_text:
                        docs.append({
                            "text": ocr_text,
                            "meta": {
                                "source": pdf_path,
                                "page": page_no,
                                "type": "ocr_image",
                                "image": img_path,
                            }
                        })
                except Exception:
                    # Ne jamais bloquer l’ingestion sur une image corrompue
                    pass

        # 3) Fallback OCR page entière (utile pour PDF scannés sans images extractibles)
        if ocr_full_page_fallback and native_text_is_low and not extracted_image_paths:
            try:
                # rendu page -> image
                zoom = dpi_full_page / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                page_img_name = f"{base}_p{page_no}_page.png"
                page_img_path = os.path.join(out_images_dir, page_img_name)
                pix.save(page_img_path)

                img = Image.open(page_img_path)
                ocr_text = ocr_image_pil(img, lang=ocr_lang)
                if ocr_text:
                    docs.append({
                        "text": ocr_text,
                        "meta": {
                            "source": pdf_path,
                            "page": page_no,
                            "type": "ocr_page",
                            "image": page_img_path,
                        }
                    })
            except Exception:
                pass

    doc.close()
    return docs
