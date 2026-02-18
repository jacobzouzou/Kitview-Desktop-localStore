import sys, os, re, shutil, threading
from pathlib import Path

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QMovie, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout,QLineEdit, QPushButton, QLabel, QHBoxLayout, QFileDialog, QTextBrowser
import torch

# get  api key from .env file
from Helpers.azure import download_knowledge_files_from_azure
from rag_local.final_answer_formatter import format_final_answer_html

from typing import Optional
from rag_local import index_helper
from sentence_transformers import CrossEncoder
from typing import Optional
from dotenv import load_dotenv


def _app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _bundle_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _app_base_dir())
    return _app_base_dir()


def _first_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


APP_DIR = _app_base_dir()
BUNDLE_DIR = _bundle_base_dir()
ASSETS_DIR = _first_existing_path(
    os.path.join(APP_DIR, "assets"),
    os.path.join(BUNDLE_DIR, "assets"),
)
KNOWLEDGE_BASE_DIR = _first_existing_path(
    os.path.join(APP_DIR, "knowledge_base"),
    os.path.join(APP_DIR, "Knowledge_base"),
    os.path.join(BUNDLE_DIR, "knowledge_base"),
    os.path.join(BUNDLE_DIR, "Knowledge_base"),
)
KB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "raw")
os.makedirs(KB_DIR, exist_ok=True)
ENV_PATH = os.path.join(APP_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

APPLICATION_NAME = "kitview"
GREETING_MESSAGE = "Bonjour, je suis votre assistant Kitview. En quoi puis-je vous aider?"
BOT_AVATAR = f"<img src='{Path(os.path.join(ASSETS_DIR, 'orqual_bot.jpeg')).as_uri()}' width='30' height='30' style='background-color:transparent;'/>"
KNOWLEDGE_FILES = []
_RERANKER = None
LLM_MODEL = "phi3:mini"  # "llama3.1:8b", "qwen2.5", "gpt-4o-mini"

_RERANKER: Optional[CrossEncoder] = None
_LOCK = threading.Lock()

def get_reranker() -> CrossEncoder:
    global _RERANKER

    with _LOCK:
        if _RERANKER is None:
            model_name = os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

            try:
                _RERANKER = CrossEncoder(
                    model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load reranker model '{model_name}'"
                ) from e

        return _RERANKER

def normalize_path(path):
    if sys.platform == "win32":
        path = os.path.normpath(path)
        if len(path) > 260:
            return r"\\?\{}".format(path)  # Allows long paths in Windows
    return path

def save_kb_files(files: list, outputs_dir: str ) -> None:
    # Save user files to outputs_dir if file not in outputs_dir already
    new_files = [] 
    for file in files:
        filename = os.path.basename(file)
        dest_path = normalize_path(os.path.join(outputs_dir, filename))
        if not os.path.exists(dest_path):
            shutil.copy2(file, dest_path)
        new_files.append(dest_path)   
         
    KNOWLEDGE_FILES.extend(new_files)    
    return new_files
def select_images_diverse(images, max_n=2):
    seen = set()
    out = []

    for im in sorted(images, key=lambda x: x.get("score", 0), reverse=True):
        key = (im.get("source_file"), im.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(im)
        if len(out) >= max_n:
            break

    return out
class InsgestNewFilesThread(QThread):
    processing_done = pyqtSignal(list)
    failed = pyqtSignal(str)
    
    def __init__(self, files: list):
        super().__init__()
        self.files = files

    def run(self):
        try:
            file_names = save_kb_files(self.files, outputs_dir=KB_DIR)
            index_helper.ingest(KB_DIR)
            self.processing_done.emit(file_names)
        except Exception as e:
            self.failed.emit(str(e)) 
            
class IngestKbFilesThread(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def run(self):
        try:
            index_helper.ingest(KB_DIR)
            self.done.emit("Ingestion completed successfully.")
        except Exception as e:
            self.failed.emit(str(e)) 
            
class ChatbotApp(QWidget):
    def __init__(self, application_name):
        super().__init__()
        self.application_name = application_name    
        
        self.setWindowTitle(f"Assistant IA - {application_name[:1].upper()}{application_name[1:]}")
        self.setGeometry(150, 150, 600, 800)
        self.setWindowIcon(self.get_icon(application_name))

        self.layout = QVBoxLayout()

        self.chat_display = QTextBrowser(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setHtml(f"{BOT_AVATAR}<p style='font-size:15px;'>{GREETING_MESSAGE}</p>")
        self.layout.addWidget(self.chat_display)

        self.loading_label = QLabel(self)
        self.spinner = QMovie(os.path.join(ASSETS_DIR, "typing.gif"))
        self.spinner.setScaledSize(QSize(75, 50))
        self.spinner.backgroundColor = Qt.transparent
        self.loading_label.setMovie(self.spinner)   
        # self.loading_label.setAlignment(Qt.AlignRight)
        self.loading_label.hide()
        self.layout.addWidget(self.loading_label)

        input_layout = QHBoxLayout()
        
        self.button_layout = QHBoxLayout()
        self.select_folder_button = QPushButton(self)
        dir_icon = QPixmap(os.path.join(ASSETS_DIR, "dir_icon.png"))  # Image du bouton
        self.select_folder_button.setIcon(QIcon(dir_icon))
        self.select_folder_button.setIconSize(QSize(30, 30))  # Ajuste la taille de l'icône
        self.select_folder_button.setFixedSize(52, 52)  # Ajuste la taille du bouton
        self.select_folder_button.clicked.connect(self.select_folder)
        input_layout.addWidget(self.select_folder_button)
        
        self.input_text = QLineEdit(self)
        self.input_text.setFixedHeight(49)  # Définit une hauteur fixe plus grande
        self.input_text.setPlaceholderText("Écrivez votre message...")
        self.input_text.installEventFilter(self)  # Permet la détection des touches UP/DOWN
        self.input_text.setStyleSheet("font-size: 15px;")
        self.input_text.returnPressed.connect(lambda: self.send_message(application_name))
        input_layout.addWidget(self.input_text)
        self.input_text.setFocus()
        
        self.button_layout = QHBoxLayout()
        self.clear_button = QPushButton(self)
        clear_icon = QPixmap(os.path.join(ASSETS_DIR, "reset.png"))  # Image du bouton
        self.clear_button.setIcon(QIcon(clear_icon))
        self.clear_button.setIconSize(QSize(30,30))  # Ajuste la taille de l'icône
        self.clear_button.setFixedSize(52, 52)  # Ajuste la taille du bouton
        self.clear_button.clicked.connect(self.clear_conversation)
        input_layout.addWidget(self.clear_button)

        self.button_layout = QHBoxLayout()
        self.send_button = QPushButton(self)
        send_icon = QPixmap(os.path.join(ASSETS_DIR, "send.png"))  # Image du bouton
        self.send_button.setIcon(QIcon(send_icon))
        self.send_button.setIconSize(QSize(30, 30))  # Ajuste la taille de l'icône
        self.send_button.setFixedSize(52, 52)  # Ajuste la taille du bouton        
        self.send_button.clicked.connect(lambda: self.send_message(application_name))
        input_layout.addWidget(self.send_button)

        
        self.layout.addLayout(input_layout)

        self.layout.addLayout(self.button_layout)

        self.history = []
        self.history_index = -1

        self.selected_directory = ""  # Store the selected directory

        self.setLayout(self.layout)

    def select_folder(self):
        # Empêche le lancement multiple
        if hasattr(self, "file_thread") and self.file_thread.isRunning():
            self.chat_display.append("<p style='color: orange;'>⏳ Un traitement est déjà en cours.</p>")
            return

        files, _ = QFileDialog.getOpenFileNames(self,"Sélectionner un ou plusieurs fichiers",os.getcwd(),
            "Tous les fichiers (*);;PDF (*.pdf);;Images (*.png *.jpg *.jpeg)"
        )

        if not files:
            return

        self.selected_files = files       
        
        # Thread avec liste de fichiers
        self.file_thread = InsgestNewFilesThread(self.selected_files)
        self.file_thread.processing_done.connect(self.on_ingest_new_files_done)
        self.file_thread.finished.connect(self.file_thread.deleteLater)
        self.file_thread.start()

    def on_ingest_new_files_done(self, file_ids):
        # UI reset
        self.input_text.setPlaceholderText("Écrivez votre message...")
        self.spinner.stop()
        self.input_text.setFocus()

        if not file_ids:
            self.send_to_bot("Aucun nouveau fichier n’a été indexé.")
            return

        # Format propre pour le bot
        file_list = "\n".join(f"• {fid}" for fid in file_ids)

        bot_message = (
            "📌 **Ingestion terminée avec succès**\n\n"
            "Les fichiers suivants ont été ajoutés à la base de connaissance :\n\n"
            f"{file_list}"
        )

        self.send_to_bot(bot_message)

            
    def eventFilter(self, obj, event):
        if obj == self.input_text and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Up:
                if self.history and self.history_index > 0:
                    self.history_index -= 1
                    self.input_text.setText(self.history[self.history_index])
                return True
            elif event.key() == Qt.Key_Down:
                if self.history and self.history_index < len(self.history) - 1:
                    self.history_index += 1
                    self.input_text.setText(self.history[self.history_index])
                else:
                    self.history_index = len(self.history)
                    self.input_text.clear()
                return True
        return super().eventFilter(obj, event)

    def send_message(self, application_name):
        self.application_name = application_name
        user_message = self.input_text.text().strip()
        
        if user_message:
            self.chat_display.append(f"<p style='font-size: 20px;'>👤</p> <p style='font-size: 15px;'> {user_message}</p>")
            self.chat_display.append("")
            self.history.append(user_message)
            self.history_index = len(self.history)
            self.input_text.clear()
            self.loading_label.show()
            self.spinner.start()
            
            self.worker = ChatbotWorker(user_message, self.application_name)
            self.worker.response_ready.connect(self.display_response)
            self.worker.start()
            
    def display_response(self, bot_reply_html):
        self.spinner.stop()
        self.loading_label.hide()
        # S'assurer que QTextBrowser accepte les liens externes
        self.chat_display.setOpenExternalLinks(True)
        # Remplacement des balises <p> par <span> pour éviter les sauts de ligne excessifs
        # bot_reply_html = bot_reply_html.replace("<p", "<span ").replace("</p>", "</span>").replace("30", "20")

        # Ajouter le nouveau contenu à la fin du body
        self.chat_display.moveCursor(QtGui.QTextCursor.End)  # Place le curseur à la fin
        self.chat_display.insertHtml(f"<br>{BOT_AVATAR}<br>{bot_reply_html}<br>")  # Ajoute le nouveau message

    def clear_conversation(self):
        self.chat_display.setHtml(f"{BOT_AVATAR} <p style='font-size:15px;'>{GREETING_MESSAGE}</p>")
        
    def get_icon(self, app_name):
        # Dictionnaire des icônes en fonction du nom de l'application
        icons = {
            "orqual": os.path.join(ASSETS_DIR, "orqual-removebg-preview.png"),
            "orthalis": os.path.join(ASSETS_DIR, "Orthalis-new.png"),
            "dentalis": os.path.join(ASSETS_DIR, "Dentalis.png"),
            "dentapoche": os.path.join(ASSETS_DIR, "Dentapoche.png"),
            "kitview": os.path.join(ASSETS_DIR, "KitView.png"),
            "ceph": os.path.join(ASSETS_DIR, "ceph.png"),
        }

        # Récupérer le chemin de l'icône ou une icône par défaut
        icon_path = icons.get(app_name.lower(), os.path.join(ASSETS_DIR, "orqual.png"))

        # Vérifier si le fichier existe avant de le charger
        if not os.path.exists(icon_path):
            print(f"⚠️ L'icône pour '{app_name}' n'existe pas : {icon_path}")
            return QIcon()  # Retourne une icône vide si le fichier n'existe pas
        # Charger l'icône avec une haute résolution
        pixmap = QPixmap(icon_path)
        
        # Vérifier la taille de l'icône et la redimensionner si nécessaire
        if pixmap.width() < 128 or pixmap.height() < 128:  # Taille minimale recommandée
            pixmap = pixmap.scaled(256, 256)  # Redimensionner en 256x256 pour améliorer la qualité
        
        return QIcon(pixmap)
    
    def start_ingest(self):
        self.loading_label.show()
        self.spinner.start()
        self.input_text.setDisabled(True)

        self.ingest_thread = InsgestNewFilesThread()
        self.ingest_thread.done.connect(self.on_ingest_done)
        self.ingest_thread.failed.connect(self.on_ingest_failed)
        self.ingest_thread.start()

    def on_ingest_done(self, msg):
        self.spinner.stop()
        self.loading_label.hide()
        self.input_text.setDisabled(False)
        self.chat_display.append(f"<p style='color: green;'>✅ {msg}</p>")

    def on_ingest_failed(self, err):
        self.spinner.stop()
        self.loading_label.hide()
        self.input_text.setDisabled(False)
        self.chat_display.append(f"<p style='color: red;'>❌ {err}</p>")

class ChatbotWorker(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, user_input, application_name):
        super().__init__()
        self.user_input = user_input
        self.application_name = application_name

    def run(self):
        def normalize_text(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"\s+", " ", s)
            return s

        def build_context(res_, max_length=6000):
            parts = []
            total = 0

            for c in res_.get("contexts", []):
                source = c.get("source_file") or "Unknown source"
                page = c.get("page")
                page_info = f"page {page}" if page is not None else "page ?"

                text = (c.get("text") or "").strip()
                if not text:
                    continue

                block = f"[{source} ({page_info})] {text}\n\n"
                if total + len(block) > max_length:
                    break

                parts.append(block)
                total += len(block)

            return "".join(parts).rstrip()

        def select_contexts(query_response, top_n=3, text_score_min=0.25, rerank_min=0.0):
            contexts = query_response.get("contexts", [])

            # 1) Filtre de base: garder les chunks ayant du contenu
            contexts = [c for c in contexts if (c.get("text") or "").strip()]

            # 2) Filtre score initial (si pas de rerank, c'est votre garde-fou)
            #    Si rerank existe, on peut être un peu moins strict sur le score initial.
            has_rerank = any("rerank" in c for c in contexts)
            if not has_rerank:
                contexts = [c for c in contexts if c.get("score", 0.0) >= text_score_min]
            else:
                # Garder un minimum de qualité initiale, mais ne pas tuer le rerank
                contexts = [c for c in contexts if c.get("score", 0.0) >= (text_score_min * 0.5)]

            # 3) Trier: rerank d'abord si dispo, sinon score
            def sort_key(c):
                if "rerank" in c:
                    return float(c.get("rerank", 0.0))
                return float(c.get("score", 0.0))

            contexts.sort(key=sort_key, reverse=True)

            # 4) Filtre rerank optionnel
            if has_rerank and rerank_min > 0.0:
                contexts = [c for c in contexts if float(c.get("rerank", 0.0)) >= rerank_min]

            # 5) Dédoublonnage par texte normalisé (moins destructeur que (source,page))
            seen = set()
            dedup = []
            for c in contexts:
                key = normalize_text(c.get("text", ""))[:400]  # clé stable, légère
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(c)

            return dedup[:top_n]

        def select_images(query_response, top_n=2, img_score_min=0.20):
            images = query_response.get("images", []) or []

            # 1) Filtrer
            images = [i for i in images if float(i.get("score", 0.0)) >= float(img_score_min)]

            # 2) Trier (score FAISS, car pas de rerank image)
            images.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

            # 3) Diversité / dédoublonnage (évite toujours la même page)
            seen = set()
            out = []
            for im in images:
                key = (im.get("source_file"), im.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(im)
                if len(out) >= top_n:
                    break

            return out



        # --- Dans votre pipeline principal ---
        _SMALL_TALK = re.compile(r"^\s*(hello|hi|hey|bonjour|bonsoir|salut|coucou|yo|cc)\s*[.!?]*\s*$", re.I)

        def is_small_talk(q: str) -> bool:
            return bool(_SMALL_TALK.match(q or ""))

        q = (self.user_input or "").strip()

        if is_small_talk(q):
            final_answer = "Bonjour. En quoi puis-je vous aider sur Kitview ?"
            query_response = {"contexts": [], "images": []}
            bot_reply_html = format_final_answer_html(query_response, final_answer)
            self.response_ready.emit(bot_reply_html)
            return
        
        from rag_local.query_helper import answer
        query_response = answer(self.user_input, top_k_text=4, top_k_images=4)

        # Rerank
        try:
            reranker = get_reranker()
            pairs = [(self.user_input, (context.get("text", "") or "")[:1200]) for context in query_response.get("contexts", [])]
            if pairs:
                scores = reranker.predict(pairs)
                for c, s in zip(query_response.get("contexts", []), scores):
                    c["rerank"] = float(s)
        except Exception as e:
            print("Rerank skipped:", e)

        # Sélection finale
        query_response["contexts"] = select_contexts(query_response, top_n=3, text_score_min=0.25, rerank_min=0.0)
        query_response["images"] = select_images(query_response, top_n=2, img_score_min=0.20)

        context = build_context(query_response, max_length=6000)
        
        # Buisiness prompt
        prompt = (
            "Vous êtes un assistant produit Kitview.\n"
            "Répondez précisément à la question en français, en vous basant UNIQUEMENT sur le contexte.\n"
            "Si l'information n'est pas dans le contexte, dites-le clairement.\n"
            f"Question: {self.user_input}\n"
            f"Contexte:\n{context}\n"
            "Format de réponse:\n"
            "- Titre court\n"
            "- 5 à 10 puces maximum\n"
            "- Chaque puce: action/fonction + courte explication\n"
        )

        try:
            if not query_response.get("contexts") and not query_response.get("images"):
                final_answer = "Je n’ai pas trouvé d’information pertinente dans la base de connaissance pour répondre."
                bot_reply_html = format_final_answer_html(query_response, final_answer)
                self.response_ready.emit(bot_reply_html)
                return

            # from rag_local.local_llm_client import call_ollama_llm_generate
            # final_answer = call_ollama_llm_generate(prompt, model=LLM_MODEL, temperature=0.2)
            from rag_local.local_llm_client import call_ollama_llm_chat
            final_answer = call_ollama_llm_chat(prompt, context, model=LLM_MODEL)

            
        except Exception as e:
            final_answer = f"(Synthèse indisponible) {e}"

        bot_reply_html = format_final_answer_html(query_response, final_answer)
        self.response_ready.emit(bot_reply_html)      
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    args = sys.argv[1:]  # Exclude the script name
    
    # download knowledge files from google drive or azure at startup (optionnel)
    if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        try:
            download_knowledge_files_from_azure(dest_dir=KB_DIR)
            print("✅ Fichiers Azure téléchargés avec succès")
        except Exception as e:
            print(f"⚠️ Erreur lors du téléchargement Azure: {e}")
    else:
        print("ℹ️ Azure Storage non configuré, utilisation de la base locale uniquement")
    
    # build or load the index at startup
    index_helper.ingest()
    window = ChatbotApp(args[0].lower() if args else "kitview") 
    window.show()
    sys.exit(app.exec_())
