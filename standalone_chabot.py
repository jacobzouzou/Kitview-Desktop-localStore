import sys, os, csv,time,shutil,json,stat,docx,markdown
import win32com.client
import html

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QMovie, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout,QLineEdit, QPushButton, QLabel, QHBoxLayout, QFileDialog, QTextBrowser

from PyPDF2 import PdfMerger
from pptx import Presentation
from bs4 import BeautifulSoup 

import openai
import pandas as pd

# get  api key from .env file
from Helpers.google import download_knowledge_files_from_googleDrive
from Helpers.azure import download_knowledge_files_from_azure
from rag_local.final_answer_formatter import format_final_answer_html

import shutil
from typing import Optional, Set
from rag_local import index_helper
from sentence_transformers import CrossEncoder
from threading import Lock
from dotenv import load_dotenv
load_dotenv()

 
# openai.api_key = os.getenv("KITVIEW_DESKTOP_OPENAI_API_KEY")
# OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("KITVIEW_DESKTOP_OPENAI_API_KEY"))
APPLICATION_NAME = "kitview"
GREETING_MESSAGE = "Bonjour, je suis votre assistant Kitview. Comment puis-je vous aider aujourd'hui ?"
BOT_AVATAR = "<img src='./assets/orqual_bot.jpeg' width='30' height='30' style='background-color:transparent;'/>"
KNOWLEDGE_FILES = []
KB_DIR = "./knowledge_base/raw"
_RERANKER = None
_LOCK = Lock()

def get_reranker():
    global _RERANKER
    with _LOCK:
        if _RERANKER is None:
            _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
        self.spinner = QMovie("./assets/typing.gif")
        self.spinner.setScaledSize(QSize(75, 50))
        self.spinner.backgroundColor = Qt.transparent
        self.loading_label.setMovie(self.spinner)   
        # self.loading_label.setAlignment(Qt.AlignRight)
        self.loading_label.hide()
        self.layout.addWidget(self.loading_label)

        input_layout = QHBoxLayout()
        
        self.button_layout = QHBoxLayout()
        self.select_folder_button = QPushButton(self)
        dir_icon = QPixmap("./assets/dir_icon.png")  # Image du bouton
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
        clear_icon = QPixmap("./assets/reset.png")  # Image du bouton
        self.clear_button.setIcon(QIcon(clear_icon))
        self.clear_button.setIconSize(QSize(30,30))  # Ajuste la taille de l'icône
        self.clear_button.setFixedSize(52, 52)  # Ajuste la taille du bouton
        self.clear_button.clicked.connect(self.clear_conversation)
        input_layout.addWidget(self.clear_button)

        self.button_layout = QHBoxLayout()
        self.send_button = QPushButton(self)
        send_icon = QPixmap("./assets/send.png")  # Image du bouton
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
            "orqual": "./assets/orqual-removebg-preview.png",    
            "orthalis": "./assets/Orthalis-new.png",
            "dentalis": "./assets/Dentalis.png",
            "dentapoche": "./assets/Dentapoche.png",
            "kitview": "./assets/KitView.png",
            "ceph":"./assets/ceph.png",
        }

        # Récupérer le chemin de l'icône ou une icône par défaut
        icon_path = icons.get(app_name.lower(), "./assets/orqual.png")

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
        def build_context(res_):
            parts = []
            for c in res_.get("contexts", []):
                where = f'{c.get("source_file")} (page {c.get("page")})'
                parts.append(f"[{where}] {c.get('text','')}")
            return "\n\n".join(parts)[:6000]
        
        try:
            import html as html_lib
            from rag_local.query_helper import answer

            # 1) Retrieve large enough, then rerank/filter
            query_response = answer(self.user_input, top_k_text=12, top_k_images=4)

            # 2) Rerank (AVANT filtrage final)
            try:
                reranker = get_reranker()
                pairs = [(self.user_input, c.get("text", "")[:1200]) for c in query_response.get("contexts", [])]
                if pairs:
                    scores = reranker.predict(pairs)
                    for c, s in zip(query_response["contexts"], scores):
                        c["rerank"] = float(s)

                    query_response["contexts"].sort(key=lambda x: x.get("rerank", 0.0), reverse=True)
            except Exception as e:
                print("Rerank skipped:", e)

            # 3) Filtrage + dédoublonnage + limitation
            TEXT_SCORE_MIN = 0.25
            IMG_SCORE_MIN = 0.20

            texts = [c for c in query_response.get("contexts", []) if c.get("score", 0.0) >= TEXT_SCORE_MIN]
            imgs = [i for i in query_response.get("images", []) if i.get("score", 0.0) >= IMG_SCORE_MIN]

            seen = set()
            dedup_texts = []
            for c in texts:
                key = (c.get("source_file"), c.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                dedup_texts.append(c)

            query_response["contexts"] = dedup_texts[:3]
            query_response["images"] = imgs[:2]

            # 4) Construire le contexte POUR le LLM (après nettoyage)


            context = build_context(query_response)

            prompt = f"""
                Vous êtes un assistant produit Kitview.
                Répondez précisément à la question en français, en vous basant UNIQUEMENT sur le contexte.
                Si l'information n'est pas dans le contexte, dites-le clairement.
                Question: {self.user_input}
                Contexte: {context}
                Format de réponse:
                - Titre court
                - 5 à 10 puces maximum
                - Chaque puce: action/fonction + courte explication
                """.strip()

            # 5) Synthèse LLM
            
            try:
                # from rag_local.openai_client import call_llm
                # final_answer = call_llm(prompt, model="gpt-4.1-mini")
                from rag_local.local_llm_client import call_llm_local
                final_answer = call_llm_local(prompt, model="mistral", temperature=0.2)
            except Exception as e:
                final_answer = f"(Synthèse indisponible) {e}"
                # 6) HTML propre
                
            bot_reply_html = format_final_answer_html(query_response, final_answer)        
            self.response_ready.emit(bot_reply_html)
        except Exception as e:
            error_html = f"<span style='color: red;'>Erreur lors du traitement : {str(e)}</span>"
            self.response_ready.emit(error_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    args = sys.argv[1:]  # Exclude the script name
    # download knowledge files from google drive or azure at startup
    download_knowledge_files_from_azure(dest_dir=KB_DIR)
    # build or load the index at startup
    index_helper.ingest()
    window = ChatbotApp(args[0].lower() if args else "kitview") 
    window.show()
    sys.exit(app.exec_())
