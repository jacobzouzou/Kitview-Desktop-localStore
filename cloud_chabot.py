from pydoc import html
import sys, os, csv,time,shutil,json,stat,docx,markdown
import re
import win32com.client

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

from dotenv import load_dotenv
load_dotenv()

KITVIEW_DESKTOP_OPENAI_API_KEY = os.getenv("KITVIEW_DESKTOP_OPENAI_API_KEY")
KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID = os.getenv("KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID")
openai.api_key = KITVIEW_DESKTOP_OPENAI_API_KEY 
from openai import OpenAI
OPENAI_CLIENT = OpenAI(api_key=KITVIEW_DESKTOP_OPENAI_API_KEY)
ASSISTANT_IDS_DIC = json.loads(os.getenv("ASSISTANT_IDS"))
KB_DIRECTORY = "./knowledge_base/raw"
APPLICATION_NAME = "kitview"

# Récupérer une valeur spécifique
GREETING_MESSAGE = "Bonjour, je suis Kity votre assistant. Comment puis-je vous aider aujourd'hui ?"
BOT_AVATAR = "<img src='./assets/chatbot.png' width='30' height='30'>"

def normalize_path(path):
    if sys.platform == "win32":
        path = os.path.normpath(path)
        if len(path) > 260:
            return r"\\?\{}".format(path)  # Allows long paths in Windows
    return path


def openaiUploadFiles(files):
    """Uploads multiple files to OpenAI and returns their file IDs."""
    file_ids = []
    
    for files in files:
        try:
            file_obj = openai.files.create(
                file=open(files, "rb"),
                purpose="assistants"
            )
            file_ids.append(file_obj.id)
            print(f"Uploaded: {files} -> File ID: {file_obj.id}")
        except Exception as e:
            print(f"Failed to upload {files}: {e}")
    
    return file_ids

import os

def get_kb_files(directory):
    """Gets all supported document file paths from the specified directory."""
    
    SUPPORTED_EXTENSIONS = (
        ".pdf",
        ".doc", ".docx",
        ".xls", ".xlsx",
        ".ppt", ".pptx",
        ".csv",
        ".json"
    )

    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        return []

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not files:
        print("No supported documents found in the directory.")
    else:
        print(f"Found {len(files)} supported documents in {directory}.")

    return files

def wait_for_files_to_be_ready(file_ids):
    """Waits until all uploaded files are processed before attaching them."""
    print("Waiting for files to be processed...")

    for file_id in file_ids:
        while True:
            try:
                file_status = openai.files.retrieve(file_id)
                if file_status.status == "processed":
                    print(f"File {file_id} is ready.")
                    break  # Move to next file
                else:
                    print(f"File {file_id} is still processing...")
                    time.sleep(2)  # Wait before checking again
            except Exception as e:
                print(f"Error checking file {file_id} status: {e}")
                break  # Stop checking on error

def create_thread_with_files(file_ids, user_question):  
    """Creates a thread and attaches multiple files to the message."""
    try:
        vs = OPENAI_CLIENT.beta.vector_stores.retrieve(name=KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID)

        # 3) Add files to the vector store (batch)
        OPENAI_CLIENT.beta.vector_stores.file_batches.create(
            vector_store_id=vs.id,
            file_ids=file_ids
        )

        # 4) Create thread with tool_resources pointing to vector store
        thread = OPENAI_CLIENT.beta.threads.create(
            tool_resources={
                "file_search": {"vector_store_ids": [vs.id]}
            },
            messages=[{"role": "user", "content": user_question}]
        )

        # 5) Run
        run = OPENAI_CLIENT.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_IDS_DIC[APPLICATION_NAME]
        )

        print(f"Thread created: {thread.id}")
        return thread
    except Exception as e:
        print(f"Error creating thread: {e}")
        return None
    
def load_Kb_files(selected_folder):
    files_ids=openaiUploadFiles(files=get_kb_files(selected_folder))        
    if files_ids and len(files_ids) > 0:
        wait_for_files_to_be_ready(files_ids)
    return files_ids 

def clean_citations(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"【[^】]*】", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def render_light_md_to_html(text: str) -> str:
    """
    Rend un sous-ensemble Markdown (## titres, listes -/*, paragraphes) en HTML.
    Safe: on échappe d'abord, puis on structure.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out = []
    in_ul = False
    in_p = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def close_p():
        nonlocal in_p
        if in_p:
            out.append("</p>")
            in_p = False

    for raw in lines:
        line = raw.rstrip()

        # Ligne vide => fermeture des blocs en cours
        if not line.strip():
            close_ul()
            close_p()
            continue

        # Titres "## "
        if line.lstrip().startswith("## "):
            close_ul()
            close_p()
            title = line.lstrip()[3:].strip()
            out.append(f"<h2 style='margin:10px 0 6px;'>{title}</h2>")
            continue

        # Puces "- " ou "* "
        stripped = line.lstrip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            close_p()
            if not in_ul:
                out.append("<ul style='margin:6px 0 10px 18px;'>")  
                in_ul = True
            item = stripped[2:].strip()
            out.append(f"<li style='margin:2px 0;'>{item}</li>")
            continue

        # Texte normal => paragraphe
        close_ul()
        if not in_p:
            out.append("<p style='margin:6px 0; line-height:1.45;'>")
            in_p = True

        # Conserver les retours ligne "soft" dans le paragraphe
        out.append(line + "<br>")

    close_ul()
    close_p()
    return "".join(out)

class KbFileProcessingThread(QThread):
    processing_done = pyqtSignal(list)

    def __init__(self, selected_folder):
        super().__init__()
        self.selected_folder = selected_folder

    def run(self):
        file_ids = load_Kb_files(self.selected_folder)
        self.processing_done.emit(file_ids)

class ChatbotApp(QWidget):
    def __init__(self, application_name,files_ids=None):
        super().__init__()
        self.files_ids = files_ids
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
        self.input_text.returnPressed.connect(lambda: self.send_message(application_name,self.files_ids))
        input_layout.addWidget(self.input_text)
        self.input_text.setFocus()
        
        self.button_layout = QHBoxLayout()
        self.clear_button = QPushButton(self)
        clear_icon = QPixmap("./assets/reset.png")  # Image du bouton
        self.clear_button.setIcon(QIcon(clear_icon))
        self.clear_button.setIconSize(QSize(30,30))  # Ajuste la taille de l'icône
        self.clear_button.setFixedSize(52, 52)  # Ajuste la taille du bouton
        self.clear_button.clicked.connect(self.reset_conversation)
        input_layout.addWidget(self.clear_button)

        self.button_layout = QHBoxLayout()
        self.send_button = QPushButton(self)
        send_icon = QPixmap("./assets/send.png")  # Image du bouton
        self.send_button.setIcon(QIcon(send_icon))
        self.send_button.setIconSize(QSize(30, 30))  # Ajuste la taille de l'icône
        self.send_button.setFixedSize(52, 52)  # Ajuste la taille du bouton        
        self.send_button.clicked.connect(lambda: self.send_message(application_name,self.files_ids))
        input_layout.addWidget(self.send_button)

        
        self.layout.addLayout(input_layout)

        self.layout.addLayout(self.button_layout)

        self.history = []
        self.history_index = -1

        self.selected_folder = ""  # Store the selected directory

        self.setLayout(self.layout)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", os.getcwd())
        if folder:
            self.selected_folder = folder
            self.chat_display.append(f"<p style='color: blue;'>📂 Dossier sélectionné : {folder}</p>")
            self.input_text.setPlaceholderText("Intégration en cours ...")
            self.input_text.setDisabled(True)
            
            self.spinner.start() 
            QApplication.processEvents()
            
            # ✅ Start a separate thread to process files
            self.file_thread = KbFileProcessingThread(self.selected_folder)
            self.file_thread.processing_done.connect(self.on_processing_done)
            self.file_thread.start()
            
    def on_processing_done(self, files_ids):
        # ✅ Re-enable input and reset UI
        self.chat_display.append(f"<p style='color: green;'>Intégration terminée !</p>")
        self.input_text.setPlaceholderText("Écrivez votre message...")
        self.input_text.setDisabled(False)
        self.spinner.stop()
        self.input_text.setFocus()
            
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

    def send_message(self, application_name, file_ids=None):
        self.files_ids = file_ids
        self.application_name = application_name
        user_message = self.input_text.text().strip()
        
        if user_message:
            self.chat_display.append(f"<p style='font-size: 20px; color: black;'>👤</p> <p style='font-size: 15px;'> {user_message}</p>")
            self.chat_display.append("")
            self.history.append(user_message)
            self.history_index = len(self.history)
            self.input_text.clear()
            self.loading_label.show()
            self.spinner.start()
            
            self.worker = ChatbotWorker(user_message, self.application_name, self.files_ids)
            self.worker.response_ready.connect(self.display_response)
            self.worker.start()       

   
    def display_response(self, bot_reply_html):
        self.spinner.stop()
        self.loading_label.hide()

        # S'assurer que QTextBrowser accepte les liens externes
        self.chat_display.setOpenExternalLinks(True)

        # Remplacement des balises <p> par <span> pour éviter les sauts de ligne excessifs
        bot_reply_html = render_light_md_to_html(bot_reply_html)
        bot_reply_html = bot_reply_html.replace("<p", "<span ").replace("</p>", "</span>").replace("\n", "<br>").replace("30", "15")
        # Ajouter le nouveau contenu à la fin du body
        self.chat_display.moveCursor(QtGui.QTextCursor.End)  # Place le curseur à la fin
        self.chat_display.insertHtml(f"<br>{BOT_AVATAR}<br>{bot_reply_html}<br>")  # Ajoute le nouveau message

    def reset_conversation(self):
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
class ChatbotWorker(QThread):
    response_ready = pyqtSignal(str)
    
    def __init__(self, user_input, application_name, file_ids=None):
        super().__init__()
        self.user_input = user_input
        self.files_ids = file_ids
        self.application_name = application_name
        
    def run(self):
        try:
            thread = OPENAI_CLIENT.beta.threads.create()           
                
            if len(self.files_ids) > 0:
                thread = create_thread_with_files(self.files_ids, self.user_input)
            else:                
                OPENAI_CLIENT.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=self.user_input
                )

            run = OPENAI_CLIENT.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=ASSISTANT_IDS_DIC.get(self.application_name, "Clé non trouvée")
            )
            while True:
                run_status = OPENAI_CLIENT.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

                status = run_status.status

                if status == "completed":
                    break

                if status in ("failed", "cancelled", "expired"):
                    err = getattr(run_status, "last_error", None)
                    if err:
                        raise RuntimeError(f"Run {status}: {err.code} - {err.message}")
                    raise RuntimeError(f"Run {status} (no last_error provided)")

                time.sleep(0.25)


            messages = OPENAI_CLIENT.beta.threads.messages.list(thread_id=thread.id)
            bot_reply = messages.data[0].content[0].text.value
            bot_reply = clean_citations(bot_reply)
             # Convertir le texte Markdown en HTML
            html = markdown.markdown(bot_reply, output_format='html5')
            html = html.replace('<a ', '<a target="_blank" ')
            bot_reply_html = f"<p style='font-size: 15px'>{html}</p>"

        except Exception as e:
            bot_reply_html = f"<span style='color: red;'>Erreur : {str(e)}</span>"
        
        self.response_ready.emit(bot_reply_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    args = sys.argv[1:]  # Exclude the script name
    # download_knowledge_files_from_googleDrive(dest_dir="./Knowledge_base")
    # download_knowledge_files_from_azure(dest_dir=KB_DIRECTORY)
    # file_ids = load_Kb_files()  
    file_ids = []
    window = ChatbotApp(args[0].lower() if args else "kitview", file_ids) 
    window.show()
    sys.exit(app.exec_())
