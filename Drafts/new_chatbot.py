import os
from PyQt5.QtCore import Qt, QSize, QUrl
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextBrowser, QLabel, QLineEdit,
    QPushButton, QFileDialog, QHBoxLayout, QApplication
)
from PyQt5 import QtGui


class ChatbotApp(QWidget):
    def __init__(self, application_name, files_ids=None):
        super().__init__()
        self.files_ids = files_ids
        self.application_name = application_name

        # --- Assets / HTML ---
        # Bot avatar provided by you
        self.botAvatar = "<img src='./assets/chatbot.png' width='30' height='30'>"

        # Typing gif absolute path -> reliable for QTextBrowser
        self.typing_gif_path = os.path.abspath("./assets/typing.gif")

        # Markers to remove typing block cleanly
        self.typing_marker_start = "<!--TYPING_START-->"
        self.typing_marker_end = "<!--TYPING_END-->"

        # --- Window ---
        self.setWindowTitle(f"Assistant IA - {application_name[:1].upper()}{application_name[1:]}")
        self.setGeometry(150, 150, 600, 800)
        self.setWindowIcon(self.get_icon(application_name))

        # --- Layout ---
        self.layout = QVBoxLayout()

        # --- Chat display ---
        self.chat_display = QTextBrowser(self)
        self.chat_display.setReadOnly(True)

        # IMPORTANT: allow loading local images referenced by relative or file:// paths
        self.chat_display.document().setBaseUrl(QUrl.fromLocalFile(os.getcwd() + os.sep))

        # Greeting (assumes greetingMessage exists in your codebase)
        self.chat_display.setHtml(f"{self.botAvatar}<p style='font-size:15px;'>{greetingMessage}</p>")
        self.layout.addWidget(self.chat_display)

        # --- Input row ---
        input_layout = QHBoxLayout()

        self.select_folder_button = QPushButton(self)
        dir_icon = QPixmap("./assets/dir_icon.png")
        self.select_folder_button.setIcon(QIcon(dir_icon))
        self.select_folder_button.setIconSize(QSize(30, 30))
        self.select_folder_button.setFixedSize(52, 52)
        self.select_folder_button.clicked.connect(self.select_folder)
        input_layout.addWidget(self.select_folder_button)

        self.input_text = QLineEdit(self)
        self.input_text.setFixedHeight(49)
        self.input_text.setPlaceholderText("Écrivez votre message...")
        self.input_text.installEventFilter(self)
        self.input_text.setStyleSheet("font-size: 15px;")
        self.input_text.returnPressed.connect(lambda: self.send_message(self.application_name, self.files_ids))
        input_layout.addWidget(self.input_text)
        self.input_text.setFocus()

        self.clear_button = QPushButton(self)
        clear_icon = QPixmap("./assets/reset.png")
        self.clear_button.setIcon(QIcon(clear_icon))
        self.clear_button.setIconSize(QSize(30, 30))
        self.clear_button.setFixedSize(52, 52)
        self.clear_button.clicked.connect(self.clear_conversation)
        input_layout.addWidget(self.clear_button)

        self.send_button = QPushButton(self)
        send_icon = QPixmap("./assets/send.png")
        self.send_button.setIcon(QIcon(send_icon))
        self.send_button.setIconSize(QSize(30, 30))
        self.send_button.setFixedSize(52, 52)
        self.send_button.clicked.connect(lambda: self.send_message(self.application_name, self.files_ids))
        input_layout.addWidget(self.send_button)

        self.layout.addLayout(input_layout)

        # --- History ---
        self.history = []
        self.history_index = -1

        # --- Folder ---
        self.selected_directory = ""

        self.setLayout(self.layout)

    # -------------------------
    # Typing indicator (INLINE)
    # -------------------------
    def show_typing_indicator(self):
        gif_url = QUrl.fromLocalFile(self.typing_gif_path).toString()

        typing_html = f"""
        {self.typing_marker_start}
        <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
            {self.botAvatar}
            <img src="{gif_url}" width="55" height="35"/>
        </div>
        {self.typing_marker_end}
        """

        self.chat_display.append(typing_html)
        self.chat_display.moveCursor(QtGui.QTextCursor.End)

    def remove_typing_indicator(self):
        html = self.chat_display.toHtml()
        start = html.find(self.typing_marker_start)
        end = html.find(self.typing_marker_end)

        if start != -1 and end != -1:
            end += len(self.typing_marker_end)
            html = html[:start] + html[end:]
            self.chat_display.setHtml(html)
            self.chat_display.moveCursor(QtGui.QTextCursor.End)

    # -------------------------
    # Folder selection
    # -------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", os.getcwd())
        if folder:
            self.selected_directory = folder
            self.chat_display.append(f"<p style='color: blue;'>📂 Dossier sélectionné : {folder}</p>")
            self.input_text.setPlaceholderText("Chargement en cours ...")
            self.input_text.setDisabled(True)

            # ✅ Show inline typing indicator while processing folder
            self.show_typing_indicator()
            QApplication.processEvents()

            # ✅ Start a separate thread to process files (assumes FileProcessingThread exists)
            self.file_thread = FileProcessingThread(self.selected_directory)
            self.file_thread.processing_done.connect(self.on_processing_done)
            self.file_thread.start()

    def on_processing_done(self):
        # ✅ Remove typing indicator and restore UI
        self.remove_typing_indicator()

        self.chat_display.append(
            f"<p style='color: green;'>🎉 Fusion terminée pour {self.selected_directory}!</p>"
            f"dans répertoire ./Outputs"
        )
        self.input_text.setPlaceholderText("Écrivez votre message...")
        self.input_text.setDisabled(False)
        self.input_text.setFocus()

    # -------------------------
    # Input history navigation
    # -------------------------
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

    # -------------------------
    # Chat sending
    # -------------------------
    def send_message(self, application_name, file_ids=None):
        self.files_ids = file_ids
        self.application_name = application_name
        user_message = self.input_text.text().strip()

        if user_message:
            self.chat_display.append(
                f"<p style='font-size: 20px;'>👤</p> "
                f"<p style='font-size: 15px;'>{user_message}</p>"
            )
            self.chat_display.append("")

            self.history.append(user_message)
            self.history_index = len(self.history)
            self.input_text.clear()

            # ✅ Inline typing indicator on same line as avatar
            self.show_typing_indicator()

            # Assumes ChatbotWorker exists in your code
            self.worker = ChatbotWorker(user_message, self.application_name, self.files_ids)
            self.worker.response_ready.connect(self.display_response)
            self.worker.start()

    def display_response(self, bot_reply_html):
        # ✅ Remove typing indicator
        self.remove_typing_indicator()

        # Allow links
        self.chat_display.setOpenExternalLinks(True)

        # Your existing formatting logic
        bot_reply_html = (
            bot_reply_html
            .replace("<p", "<span ")
            .replace("</p>", "</span>")
            .replace("30", "20")
        )

        self.chat_display.moveCursor(QtGui.QTextCursor.End)
        self.chat_display.insertHtml(f"<br>{self.botAvatar}<br>{bot_reply_html}<br>")

    # -------------------------
    # Utilities
    # -------------------------
    def clear_conversation(self):
        self.chat_display.setHtml(f"{self.botAvatar}<p style='font-size:20px;'>{greetingMessage}</p>")

    def get_icon(self, app_name):
        icons = {
            "orqual": "./assets/orqual-removebg-preview.png",
            "orthalis": "./assets/Orthalis-new.png",
            "dentalis": "./assets/Dentalis.png",
            "dentapoche": "./assets/Dentapoche.png",
            "kitview": "./assets/KitView.png",
            "ceph": "./assets/ceph.png",
        }

        icon_path = icons.get(app_name.lower(), "./assets/orqual.png")

        if not os.path.exists(icon_path):
            print(f"⚠️ L'icône pour '{app_name}' n'existe pas : {icon_path}")
            return QIcon()

        pixmap = QPixmap(icon_path)

        if pixmap.width() < 128 or pixmap.height() < 128:
            pixmap = pixmap.scaled(256, 256)

        return QIcon(pixmap)
