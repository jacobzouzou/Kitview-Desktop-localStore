# Kitview Desktop Chatbot - Documentation Technique Complète

## Table des matières

1. [Vue d'ensemble architecturale](#vue-densemble-architecturale)
2. [Architecture du système](#architecture-du-système)
3. [API et fonctions principales](#api-et-fonctions-principales)
4. [Intégrations cloud](#intégrations-cloud)
5. [Système RAG local](#système-rag-local)
6. [Configuration](#configuration)
7. [Déploiement et compilation](#déploiement-et-compilation)
8. [Dépannage](#dépannage)

---

## Vue d'ensemble architecturale

### Description générale

Kitview est une application chatbot intelligent multi-mode offrant deux architectures :

1. **Cloud Mode** (`cloud_chatbot.py`): Utilise OpenAI API avec Vector Store intégré
2. **Local RAG Mode** (`standalone_chabot.py`): Utilise un système RAG local avec Ollama/LLMs locaux

### Modes de fonctionnement

```
┌─────────────────────────────────────────────────────────────┐
│                  Kitview Chatbot System                     │
├──────────────────────┬──────────────────────────────────────┤
│   CLOUD MODE         │        LOCAL RAG MODE                │
│ (cloud_chatbot.py)   │   (standalone_chabot.py)             │
├──────────────────────┼──────────────────────────────────────┤
│ • OpenAI API         │ • Ollama/Local LLM                   │
│ • Vector Store       │ • FAISS Indexes (text + image)       │
│ • Threading          │ • SQLite Metadata                    │
│ • File Upload        │ • Sentence Transformers              │
│                      │ • CrossEncoder Reranker              │
│                      │ • Multi-modal retrieval              │
└──────────────────────┴──────────────────────────────────────┘
```

---

## Architecture du système

### Composants principaux

#### 1. **Couche Interface Utilisateur (PyQt5)**

```
ChatbotApp (QWidget)
├── chat_display (QTextBrowser)
│   └── Affiche les messages et images en HTML
├── input_text (QLineEdit)
│   └── Saisie utilisateur avec historique (↑/↓)
├── select_folder_button (QPushButton)
│   └── Navigation et upload de dossiers
├── send_button, clear_button (QPushButton)
│   └── Actions d'envoi et de réinitialisation
└── loading_label (QLabel + QMovie)
    └── Animation de chargement
```

#### 2. **Couche métier globale**

```
├── Helpers/
│   ├── google.py          # Intégration Google Drive
│   ├── azure.py           # Intégration Azure Storage
│   ├── faissClass.py      # Gestion des indexes FAISS
│   ├── chuncks.py         # Chunking de documents
│   ├── ExtractPdfOCR.py   # Extraction OCR
│   ├── buildprompt.py     # Construction de prompts
│   └── localPersistence.py # Persistance locale
├── rag_local/
│   ├── index_helper.py    # Construction et mises à jour d'index
│   ├── query_helper.py    # Requêtes aux indexes FAISS
│   ├── llm_router.py      # Routage vers LLM (Ollama/OpenAI)
│   ├── local_llm_client.py # Client Ollama
│   ├── openai_client.py   # Client OpenAI API
│   ├── final_answer_formatter.py # Formatage réponses
│   └── requirements.txt   # Dépendances RAG
```

#### 3. **Couche données**

```
Knowledge_base/
├── raw/                   # Documents bruts
├── faiss_text.index       # Index FAISS texte
├── faiss_image.index      # Index FAISS images
├── store.sqlite           # Métadonnées SQLite
├── text_chunks.jsonl      # Chunks texte
├── images.jsonl           # Métadonnées images
├── kb_meta.json           # Configuration KB
└── cache/                 # Cache OCR/processing
```

---

## API et fonctions principales

### Cloud Mode - cloud_chatbot.py

#### Helper: `get_asset_path(asset_name)`

**Description**: Résout les chemins des ressources pour PyInstaller

```python
def get_asset_path(asset_name: str) -> str:
    """
    Retourne le chemin correct pour les ressources à la fois 
    en mode bundle PyInstaller et en mode développement.
    """
```

**Paramètres**:
- `asset_name` (str): Nom du fichier dans le dossier assets/

**Retour**: Chemin absolu vers la ressource

**Exemple**:
```python
icon_path = get_asset_path("kitview_icon.png")
spinner = QMovie(get_asset_path("typing.gif"))
```

---

#### Fonction: `openaiUploadFiles(files: list) -> list`

**Description**: Upload multiple fichiers vers OpenAI

```python
def openaiUploadFiles(files: list) -> list:
    """
    Upload des fichiers vers OpenAI pour utilisation avec l'assistant.
    """
```

**Paramètres**:
- `files` (list): Liste de chemins de fichiers à uploader

**Retour**: Liste des IDs de fichiers OpenAI

**Exceptions**:
- `FileNotFoundError`: Fichier non trouvé
- `OpenAIError`: Erreur API OpenAI

**Exemple**:
```python
file_ids = openaiUploadFiles(["/path/to/doc1.pdf", "/path/to/doc2.docx"])
# Retour: ['file-xxxxxxx', 'file-yyyyyyy']
```

---

#### Fonction: `get_kb_files(directory: str) -> list`

**Description**: Récupère tous les fichiers supportés d'un répertoire

```python
def get_kb_files(directory: str) -> list:
    """
    Récupère les chemins de tous les fichiers supportés
    dans un répertoire donné.
    """
```

**Formats supportés**: `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, `.csv`, `.json`

**Paramètres**:
- `directory` (str): Chemin du répertoire

**Retour**: Liste des chemins de fichiers valides

**Exemple**:
```python
files = get_kb_files("./documents")
# Retour: ['./documents/file1.pdf', './documents/file2.docx']
```

---

#### Fonction: `wait_for_files_to_be_ready(file_ids: list)`

**Description**: Attend que les fichiers uploadés soit traités par OpenAI

```python
def wait_for_files_to_be_ready(file_ids: list):
    """
    Boucle d'attente jusqu'à ce que tous les fichiers
    soient traités et prêts pour utilisation.
    """
```

**Paramètres**:
- `file_ids` (list): IDs des fichiers à suivre

**Délai d'attente**: 2 secondes entre chaque vérification

---

#### Fonction: `create_thread_with_files(file_ids: list, user_question: str) -> Thread`

**Description**: Crée un thread OpenAI avec fichiers attachés

```python
def create_thread_with_files(file_ids: list, user_question: str):
    """
    Crée un thread OpenAI et y attache des fichiers
    pour la recherche vectorielle.
    """
```

**Paramètres**:
- `file_ids` (list): IDs des fichiers uploadés
- `user_question` (str): Question initiale

**Retour**: Objet Thread OpenAI

**Workflow interne**:
1. Récupère le Vector Store configuré
2. Ajoute les fichiers au Vector Store en batch
3. Crée un thread avec les tool_resources configurées
4. Lance un run avec l'assistant spécifié

---

#### Classe: `ChatbotApp(QWidget)`

**Description**: Interface graphique principale de l'application

**Attributs principaux**:
```python
self.chat_display      # QTextBrowser - Affichage des messages
self.input_text        # QLineEdit - Zone de saisie
self.loading_label     # QLabel - Animation de chargement
self.spinner           # QMovie - Animation GIF
self.bot_avatar        # str - HTML <img> pour avatar bot
self.files_ids         # list - IDs des fichiers chargés
self.history           # list - Historique des messages
self.history_index     # int - Index navigation historique
self.selected_folder   # str - Dossier sélectionné
```

**Méthodes principales**:

##### `__init__(application_name, files_ids=None)`
Initialise l'interface avec le nom et les fichiers

##### `send_message(application_name, files_ids)`
Envoie un message à l'assistant OpenAI
- Lance `ChatbotWorker` en thread séparé
- Affiche l'animation de chargement
- Affiche la réponse en HTML

##### `reset_conversation()`
Réinitialise la conversation

##### `select_folder()`
Ouvre un dialogue pour sélectionner un dossier de documents
- Lance `KbFileProcessingThread`
- Upload les fichiers vers OpenAI

##### `get_icon(app_name: str) -> QIcon`
Récupère l'icône correspondant à l'application

---

#### Classe: `ChatbotWorker(QThread)`

**Description**: Thread worker pour les appels API asynchrones

**Signals**:
```python
response_received = pyqtSignal(str)  # Réponse reçue
error_occurred = pyqtSignal(str)     # Erreur survenue
```

**Workflow**:
1. Récupère l'assistant spécifié
2. Crée un run sur le thread
3. Attend la complétion avec polling
4. Émet le signal `response_received`

---

#### Classe: `KbFileProcessingThread(QThread)`

**Description**: Thread pour traiter et uploader les fichiers

**Signals**:
```python
processing_done = pyqtSignal(list)   # Upload complété
failed = pyqtSignal(str)             # Erreur survenue
```

---

### Local RAG Mode - standalone_chabot.py

#### Classe: `InsgestNewFilesThread(QThread)`

**Description**: Thread pour ingérer de nouveaux fichiers dans la base de connaissances

**Signals**:
```python
processing_done = pyqtSignal(list)   # Fichiers ingérés
failed = pyqtSignal(str)             # Erreur survenue
```

**Workflow**:
1. Sauvegarde les fichiers dans `KB_DIR`
2. Lance l'indexation via `index_helper.ingest()`
3. Émet le signal `processing_done`

---

### RAG Local - Module rag_local/

#### `index_helper.py`

**Configuration**:
```python
KB_DIR = "./knowledge_base"
RAW_DIR = KB_DIR + "/raw"           # Documents bruts
CACHE_DIR = KB_DIR + "/cache"       # Cache OCR
DB_PATH = KB_DIR + "/store.sqlite"  # Base métadonnées
TEXT_INDEX_PATH = KB_DIR + "/faiss_text.index"
IMAGE_INDEX_PATH = KB_DIR + "/faiss_image.index"
TEXT_CHUNKS_JSONL = KB_DIR + "/text_chunks.jsonl"
IMAGES_JSONL = KB_DIR + "/images.jsonl"
KB_META_JSON = KB_DIR + "/kb_meta.json"

# Modèles
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
```

**Formats supportés**: `.pdf`, `.docx`, `.csv`, `.txt`

**Fonctions principales**:

##### `ingest(kb_dir: str)`
Ingère tous les fichiers non-traités du répertoire brut

**Workflow**:
1. Parcourt `raw_dir`
2. Traite les fichiers PDF (texte + images via OCR)
3. Traite les fichiers DOCX
4. Crée les embeddings via Sentence Transformers
5. Construit les indexes FAISS
6. Stocke les métadonnées en SQLite

---

#### `query_helper.py`

**Fonctions principales**:

##### `search_text(query: str, top_k: int = 5) -> List[Dict]`
Recherche textuelle dans la base de connaissances

**Paramètres**:
- `query` (str): Requête de recherche
- `top_k` (int): Nombre de résultats

**Retour**: Liste des chunks texte avec scores

##### `search_images(query: str, max_n: int = 2) -> List[Dict]`
Recherche d'images

**Paramètres**:
- `query` (str): Requête (texte ou images)
- `max_n` (int): Nombre maximum d'images

**Retour**: Liste des images avec métadonnées

##### `retrieve(query: str, top_k_text: int = 5, top_k_images: int = 2) -> Dict`
Recherche complète (texte + images)

---

#### `llm_router.py`

**Fonction**: `call_llm(prompt: str, model: str) -> str`

**Backends supportés**:
- `ollama`: Modèles locaux via Ollama
- `openai`: API OpenAI
- `llamacpp`: Ollama/llama-cpp-python
- `hf`: Hugging Face

**Sélection**:
```python
backend = os.getenv("LLM_BACKEND", "ollama")
```

---

#### `local_llm_client.py`

**Fonction**: `call_ollama_llm_generate(prompt: str, model: str) -> str`

Appelle le serveur Ollama local

**Configuration**:
```
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"  # Par défaut
```

---

#### `openai_client.py`

**Fonction**: `call_llm(prompt: str, model: str) -> str`

Appelle l'API OpenAI

**Nécessite**:
- `OPENAI_API_KEY` dans les variables d'environnement

---

#### `final_answer_formatter.py`

**Fonction**: `format_final_answer_html(text: str) -> str`

Formate la réponse du LLM en HTML avec:
- Titres Markdown
- Listes à puces
- Gestion des citations
- Highlighting des passages importants

---

## Intégrations Cloud

### 1. OpenAI Integration (Cloud Mode)

#### Configuration

```env
KITVIEW_DESKTOP_OPENAI_API_KEY=sk-xxxx
KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID=vs-xxxxx
ASSISTANT_IDS={"kitview":"asst_xxxx","orqual":"asst_yyyy"}
```

#### Architecture

```
User Input
    ↓
OpenAI Assistant API
    ├── File Search (Vector Store)
    ├── Tools (if configured)
    └── Built-in system prompt
    ↓
Response with citations
```

#### Workflow

1. **Upload de fichiers**:
   - `openaiUploadFiles()` → OpenAI Files API
   - Retourne `file_ids`

2. **Création de thread**:
   - `create_thread_with_files()` crée un thread
   - Ajoute les fichiers au Vector Store
   - Configure tool_resources

3. **Génération de réponse**:
   - `ChatbotWorker` lance un run
   - Polling du statut du run
   - Récupération des messages générés

---

### 2. Azure Storage Integration

#### Configuration

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
```

#### Fonction: `download_knowledge_files_from_azure()`

**Paramètres**:
```python
container_name="kitview"      # Conteneur Azure
dest_dir="./Knowledge_base"   # Destination locale
```

**Workflow**:
1. Connexion au Blob Service
2. Liste tous les blobs
3. Vérifie les fichiers existants (taille)
4. Télécharge les nouveaux/modifiés
5. Crée la structure de répertoires

**Formats de chemin**:
- Les caractères Windows réservés sont remplacés: `<>:"/\|?*` → `_`

---

### 3. Google Drive Integration

#### Configuration

```env
CLIENT_SECRETS=config/client_secrets.json
GOOGLE_DRIVE_TARGET_FOLDER_ID=folder_id_xxx
```

#### Dépendances (optionnel)

- Google Drive nécessite PyDrive (ou PyDrive2)
- Si PyDrive n'est pas installé, la fonctionnalité est désactivée et une erreur explicite est levée seulement lors de l'appel

#### Fonction: `download_knowledge_files_from_googleDrive()`

**Authentication**:
- Utilise `settings.yaml` pour configuration PyDrive
- Sauvegarde tokens dans `token.json`
- Refresh automatique des tokens expirés

**Workflow**:
1. Auth via Google OAuth2
2. Accès au dossier cible
3. Filtre par MIME types:
   - `application/pdf`
   - `application/msword`
   - `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
4. Télécharge les fichiers

---

## Système RAG Local

### Vue d'ensemble

```
Input Query
    ↓
[Embedding via Sentence Transformers]
    ↓
┌─────────────────────────────────┐
│  FAISS Index Search             │
├──────────────┬──────────────────┤
│ Text Index   │ Image Index      │
│ all-MiniLM   │ CLIP ViT-B-32    │
└──────────────┴──────────────────┘
    ↓                    ↓
Text Results      Image Results
    ↓                    ↓
[CrossEncoder Reranking (optionnel)]
    ↓                    ↓
Top-k Results
    ↓
[LLM Router]
├── Ollama (local)
├── OpenAI API
└── Llama-cpp
    ↓
Final Answer (formatted HTML)
```

### Modèles Embedding

#### Modèle Texte: `all-MiniLM-L6-v2`
- Dimension: 384
- Optimisé pour: Recherche dense
- License: Apache 2.0

#### Modèle Image: `clip-ViT-B-32-multilingual-v1`
- Dimension: 512
- Basé sur: CLIP (OpenAI)
- Multilingue
- Optimisé pour: Image/text matching

### Métadonnées SQLite

#### Schéma

```sql
-- Métadonnées texte
CREATE TABLE text_metadata (
    id INTEGER PRIMARY KEY,
    faiss_id INTEGER,
    source_file TEXT,
    page_number INTEGER,
    chunk_index INTEGER,
    text BLOB,
    embedding_date TIMESTAMP,
    ...
);

-- Métadonnées images
CREATE TABLE image_metadata (
    id INTEGER PRIMARY KEY,
    faiss_id INTEGER,
    source_file TEXT,
    page_number INTEGER,
    image_data BLOB,
    embedding_date TIMESTAMP,
    ...
);
```

### Index FAISS

#### Index Texte
- Type: L2 (Euclidean distance)
- Dimension: 384 (all-MiniLM)
- Optimisé pour: Recherche rapide sur grands volumes

#### Index Image
- Type: L2
- Dimension: 512 (CLIP)
- Optimisé pour: Matching image/texte

### Reranking (CrossEncoder)

**Modèle par défaut**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Configuration**:
```python
_RERANKER = CrossEncoder(model_name, device="cuda|cpu")
```

**Utilité**:
- Affine les résultats FAISS avec des scores cross-encoder
- Améliore la pertinence
- Thread-safe (mutex `_LOCK`)

---

## Configuration

### Variables d'environnement (.env)

**Note**: en mode standalone, le fichier `.env` est chargé depuis le dossier de l'exécutable.

#### Pour Cloud Mode

```env
# OpenAI
KITVIEW_DESKTOP_OPENAI_API_KEY=sk-xxxx
KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID=vs-xxxxx
ASSISTANT_IDS={"kitview":"asst_xxxx","orqual":"asst_yyyy"}

# Google Drive (optionnel)
CLIENT_SECRETS=config/client_secrets.json
GOOGLE_DRIVE_TARGET_FOLDER_ID=folder_id

# Note: nécessite PyDrive/PyDrive2 uniquement si la synchronisation Google Drive est utilisée

# Azure (optionnel)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
```

#### Pour Local RAG Mode

```env
# LLM Backend
LLM_BACKEND=ollama  # ou openai, llamacpp, hf

# Pour Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini

# Pour OpenAI
OPENAI_API_KEY=sk-xxxx

# RAG
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Fichiers de configuration

#### config/client_secrets.json
Credentials Google OAuth2 (obtenu via Google Cloud Console)

#### config/credentials.json
Credentials Azure AD (optionnel)

#### settings.yaml
Configuration PyDrive
```yaml
client_config_backend: settings_yaml
client_config:
  client_id: "...google.com"
  client_secret: "..."
  auth_uri: "https://accounts.google.com/o/oauth2/auth"
  token_uri: "https://oauth2.googleapis.com/token"
  redirect_uri: "http://localhost:8080/"
  scopes:
    - "https://www.googleapis.com/auth/drive"
```

---

## Déploiement et compilation

### Prérequis

```bash
# Python 3.13+
# Créer l'environnement virtuel
python -m venv env313
source env313/Scripts/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```
# Interface
PyQt5==5.15.x

# Cloud
openai>=1.0
google-auth-oauthlib  # optionnel (Google Drive)
google-auth-httplib2  # optionnel (Google Drive)
google-api-python-client  # optionnel (Google Drive)
PyDrive  # ou PyDrive2, optionnel (Google Drive)

# Azure
azure-storage-blob
azure-identity
azure-core

# RAG Local
sentence-transformers
faiss-cpu  # ou faiss-gpu
torch
python-docx
PyPDF2
python-pptx
beautifulsoup4
pandas
pytesseract
Pillow
numpy
scikit-learn

# Utilitaires
python-dotenv
```

### Compilation PyInstaller

#### Build cloud_chatbot

```bash
# Build via spec (recommandé)
pyinstaller cloud_chatbot.spec
```

**Résultat**: `dist/cloud_chatbot.exe` (~200 MB)

**Option**: pour réduire la taille si Google Drive n'est pas utilisé, passez `EXCLUDE_GOOGLE = True` dans `cloud_chatbot.spec`.

#### Build standalone_chatbot

```bash
# Build via spec (recommandé)
pyinstaller standalone_chatbot.spec
```

**Résultat**: `dist/standalone_chatbot/` (~1.5 GB avec modèles)

### Distribution

Pour `cloud_chatbot.exe`:
```
Distributable/
├── cloud_chatbot.exe
├── .env (à remplir par utilisateur)
└── assets/ (images, icônes)
```

Pour `standalone_chatbot/`:
```
Distributable/
├── standalone_chatbot/
│   ├── standalone_chatbot.exe
│   └── _internal/
│       ├── assets/
│       ├── config/
│       └── knowledge_base/
├── .env
└── knowledge_base/ (optionnel, priorité locale)
    └── raw/
```

**Note**: le binaire cherche `.env` à côté de l'exe, charge les assets depuis `_internal`, et privilégie `knowledge_base/raw` à côté de l'exe si ce dossier existe.

---

## Dépannage

### Cloud Mode

#### Problème: "ImportError: No module named 'openai'"

**Solution**:
```bash
pip install openai>=1.0
```

#### Problème: "API Key invalide"

**Vérification**:
```python
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("KITVIEW_DESKTOP_OPENAI_API_KEY")
print(f"Key loaded: {key[:10]}...")
```

#### Problème: "ModuleNotFoundError: No module named 'pydrive'"

**Cause**: Google Drive activé sans PyDrive installé.

**Solution**:
```bash
pip install PyDrive
```

Ou désactivez Google Drive dans la configuration (pas d'appel à `download_knowledge_files_from_googleDrive`).

#### Problème: "Vector Store not found"

**Vérification**:
- `KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID` correcte
- Vector Store existe dans le workspace OpenAI

#### Problème: "Assets manquants dans l'exe PyInstaller"

**Solution**:
Le code résout les assets via `_MEIPASS` (bundle) ou le dossier de l'exe

- Vérifier que `--add-data "assets;assets"` est présent
- Assets doivent exister dans `assets/` du projet

---

### Local RAG Mode

#### Problème: "CUDA out of memory"

**Solutions**:
```python
# 1. Réduire CHUNK_SIZE dans index_helper.py
CHUNK_SIZE = 500  # au lieu de 900

# 2. Utiliser CPU pour embeddings
device = "cpu"  # au lieu de "cuda"

# 3. Utiliser modèle plus léger
TEXT_MODEL = "all-MiniLM-L6-v2"
```

#### Problème: "Ollama connection refused"

**Vérification**:
```bash
# 1. Ollama must be running
ollama serve

# 2. Test la connexion
curl http://localhost:11434/api/tags

# 3. Vérifier l'URL dans .env
OLLAMA_BASE_URL=http://localhost:11434
```

#### Problème: "FAISS index not found"

**Solution**: Ingérer les fichiers
```python
from rag_local.index_helper import ingest
ingest("./knowledge_base")
```

---

### Logs et Debugging

#### Activer les logs

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Fichiers de log recommandés

Ajouter dans l'app:
```python
# cloud_chatbot.py
logging.basicConfig(
    filename='cloud_chatbot.log',
    level=logging.DEBUG
)
```

#### Vérification des indexes FAISS

```python
import faiss

# Charger et vérifier l'index
index = faiss.read_index("knowledge_base/faiss_text.index")
print(f"Index ntotal: {index.ntotal}")
print(f"Index d (dimension): {index.d}")
```

---

## Performance et Optimisations

### Cloud Mode

| Métrique | Valeur |
|----------|--------|
| Latence réponse | 2-5s |
| Coût par requête | $0.01-0.10 |
| Limite fichiers | 20 GB/assistant |
| Threads simultanés | 10+ |

### Local RAG Mode

| Métrique | Valeur |
|----------|--------|
| Latence recherche | 50-200 ms |
| Latence génération | 5-30s (selon modèle) |
| VRAM requise | 4-12 GB |
| Taille max KB | 50 GB (SSD) |

### Optimisations recommandées

1. **Chunking**: Ajuster `CHUNK_SIZE` et `CHUNK_OVERLAP`
2. **Batch processing**: Traiter plusieurs requêtes en parallèle
3. **Caching**: Utiliser Redis pour les embeddings fréquents
4. **Filtering**: Filtrer par métadonnées avant recherche FAISS

---

## Ressources et références

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [PyQt5 Documentation](https://doc.qt.io/qt-5/)
- [Ollama Documentation](https://ollama.ai/)

---

**Version**: 1.0  
**Date**: 2026-02-18  
**Auteur**: Kitview Team
