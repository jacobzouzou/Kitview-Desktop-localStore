# Kitview Desktop Chatbot

Assistant IA intelligent avec interface PyQt5 pour diverses applications médicales et techniques.

## 📋 Table des matières

- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)

## 🚀 <a id="installation">Installation</a>

### Prérequis
- Python 3.13+
- Clé API OpenAI
- Windows (optimisé pour Win32)

### Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv env313
source env313/Scripts/activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- PyQt5 (interface graphique)
- OpenAI (API GPT)
- PyPDF2, python-docx, python-pptx (traitement documents)
- beautifulsoup4 (parsing HTML)
- pandas (manipulation données)

## ⚙️ <a id="configuration">Configuration</a>

1. Créez un fichier `.env` à la racine :
```env
KITVIEW_DESKTOP_OPENAI_API_KEY=your_openai_api_key
KITVIEW_DESKTOP_OPENAI_ASSITANT_VECTORE_STORE_ID=your_vector_store_id
ASSISTANT_IDS={"kitview":"assistant_id","orqual":"assistant_id2"}
```

2. Configurez les secrets dans `config/`:
   - `client_secrets.json` (Google Drive)
   - `credentials.json` (Azure)

## 🖥️ <a id="utilisation">Utilisation</a>

### Lancement de l'application
```bash
python cloud_chabot.py kitview
```

### Fonctionnalités principales
- **Chat intelligent** : Conversations avec assistant IA spécialisé
- **Upload de documents** : Support PDF, DOC, XLS, PPT, JSON, CSV
- **Base de connaissances** : Intégration automatique de dossiers
- **Historique** : Navigation avec ↑/↓ dans les messages précédents

## 🏗️ <a id="architecture">Architecture</a>

### Structure du projet
```
├── cloud_chabot.py          # Application principale
├── standalone_chabot.py     # Version locale avec RAG
├── Helpers/                 # Modules utilitaires
│   ├── azure.py            # Intégration Azure
│   ├── google.py           # Intégration Google Drive
│   └── faissClass.py       # Recherche vectorielle
├── rag_local/              # RAG local
│   ├── llm_router.py       # Routage LLM
│   └── query_helper.py     # Traitement requêtes
└── Knowledge_base/         # Base de connaissances
```

### Classes principales

#### `ChatbotApp`
Interface principale PyQt5 gérant :
- Affichage des conversations
- Upload de fichiers
- Interactions utilisateur

#### `ChatbotWorker` 
Thread worker pour :
- Appels API OpenAI asynchrones
- Traitement des réponses
- Gestion des erreurs

#### `KbFileProcessingThread`
Thread pour le traitement des fichiers :
- Upload vers OpenAI
- Intégration vector store
- Feedback utilisateur

## 📚 <a id="api-documentation">API Documentation</a>

### Fonctions principales

#### `openaiUploadFiles(files) -> list`
Upload multiple files vers OpenAI.
- **Paramètres** : `files` (list) - Chemins des fichiers
- **Retour** : Liste des file IDs OpenAI
- **Exceptions** : Exception si échec upload

#### `get_kb_files(directory) -> list`
Récupère les fichiers supportés d'un dossier.
- **Formats supportés** : PDF, DOC, XLS, PPT, CSV, JSON
- **Retour** : Liste des chemins de fichiers valides

#### `create_thread_with_files(file_ids, user_question) -> Thread`
Crée un thread OpenAI avec fichiers attachés.
- **Paramètres** : 
  - `file_ids` (list) - IDs des fichiers uploadés
  - `user_question` (str) - Question utilisateur
- **Retour** : Objet Thread OpenAI

## 🔧 Compilation

### Création d'un exécutable
```bash
pyinstaller --onefile --windowed cloud_chabot.py
```

### Options avancées
```bash
pyinstaller --onefile --windowed --icon=assets/kitview.ico --add-data "assets;assets" --add-data "config;config" cloud_chabot.py
```

## 🐛 Dépannage

### Erreurs communes
- **ImportError** : Vérifiez l'installation des dépendances
- **API Key invalide** : Vérifiez le fichier `.env`
- **Fichiers non supportés** : Consultez la liste des formats supportés

### Logs et debugging
Les erreurs sont affichées dans la console et l'interface utilisateur.

## 🤝 Contribution

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 License

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.