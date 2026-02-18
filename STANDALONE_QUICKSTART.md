# 🚀 Guide de démarrage rapide - Kitview Standalone

## ✅ Le problème résolu

L'erreur `AZURE_STORAGE_CONNECTION_STRING manquant` a été corrigée. L'application vérifie maintenant si Azure est configuré avant d'essayer de se connecter.

## 📦 Fichiers nécessaires

```
standalone_chatbot.exe (ou dossier)
├── .env                    # Configuration (créer depuis .env.example)
├── assets/                 # Images et icônes (inclus)
├── config/                 # Configurations optionnelles
└── knowledge_base/         # Base de connaissances locale
    └── raw/               # Déposez vos documents ici
```

## 🏗️ Build & distribution

### Build recommandé (mode dossier / onedir)

```bash
pyinstaller standalone_chatbot.spec
```

Le livrable à distribuer est le dossier complet:

```
dist/standalone_chatbot/
├── standalone_chatbot.exe
├── .env
└── _internal/
    ├── assets/
    ├── config/
    └── knowledge_base/
```

Option recommandé pour une KB modifiable côté utilisateur:

```
dist/standalone_chatbot/
└── knowledge_base/
    └── raw/
```

Le chatbot utilisera en priorité `knowledge_base/raw` à côté de l'exe si ce dossier existe.

### Pourquoi onedir ?

- `.env` reste modifiable sans recompiler
- `assets/` reste accessible pour l'UI
- `knowledge_base/raw/` reste accessible pour ingestion locale
- Évite les problèmes de chemins relatifs au démarrage

## ⚙️ Configuration minimale

### 1. Créez un fichier `.env` à côté de l'exécutable

```env
# Configuration minimale pour démarrer
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

### 2. Installez et lancez Ollama

```bash
# Téléchargez Ollama depuis https://ollama.ai
# Installez le modèle
ollama pull phi3:mini

# Lancez le serveur (déjà lancé par défaut)
ollama serve
```

### 3. Lancez l'application

```bash
# Windows
standalone_chatbot.exe kitview

# Ou double-cliquez sur l'exécutable
```

## 📚 Ajout de documents

### Méthode 1: Dossier local (recommandé)

1. Créez le dossier `knowledge_base/raw/` à côté de l'exécutable
2. Copiez vos documents (PDF, DOCX, TXT, CSV)
3. L'indexation se fait automatiquement au démarrage

### Méthode 2: Via l'interface

1. Cliquez sur l'icône 📁 dans l'interface
2. Sélectionnez un dossier
3. Les fichiers seront automatiquement indexés

## ☁️ Configuration cloud (Optionnel)

### Azure Storage

Pour synchroniser depuis Azure Blob Storage:

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your-account;AccountKey=your-key;EndpointSuffix=core.windows.net
```

### Google Drive

Pour synchroniser depuis Google Drive:

1. Obtenez `client_secrets.json` depuis Google Cloud Console
2. Placez-le dans `config/client_secrets.json`
3. Ajoutez dans `.env`:

```env
CLIENT_SECRETS=config/client_secrets.json
GOOGLE_DRIVE_TARGET_FOLDER_ID=your-folder-id
```

## 🔧 Résolution de problèmes

### L'application ne démarre pas

**Vérifiez**:
- Ollama est installé et lancé (`ollama serve`)
- Le modèle est téléchargé (`ollama list`)
- Le fichier `.env` existe

### Erreur Ollama connection refused

```bash
# Vérifiez qu'Ollama fonctionne
curl http://localhost:11434/api/tags

# Ou testez
ollama run phi3:mini "Hello"
```

### Pas de réponses / Recherche vide

**Vérifiez**:
- Les documents sont dans `knowledge_base/raw/`
- L'indexation s'est terminée (voir logs au démarrage)
- Les formats sont supportés: `.pdf`, `.docx`, `.txt`, `.csv`

### Azure Storage non configuré

✅ **C'est normal!** Azure est optionnel. L'application fonctionne uniquement en local si vous ne le configurez pas.

Message au démarrage:
```
ℹ️ Azure Storage non configuré, utilisation de la base locale uniquement
```

## 🎯 Utilisation

### Chat simple

Tapez votre question et appuyez sur Entrée:
```
Quelle est la procédure pour...?
```

### Navigation historique

- `↑` (Flèche haut): Message précédent
- `↓` (Flèche bas): Message suivant

### Réinitialisation

Cliquez sur le bouton 🔄 pour effacer la conversation

## 📊 Modèles LLM recommandés

| Modèle | Taille | VRAM | Performance |
|--------|--------|------|-------------|
| `phi3:mini` | 3.8 GB | 4 GB | ⭐⭐⭐ Rapide |
| `llama3.1:8b` | 8 GB | 8 GB | ⭐⭐⭐⭐ Équilibré |
| `qwen2.5:7b` | 7 GB | 8 GB | ⭐⭐⭐⭐ Multilingue |
| `mistral:7b` | 7 GB | 8 GB | ⭐⭐⭐⭐ Précis |

### Changer de modèle

```bash
# Télécharger un nouveau modèle
ollama pull llama3.1:8b

# Modifier dans .env
OLLAMA_MODEL=llama3.1:8b
```

## 🔐 Sécurité

- Les clés API restent sur votre machine
- Aucune donnée n'est envoyée en ligne (sauf si OpenAI configuré)
- La base de connaissances est locale

## 🆘 Support

### Logs

Pour activer les logs détaillés, lancez depuis le terminal:

```bash
# Windows PowerShell
$env:PYTHONVERBOSE="1"
.\standalone_chatbot.exe kitview
```

### Réinitialiser la base de connaissances

```bash
# Supprimez les index
rm -rf knowledge_base/*.index
rm -rf knowledge_base/store.sqlite

# Relancez l'application pour ré-indexer
```

## 📖 Documentation complète

Voir `TECHNICAL_DOCUMENTATION.md` pour:
- Architecture détaillée
- API complète
- Configuration avancée
- Optimisations

---

**Version**: 1.0  
**Build**: 2026-02-18  
**Taille**: ~336 MB
