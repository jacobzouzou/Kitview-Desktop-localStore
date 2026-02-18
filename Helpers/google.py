import os, re
import logging  

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    _GOOGLE_AUTH_IMPORT_ERROR = None
except Exception as e:
    InstalledAppFlow = None
    build = None
    _GOOGLE_AUTH_IMPORT_ERROR = e

from pathlib import Path

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    _PYDRIVE_IMPORT_ERROR = None
except Exception as e:
    GoogleAuth = None
    GoogleDrive = None
    _PYDRIVE_IMPORT_ERROR = e

_WINDOWS_RESERVED = r'<>:"/\\|?*'
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

CLIENT_SECRETS = os.getenv("CLIENT_SECRETS")
SETTINGS_YAML = "settings.yaml"
FOLDER_ID = os.getenv("GOOGLE_DRIVE_TARGET_FOLDER_ID")
MIME_TYPES = [
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]


def _require_pydrive() -> None:
    if GoogleAuth is None or GoogleDrive is None:
        raise RuntimeError(
            "Google Drive nécessite PyDrive. Installez-le avec: pip install PyDrive"
        ) from _PYDRIVE_IMPORT_ERROR


def _require_google_auth_libs() -> None:
    if InstalledAppFlow is None or build is None:
        raise RuntimeError(
            "Google Drive nécessite google-auth-oauthlib et google-api-python-client."
        ) from _GOOGLE_AUTH_IMPORT_ERROR


def get_drive() -> "GoogleDrive":
    _require_pydrive()
    _require_google_auth_libs()
    if not CLIENT_SECRETS:
        raise RuntimeError("CLIENT_SECRETS non défini")

    gauth = GoogleAuth(SETTINGS_YAML)
    gauth.LoadClientConfigFile(CLIENT_SECRETS)

    # 🔐 Auth persistante (clé du problème)
    gauth.LoadCredentialsFile("token.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("token.json")
    return GoogleDrive(gauth)


def _safe_filename(name: str) -> str:
    # Nettoyage minimal pour Windows + chemins
    name = name.strip().replace("\u0000", "")
    name = re.sub(f"[{re.escape(_WINDOWS_RESERVED)}]", "_", name)
    return name or "unnamed"

# --- Votre fonction existante, inchangée (elle liste) ---
def list_file_ids_by_types(mime_types: list[str]) -> list[dict]:
    folder_id = os.getenv("GOOGLE_DRIVE_TARGET_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("GOOGLE_DRIVE_TARGET_FOLDER_ID non défini")

    drive = get_drive()

    mime_query = " or ".join([f"mimeType='{m}'" for m in mime_types])
    query = (
        f"'{folder_id}' in parents and "
        f"({mime_query}) and trashed=false"
    )

    files = drive.ListFile({'q': query}).GetList()

    return [
        {
            "id": f["id"],
            "name": f["title"],
            "mimeType": f["mimeType"]
        }
        for f in files
    ]

# --- Nouvelle fonction pour télécharger les fichiers en sautant ceux qui existent déjà ---
def download_knowledge_files_from_googleDrive(dest_dir: str = "../Knowledge_base") -> None:
    """
    Télécharge les fichiers (liste issue de list_file_ids_by_types) vers dest_dir.
    Ne télécharge rien si un fichier du même nom existe déjà.
    """
    drive = get_drive()

    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    files = list_file_ids_by_types(MIME_TYPES)
    for f in files:
        file_id = f["id"]
        raw_name = f.get("name") or f.get("title") or "unnamed"
        filename = _safe_filename(raw_name)
        target = (dest_path / filename)

        # ✅ Skip si le fichier existe déjà
        if target.exists():
            skipped += 1
            print(f"[SKIP] Exists: {target}")
            continue

        # Si vous préférez gérer les collisions de noms différemment, remplacez
        # la logique ci-dessus par _unique_path(...) sans skip.
        # target = _unique_path(dest_path, filename)

        gfile = drive.CreateFile({"id": file_id})

        # Téléchargement
        gfile.GetContentFile(str(target))
        downloaded += 1
        print(f"[OK] Downloaded: {raw_name} -> {target}")

    print(f"[DONE] Downloaded={downloaded} | Skipped={skipped}")

if __name__ == "__main__":
    download_knowledge_files_from_googleDrive()

