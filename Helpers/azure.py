from azure.storage.blob import BlobServiceClient
import re
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

_WINDOWS_RESERVED = r'<>:"/\\|?*'

def _safe_rel_path(blob_name: str) -> Path:
    blob_name = blob_name.replace("\\", "/").lstrip("/")

    parts = []
    for part in blob_name.split("/"):
        part = re.sub(f"[{re.escape(_WINDOWS_RESERVED)}]", "_", part)
        if part.strip():
            parts.append(part)

    return Path(*parts) if parts else Path("unnamed_blob")

def download_knowledge_files_from_azure(container_name="kitview", dest_dir: str = "./Knowledge_base"):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING manquant dans les variables d’environnement.")

    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    downloaded = 0
    skipped = 0

    for blob in container_client.list_blobs():
        blob_name = blob.name
        rel_path = _safe_rel_path(blob_name)
        local_file_path = (dest_path / rel_path).resolve()
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # ✅ Skip if file exists
        if local_file_path.exists():
            local_size = local_file_path.stat().st_size
            remote_size = getattr(blob, "size", None)

            if remote_size is not None and local_size == remote_size:
                skipped += 1
                print(f"[SKIP] Same size: {blob_name}")
                continue

        blob_client = container_client.get_blob_client(blob_name)
        with open(local_file_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        downloaded += 1
        print(f"[OK] Downloaded: {blob_name} -> {local_file_path}")

    print(f"[DONE] Downloaded={downloaded} | Skipped={skipped}")
  
# def main():
#     download_container_to_knowledge_base_skip_existing()

# if __name__ == "__main__":
#     main()