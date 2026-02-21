"""
Initialize and inspect Chroma collection for RAG.
"""
from pathlib import Path

import chromadb

from core.config import settings


def _resolve_persist_dir(path_value: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def setup_chroma_collection():
    persist_dir = _resolve_persist_dir(settings.CHROMA_PERSIST_DIR)
    collection_name = settings.CHROMA_COLLECTION_NAME

    print(f"Using Chroma persist dir: {persist_dir}")
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection already exists: {collection_name}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Created collection: {collection_name}")

    print(f"Collection document count: {collection.count()}")
    available = [c.name if hasattr(c, 'name') else str(c) for c in client.list_collections()]
    print(f"Available collections: {available}")


if __name__ == "__main__":
    setup_chroma_collection()
