"""
Ingest Qiskit + Qiskit IBM Runtime source documentation
directly from installed Python packages into Chroma RAG.

- Extracts classes, methods, functions
- Includes signatures + docstrings
- Fully version-aware
- No web scraping
- Aligned with installed runtime version
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from services.rag_service import rag_service

# Target packages
TARGET_PACKAGES = [
    "qiskit",
    "qiskit_ibm_runtime",
]

DATASET_NAME = "qiskit_source_docs"


# ==========================================================
#  Utilities
# ==========================================================

def stable_doc_id(module: str, name: str, chunk_index: int, content: str) -> str:
    signature = f"{module}|{name}|{chunk_index}|{content[:200]}"
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()


def get_installed_version(pkg_name: str) -> str:
    try:
        module = __import__(pkg_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "unknown"


# ==========================================================
#  Source Extraction
# ==========================================================

def extract_module_docs(module) -> List[Dict[str, str]]:
    documents = []
    module_name = module.__name__

    # Extract module-level docstring
    if inspect.getdoc(module):
        documents.append(
            {
                "name": module_name,
                "content": f"MODULE: {module_name}\n\n{inspect.getdoc(module)}",
                "module": module_name,
                "type": "module",
            }
        )

    # Extract classes + functions
    for name, obj in inspect.getmembers(module):
        try:
            if inspect.isclass(obj) and obj.__module__ == module_name:
                documents.extend(extract_class_docs(obj))
            elif inspect.isfunction(obj) and obj.__module__ == module_name:
                documents.append(format_function_doc(obj, module_name))
        except Exception:
            continue

    return documents


def extract_class_docs(cls) -> List[Dict[str, str]]:
    docs = []
    class_name = cls.__name__
    module_name = cls.__module__

    class_doc = inspect.getdoc(cls) or "No documentation available."
    class_signature = ""
    try:
        class_signature = str(inspect.signature(cls))
    except Exception:
        pass

    content = (
        f"CLASS: {class_name}\n"
        f"MODULE: {module_name}\n"
        f"SIGNATURE: {class_signature}\n\n"
        f"{class_doc}"
    )

    docs.append(
        {
            "name": class_name,
            "content": content,
            "module": module_name,
            "type": "class",
        }
    )

    # Extract methods
    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if method.__module__ != module_name:
            continue
        docs.append(format_method_doc(cls, method))

    return docs


def format_method_doc(cls, method):
    method_name = method.__name__
    module_name = method.__module__

    try:
        signature = str(inspect.signature(method))
    except Exception:
        signature = ""

    docstring = inspect.getdoc(method) or "No documentation available."

    content = (
        f"METHOD: {cls.__name__}.{method_name}\n"
        f"MODULE: {module_name}\n"
        f"SIGNATURE: {signature}\n\n"
        f"{docstring}"
    )

    return {
        "name": f"{cls.__name__}.{method_name}",
        "content": content,
        "module": module_name,
        "type": "method",
    }


def format_function_doc(func, module_name):
    try:
        signature = str(inspect.signature(func))
    except Exception:
        signature = ""

    docstring = inspect.getdoc(func) or "No documentation available."

    content = (
        f"FUNCTION: {func.__name__}\n"
        f"MODULE: {module_name}\n"
        f"SIGNATURE: {signature}\n\n"
        f"{docstring}"
    )

    return {
        "name": func.__name__,
        "content": content,
        "module": module_name,
        "type": "function",
    }


def crawl_package(package_name: str) -> List[Dict[str, str]]:
    documents = []
    try:
        package = __import__(package_name)
    except Exception as exc:
        print(f"[WARN] Package not available: {package_name} ({exc})")
        return documents

    package_path = getattr(package, "__path__", None)
    if not package_path:
        return documents

    for _, modname, _ispkg in pkgutil.walk_packages(package_path, package.__name__ + "."):
        try:
            module = __import__(modname, fromlist=["dummy"])
            documents.extend(extract_module_docs(module))
        except Exception:
            continue

    return documents


# ==========================================================
#  RAG Ingestion
# ==========================================================

class RAGIngestor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    async def ingest(self, documents: List[Dict[str, str]], version_map: Dict[str, str]):
        collection_name = settings.CHROMA_COLLECTION_NAME
        total_chunks = 0

        for doc in documents:
            chunks = self.splitter.split_text(doc["content"])
            for idx, chunk in enumerate(chunks):
                module_name = doc.get("module", "")
                symbol_name = doc.get("name", "")
                metadata = {
                    "framework": "qiskit",
                    "dataset": DATASET_NAME,
                    "module": module_name,
                    "symbol": symbol_name,
                    "symbol_type": doc["type"],
                    "version_map": str(version_map),
                    "source": f"python://{module_name}",
                    "path": module_name,
                    "type": "api_docs",
                    "section": "qiskit_source_api",
                    "priority": 10,
                }

                await rag_service.add_document(
                    collection_name=collection_name,
                    doc_id=stable_doc_id(module_name, symbol_name, idx, chunk),
                    content=chunk,
                    metadata=metadata,
                )
                total_chunks += 1

        print(
            f"[SUCCESS] Ingested {total_chunks} chunks "
            f"into collection='{collection_name}' "
            f"dataset='{DATASET_NAME}'"
        )


# ==========================================================
#  Main
# ==========================================================

async def main():
    all_documents = []
    version_map = {}

    for pkg in TARGET_PACKAGES:
        print(f"[INFO] Crawling package: {pkg}")
        version_map[pkg] = get_installed_version(pkg)
        docs = crawl_package(pkg)
        all_documents.extend(docs)

    print(f"[INFO] Extracted {len(all_documents)} source documentation entries")
    print(f"[INFO] Detected versions: {version_map}")

    ingestor = RAGIngestor()
    await ingestor.ingest(all_documents, version_map)


if __name__ == "__main__":
    asyncio.run(main())
