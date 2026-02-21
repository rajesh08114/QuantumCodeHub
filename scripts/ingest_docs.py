"""
RAG document ingestion script (ChromaDB).
"""
import asyncio
import hashlib
import json
import logging
from pathlib import Path

from core.config import settings
from services.rag_service import rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Ingest framework documentation JSON files into ChromaDB."""

    def __init__(self):
        self.docs_dir = Path("data/documentation")
        self.collection_name = settings.CHROMA_COLLECTION_NAME

    @staticmethod
    def _build_doc_id(framework: str, source_file: str, index: int, content: str) -> str:
        raw = f"{framework}:{source_file}:{index}:{content[:200]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    async def ingest_framework_docs(self, framework: str):
        """
        Ingest one framework's docs from:
        data/documentation/<framework>/*.json
        """
        framework = (framework or "").strip().lower()
        framework_dir = self.docs_dir / framework
        if not framework_dir.exists():
            logger.error("Documentation directory not found: %s", framework_dir)
            return

        doc_count = 0
        for json_file in sorted(framework_dir.glob("*.json")):
            logger.info("Processing %s...", json_file.name)
            with open(json_file, "r", encoding="utf-8") as f:
                docs = json.load(f)

            if not isinstance(docs, list):
                logger.warning("Skipping %s: expected a JSON array", json_file)
                continue

            for idx, doc in enumerate(docs):
                try:
                    content = str(doc.get("content", "")).strip()
                    if not content:
                        continue

                    source = str(doc.get("source", "")).strip() or f"local://{framework}/{json_file.name}"
                    if "://" not in source:
                        source = f"local://{framework}/{source}"

                    metadata = {
                        "source": source,
                        "section": str(doc.get("section", "general")),
                        "framework": framework,
                        "version": str(doc.get("version", "latest")),
                        "type": str(doc.get("type", "framework_doc")),
                    }

                    await rag_service.add_document(
                        collection_name=self.collection_name,
                        doc_id=self._build_doc_id(framework, json_file.name, idx, content),
                        content=content,
                        metadata=metadata,
                    )
                    doc_count += 1

                    if doc_count % 100 == 0:
                        logger.info("Ingested %s documents for %s...", doc_count, framework)
                except Exception as e:
                    logger.error("Error ingesting document from %s: %s", json_file.name, e)

        logger.info("Ingested %s documents for %s into collection '%s'", doc_count, framework, self.collection_name)

    async def ingest_all_frameworks(self):
        """Ingest docs for all supported frameworks."""
        frameworks = ["qiskit", "pennylane", "cirq", "torchquantum"]
        for framework in frameworks:
            logger.info("\n%s", "=" * 50)
            logger.info("Ingesting %s documentation", framework)
            logger.info("%s", "=" * 50)
            await self.ingest_framework_docs(framework)

    def create_sample_docs(self):
        """Create minimal sample docs for local testing."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        qiskit_docs = [
            {
                "content": "The Hadamard gate (H) creates superposition. In Qiskit: qc.h(qubit_index).",
                "source": "gates_reference",
                "section": "single_qubit_gates",
                "version": "1.0",
            },
            {
                "content": "CNOT gate (CX) flips target qubit when control is |1>. In Qiskit: qc.cx(control, target).",
                "source": "gates_reference",
                "section": "two_qubit_gates",
                "version": "1.0",
            },
            {
                "content": "Create a circuit: from qiskit import QuantumCircuit; qc = QuantumCircuit(num_qubits).",
                "source": "circuit_basics",
                "section": "circuit_creation",
                "version": "1.0",
            },
        ]

        qiskit_dir = self.docs_dir / "qiskit"
        qiskit_dir.mkdir(exist_ok=True)
        with open(qiskit_dir / "gates.json", "w", encoding="utf-8") as f:
            json.dump(qiskit_docs, f, indent=2)

        logger.info("Created sample documentation files")


async def main():
    ingestion = DocumentIngestion()
    if not ingestion.docs_dir.exists():
        ingestion.create_sample_docs()
    await ingestion.ingest_all_frameworks()
    logger.info("Documentation ingestion complete")


if __name__ == "__main__":
    asyncio.run(main())
