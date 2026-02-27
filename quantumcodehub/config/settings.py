import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

load_dotenv()

class Settings:
    # ChromaDB settings
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_METADATA = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,
        "hnsw:M": 16
    }
    
    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    EMBEDDING_DIMENSION = 768
    
    # Chunking settings
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    
    # Batch processing
    BATCH_SIZE = 300
    MAX_CONCURRENT_REQUESTS = 10
    
    # Retrieval settings
    DEFAULT_TOP_K = 10
    MIN_OFFICIAL_SOURCES = 1
    MIN_KEYWORD_OVERLAP = 0.3
    
    # Research settings
    ARXIV_API_ENDPOINT = "http://export.arxiv.org/api/query"
    MAX_RESEARCH_PAPERS = 50
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "quantumcodehub.log")
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('__') and not callable(value)}