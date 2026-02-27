from typing import List, Dict, Any, Optional
from loguru import logger
import json
import os
from pathlib import Path
import hashlib

from retrieval.chroma_manager import ChromaManager
from data.processors.document_processor import DocumentProcessor
from config.constants import COLLECTIONS, Framework, DocType
from models.document import Document, DocumentMetadata

class DocumentIngestor:
    """Handles document ingestion into the retrieval system"""
    
    def __init__(self):
        self.chroma_manager = ChromaManager()
        self.document_processor = DocumentProcessor()
        self.ingestion_stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'collections': {}
        }
    
    def ingest_document(self, 
                       content: str, 
                       metadata: Dict[str, Any],
                       collection_name: Optional[str] = None) -> bool:
        """Ingest a single document"""
        try:
            # Process document into chunks
            chunks = self.document_processor.process_document(content, metadata)
            
            if not chunks:
                logger.warning("No chunks generated from document")
                return False
            
            # Determine target collection
            if not collection_name:
                collection_name = self._get_collection_for_doc_type(
                    metadata.get('doc_type')
                )
            
            # Add chunks to collection
            self.chroma_manager.add_documents(collection_name, chunks)
            
            # Update stats
            self._update_stats(collection_name, len(chunks))
            
            logger.info(f"Successfully ingested document with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return False
    
    def ingest_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest multiple documents in batch"""
        results = {
            'success': 0,
            'failed': 0,
            'total_chunks': 0
        }
        
        for doc in documents:
            success = self.ingest_document(
                content=doc.get('content', ''),
                metadata=doc.get('metadata', {})
            )
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Batch ingestion complete: {results}")
        return results
    
    def ingest_from_file(self, file_path: str) -> bool:
        """Ingest documents from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return self.ingest_batch(data)
            elif isinstance(data, dict):
                return self.ingest_document(
                    content=data.get('content', ''),
                    metadata=data.get('metadata', {})
                )
            else:
                logger.error(f"Invalid data format in {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting from file {file_path}: {e}")
            return False
    
    def ingest_directory(self, directory_path: str, pattern: str = "*.json") -> Dict[str, int]:
        """Ingest all JSON files in a directory"""
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_chunks': 0
        }
        
        path = Path(directory_path)
        for file_path in path.glob(pattern):
            results['total_files'] += 1
            logger.info(f"Ingesting {file_path}")
            
            success = self.ingest_from_file(str(file_path))
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Directory ingestion complete: {results}")
        return results
    
    def _get_collection_for_doc_type(self, doc_type: str) -> str:
        """Map document type to collection name"""
        doc_type_to_collection = {
            DocType.API.value: COLLECTIONS["api_docs"],
            DocType.TUTORIAL.value: COLLECTIONS["tutorials"],
            DocType.RESEARCH.value: COLLECTIONS["research"],
            DocType.BOOK.value: COLLECTIONS["books"],
            DocType.RELEASE.value: COLLECTIONS["release_notes"],
            DocType.DEPRECATION.value: COLLECTIONS["deprecations"]
        }
        
        return doc_type_to_collection.get(doc_type, COLLECTIONS["api_docs"])
    
    def _update_stats(self, collection_name: str, num_chunks: int):
        """Update ingestion statistics"""
        self.ingestion_stats['total_documents'] += 1
        self.ingestion_stats['total_chunks'] += num_chunks
        
        if collection_name not in self.ingestion_stats['collections']:
            self.ingestion_stats['collections'][collection_name] = 0
        
        self.ingestion_stats['collections'][collection_name] += num_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return self.ingestion_stats.copy()