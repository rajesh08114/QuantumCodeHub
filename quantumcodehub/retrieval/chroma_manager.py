import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import Settings
from config.constants import COLLECTIONS, Framework
from models.document import Document, RetrievalResult

class ChromaManager:
    """Manages ChromaDB collections and operations"""
    
    def __init__(self, persist_directory: str = Settings.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.client = None
        self.embedding_model = None
        self.collections = {}
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(Settings.EMBEDDING_MODEL)
            
            # Initialize collections
            self._initialize_collections()
            
            logger.info(f"ChromaManager initialized with persist dir: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaManager: {e}")
            raise
    
    def _initialize_collections(self):
        """Initialize all required collections"""
        for collection_name in COLLECTIONS.values():
            try:
                # Try to get existing collection
                collection = self.client.get_collection(collection_name)
            except ValueError:
                # Create new collection
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata=Settings.CHROMA_COLLECTION_METADATA
                )
            
            self.collections[collection_name] = collection
            logger.info(f"Initialized collection: {collection_name}")
    
    def add_documents(self, collection_name: str, documents: List[Document]):
        """Add documents to a collection"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection {collection_name} not found")
            
            # Prepare data for insertion
            ids = []
            embeddings = []
            metadatas = []
            texts = []
            
            for doc in documents:
                ids.append(doc.id)
                
                # Generate embedding if not present
                if doc.embeddings:
                    embeddings.append(doc.embeddings)
                else:
                    embedding = self.embedding_model.encode(doc.content).tolist()
                    embeddings.append(embedding)
                
                # Convert metadata to dict
                metadatas.append(doc.metadata.dict())
                texts.append(doc.content)
            
            # Add to collection in batches
            batch_size = Settings.BATCH_SIZE
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                
                collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    documents=texts[i:batch_end]
                )
                
                logger.info(f"Added batch {i//batch_size + 1} to {collection_name}")
            
            logger.info(f"Successfully added {len(documents)} documents to {collection_name}")
            
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            raise
    
    def query(self, 
              collection_names: List[str], 
              query_text: str,
              n_results: int = Settings.DEFAULT_TOP_K,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query multiple collections"""
        try:
            all_results = []
            
            for collection_name in collection_names:
                collection = self.collections.get(collection_name)
                if not collection:
                    logger.warning(f"Collection {collection_name} not found, skipping")
                    continue
                
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query_text).tolist()
                
                # Query collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=["metadatas", "documents", "distances"]
                )
                
                # Format results
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'collection': collection_name
                    })
            
            # Sort by distance (lower is better)
            all_results.sort(key=lambda x: x['distance'])
            
            return all_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error querying collections: {e}")
            return []
    
    def delete_documents(self, collection_name: str, ids: List[str]):
        """Delete documents from a collection"""
        try:
            collection = self.collections.get(collection_name)
            if collection:
                collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents from {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                return {}
            
            count = collection.count()
            
            return {
                'name': collection_name,
                'count': count,
                'metadata': collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def reset(self):
        """Reset all collections (for testing)"""
        try:
            for collection_name in self.collections:
                self.client.delete_collection(collection_name)
            
            self.collections.clear()
            self._initialize_collections()
            
            logger.info("Reset all collections")
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")