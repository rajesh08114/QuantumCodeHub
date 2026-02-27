from typing import List, Optional
from loguru import logger
import hashlib
import re

from models.document import Document, DocumentMetadata
from config.constants import Framework, DocType, SOURCE_PRIORITY
from data.chunking.code_chunker import CodeChunker
from data.chunking.research_chunker import ResearchChunker

class DocumentProcessor:
    """Main document processor that coordinates chunking and metadata enrichment"""
    
    def __init__(self):
        self.code_chunker = CodeChunker()
        self.research_chunker = ResearchChunker()
    
    def process_document(self, content: str, metadata: dict) -> List[Document]:
        """Process a single document into chunks with enriched metadata"""
        try:
            # Create document model
            doc_metadata = DocumentMetadata(**metadata)
            
            # Generate document ID
            doc_id = self._generate_doc_id(content, metadata)
            
            document = Document(
                id=doc_id,
                content=content,
                metadata=doc_metadata
            )
            
            # Enrich metadata
            document = self._enrich_metadata(document)
            
            # Chunk based on document type
            if doc_metadata.doc_type in [DocType.API, DocType.TUTORIAL, DocType.CODE_EXAMPLES]:
                chunks = self.code_chunker.chunk_code(document)
            elif doc_metadata.doc_type in [DocType.RESEARCH, DocType.BOOK]:
                chunks = self.research_chunker.chunk_research(document)
            else:
                # Simple chunking for other types
                chunks = self._simple_chunk(document)
            
            # Post-process chunks
            chunks = self._post_process_chunks(chunks)
            
            logger.info(f"Processed document {doc_id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return []
    
    def _enrich_metadata(self, document: Document) -> Document:
        """Enrich metadata with derived fields"""
        metadata = document.metadata
        
        # Set source priority if not set
        if not metadata.source_priority:
            metadata.source_priority = self._determine_source_priority(metadata)
        
        # Detect deprecation if not set
        if metadata.doc_type != DocType.DEPRECATION:
            is_deprecated, replacement = self._detect_deprecation(document.content)
            if is_deprecated:
                metadata.is_deprecated = True
                if replacement:
                    metadata.replacement = replacement
        
        return document
    
    def _determine_source_priority(self, metadata: DocumentMetadata) -> int:
        """Determine source priority based on metadata"""
        # Check source URL
        if metadata.source_url:
            if 'qiskit.org/documentation' in metadata.source_url:
                return SOURCE_PRIORITY['official_docs']
            elif 'pennylane.ai/qml' in metadata.source_url:
                return SOURCE_PRIORITY['official_docs']
            elif 'arxiv.org' in metadata.source_url:
                return SOURCE_PRIORITY['arxiv_papers']
            elif 'github.com' in metadata.source_url:
                return SOURCE_PRIORITY['official_tutorials']
        
        # Check document type
        if metadata.doc_type == DocType.API:
            return SOURCE_PRIORITY['official_docs']
        elif metadata.doc_type == DocType.BOOK:
            return SOURCE_PRIORITY['books']
        elif metadata.doc_type == DocType.RESEARCH:
            return SOURCE_PRIORITY['arxiv_papers']
        elif metadata.doc_type == DocType.TUTORIAL:
            return SOURCE_PRIORITY['official_tutorials']
        
        return SOURCE_PRIORITY['other']
    
    def _detect_deprecation(self, content: str) -> tuple:
        """Detect if content contains deprecation information"""
        from config.constants import DEPRECATION_KEYWORDS
        
        content_lower = content.lower()
        
        for keyword in DEPRECATION_KEYWORDS:
            if keyword.lower() in content_lower:
                # Try to extract replacement
                replacement = self._extract_replacement(content)
                return True, replacement
        
        return False, None
    
    def _extract_replacement(self, content: str) -> Optional[str]:
        """Extract replacement function name from deprecation notice"""
        patterns = [
            r'replaced by (\w+)',
            r'use (\w+) instead',
            r'migrate to (\w+)',
            r'replacement: (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _simple_chunk(self, document: Document) -> List[Document]:
        """Simple chunking for non-code, non-research documents"""
        content = document.content
        chunk_size = 600
        chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            chunk_metadata = DocumentMetadata(**document.metadata.dict())
            chunk_metadata.chunk_id = f"{document.id}_chunk_{i}"
            
            chunk_doc = Document(
                id=chunk_metadata.chunk_id,
                content=chunk_content,
                metadata=chunk_metadata
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks to ensure quality"""
        valid_chunks = []
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.content.strip():
                continue
            
            # Ensure minimum content length
            if len(chunk.content.strip()) < 10:
                continue
            
            # Update weights
            if chunk.metadata.code_weight == 0:
                chunk.metadata.code_weight = self._calculate_code_weight(chunk.content)
            
            valid_chunks.append(chunk)
        
        return valid_chunks
    
    def _generate_doc_id(self, content: str, metadata: dict) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        framework = metadata.get('framework', 'unknown')
        doc_type = metadata.get('doc_type', 'unknown')
        return f"{framework}_{doc_type}_{content_hash}"
    
    def _calculate_code_weight(self, content: str) -> float:
        """Calculate code weight for a chunk"""
        # Simple implementation - can be enhanced
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ']
        lines = content.split('\n')
        code_lines = 0
        
        for line in lines:
            if any(indicator in line for indicator in code_indicators):
                code_lines += 1
        
        return min(1.0, code_lines / max(1, len(lines)))