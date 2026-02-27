import re
from typing import List, Dict, Any
from loguru import logger

from config.settings import Settings
from models.document import Document, DocumentMetadata

class ResearchChunker:
    """Specialized chunker for research papers and books"""
    
    def __init__(self, chunk_size: int = Settings.CHUNK_SIZE):
        self.chunk_size = chunk_size
        
        # Research section patterns
        self.section_patterns = {
            "definition": r"(?i)(?:definition|def\.|define)\s+\d*\.?\s*",
            "theorem": r"(?i)(?:theorem|thm\.|lemma|corollary|proposition)\s+\d*\.?\s*",
            "algorithm": r"(?i)(?:algorithm|algo\.)\s+\d*\.?\s*",
            "proof": r"(?i)(?:proof|pf\.)\s*",
            "complexity": r"(?i)(?:complexity|runtime|time\s+complexity|space\s+complexity)"
        }
    
    def chunk_research(self, document: Document) -> List[Document]:
        """Split research paper into semantic sections"""
        try:
            content = document.content
            chunks = []
            
            # Try to split by sections first
            sections = self._extract_sections(content)
            
            if sections:
                for i, (section_type, section_content) in enumerate(sections):
                    # Further split if section is too long
                    if len(section_content) > self.chunk_size:
                        sub_chunks = self._split_long_section(section_content)
                        for j, sub_chunk in enumerate(sub_chunks):
                            chunk_doc = self._create_chunk_doc(
                                document, sub_chunk, section_type, f"{i}_{j}"
                            )
                            chunks.append(chunk_doc)
                    else:
                        chunk_doc = self._create_chunk_doc(
                            document, section_content, section_type, str(i)
                        )
                        chunks.append(chunk_doc)
            else:
                # Fallback to regular text splitting
                chunks = self._split_by_length(document)
            
            # Update research weights
            for chunk in chunks:
                chunk.metadata.research_weight = self._calculate_research_weight(
                    chunk.content, chunk.metadata.doc_type
                )
            
            logger.info(f"Split research doc into {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking research: {e}")
            return [document]
    
    def _extract_sections(self, content: str) -> List[tuple]:
        """Extract semantic sections from research content"""
        sections = []
        lines = content.split('\n')
        current_section = []
        current_type = "text"
        
        for line in lines:
            # Check for section headers
            for section_type, pattern in self.section_patterns.items():
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_section:
                        sections.append((current_type, '\n'.join(current_section)))
                    
                    # Start new section
                    current_section = [line]
                    current_type = section_type
                    break
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append((current_type, '\n'.join(current_section)))
        
        return sections
    
    def _split_long_section(self, content: str) -> List[str]:
        """Split long section into smaller chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_by_length(self, document: Document) -> List[Document]:
        """Fallback: split by character length"""
        content = document.content
        chunks = []
        
        for i in range(0, len(content), self.chunk_size):
            chunk_content = content[i:i + self.chunk_size]
            chunk_doc = self._create_chunk_doc(
                document, chunk_content, "text", str(i)
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def _create_chunk_doc(self, parent_doc: Document, content: str, 
                          section_type: str, suffix: str) -> Document:
        """Create a chunk document from parent"""
        metadata = DocumentMetadata(**parent_doc.metadata.dict())
        metadata.chunk_id = f"{parent_doc.id}_section_{suffix}"
        metadata.doc_type = section_type if section_type != "text" else "research"
        
        return Document(
            id=metadata.chunk_id,
            content=content,
            metadata=metadata
        )
    
    def _calculate_research_weight(self, content: str, doc_type: str) -> float:
        """Calculate research weight based on content type"""
        # Base weight by document type
        type_weights = {
            "definition": 0.9,
            "theorem": 1.0,
            "algorithm": 0.9,
            "proof": 0.8,
            "complexity": 0.9,
            "research": 0.7,
            "text": 0.5
        }
        
        base_weight = type_weights.get(doc_type, 0.5)
        
        # Boost based on research indicators
        research_indicators = [
            r'\[\d+\]',           # citations
            r'et al\.',             # academic references
            r'arXiv:',              # arXiv references
            r'Theorem\s+\d+',       # theorem references
            r'Lemma\s+\d+',         # lemma references
            r'Proof\.',             # proof indicator
            r'Qubit',               # quantum terms
            r'Hilbert\s+space',      # quantum math
            r'unitary',              # quantum operations
            r'entanglement',         # quantum concept
            r'superposition'         # quantum concept
        ]
        
        boost = 0
        for pattern in research_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                boost += 0.05
        
        return min(1.0, base_weight + boost)