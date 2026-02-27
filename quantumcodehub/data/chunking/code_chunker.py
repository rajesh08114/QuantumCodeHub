import re
from typing import List, Generator
# from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import Settings
from models.document import Document, DocumentMetadata

class CodeChunker:
    """Intelligent code chunker for quantum frameworks"""
    
    def __init__(self, chunk_size: int = Settings.CHUNK_SIZE, 
                 chunk_overlap: int = Settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Framework-specific separators
        self.framework_separators = {
            "qiskit": [
                r"\nclass\s+",
                r"\ndef\s+",
                r"@\w+",  # decorators
                r"\n\n",
                r"\n",
                r" "
            ],
            "pennylane": [
                r"\nclass\s+",
                r"\ndef\s+",
                r"@qml\.\w+",  # pennylane decorators
                r"\n\n",
                r"\n",
                r" "
            ],
            "cirq": [
                r"\nclass\s+",
                r"\ndef\s+",
                r"@cirq\.\w+",
                r"\n\n",
                r"\n",
                r" "
            ],
            "torchquantum": [
                r"\nclass\s+",
                r"\ndef\s+",
                r"@tq\.\w+",
                r"\n\n",
                r"\n",
                r" "
            ]
        }
        
        # Default separators
        self.default_separators = ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
    
    def chunk_code(self, document: Document) -> List[Document]:
        """Split code document into function/class-level chunks"""
        try:
            framework = document.metadata.framework.value
            separators = self.framework_separators.get(
                framework, self.default_separators
            )
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=True
            )
            
            # Split text
            chunks = text_splitter.split_text(document.content)
            
            # Create document chunks with updated metadata
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                # Create new metadata (copy from parent)
                chunk_metadata = DocumentMetadata(**document.metadata.dict())
                chunk_metadata.chunk_id = f"{document.id}_chunk_{i}"
                
                # Update code weight based on chunk content
                chunk_metadata.code_weight = self._calculate_code_weight(chunk)
                
                chunk_doc = Document(
                    id=chunk_metadata.chunk_id,
                    content=chunk,
                    metadata=chunk_metadata
                )
                chunk_docs.append(chunk_doc)
            
            logger.info(f"Split document into {len(chunk_docs)} chunks")
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error chunking code: {e}")
            return [document]  # Return original on error
    
    def _calculate_code_weight(self, chunk: str) -> float:
        """Calculate code weight based on code-to-text ratio"""
        # Count code indicators
        code_indicators = [
            r'def\s+\w+\s*\(',  # function definitions
            r'class\s+\w+',       # class definitions
            r'=\s*\w+\(',         # function calls
            r'import\s+',          # imports
            r'from\s+\w+\s+import', # from imports
            r'return\s+',          # return statements
            r'if\s+.*:',           # if statements
            r'for\s+.*:',          # for loops
            r'while\s+.*:',        # while loops
            r'@\w+'                # decorators
        ]
        
        # Count lines that look like code
        lines = chunk.split('\n')
        code_lines = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if line contains code indicators
            for pattern in code_indicators:
                if re.search(pattern, line):
                    code_lines += 1
                    break
            else:
                # If no indicators but line has code-like syntax
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=\(\)\[\]\{\}]', line):
                    code_lines += 1
        
        # Calculate weight
        total_non_empty = sum(1 for line in lines if line.strip())
        if total_non_empty > 0:
            return min(1.0, code_lines / total_non_empty)
        return 0.0