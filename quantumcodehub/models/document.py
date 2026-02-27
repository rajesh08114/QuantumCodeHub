from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from config.constants import Intent
# import Intent

from config.constants import Framework, DocType

class DocumentMetadata(BaseModel):
    """Metadata model for all documents"""
    framework: Framework
    version: str
    doc_type: DocType
    is_deprecated: bool = False
    replacement: Optional[str] = None
    code_weight: float = Field(ge=0.0, le=1.0, default=0.0)
    research_weight: float = Field(ge=0.0, le=1.0, default=0.0)
    source_priority: int = Field(ge=1, le=5, default=1)
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    authors: Optional[List[str]] = None
    published_date: Optional[datetime] = None
    arxiv_id: Optional[str] = None
    chunk_id: Optional[str] = None
    parent_doc_id: Optional[str] = None
    
    @validator('version')
    def validate_version_format(cls, v):
        # Basic semantic version validation
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(f'Invalid version format: {v}. Expected format: X.Y.Z')
        return v

class Document(BaseModel):
    """Complete document model"""
    id: Optional[str] = None
    content: str
    metadata: DocumentMetadata
    embeddings: Optional[List[float]] = None
    
class RetrievalQuery(BaseModel):
    """User query model"""
    query: str
    framework: Optional[Framework] = None
    version: Optional[str] = None
    intent: Optional[Intent] = None
    top_k: int = 10
    include_deprecated: bool = False
    
class RetrievalResult(BaseModel):
    """Retrieval result model"""
    framework: Optional[Framework]
    version: Optional[str]
    intent: Intent
    documents: List[Dict[str, Any]]
    deprecation_detected: bool = False
    deprecation_info: Optional[Dict[str, Any]] = None
    query_time_ms: float
    total_results: int