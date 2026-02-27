from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from retrieval.chroma_manager import ChromaManager
from retrieval.intent_classifier import IntentClassifier
from retrieval.version_filter import VersionFilter
from retrieval.deprecation_detector import DeprecationDetector
from retrieval.scoring_engine import ScoringEngine
from config.constants import COLLECTIONS, Intent, Framework
from models.document import RetrievalQuery, RetrievalResult

class HybridRetriever:
    """Main retrieval engine combining all components"""
    
    def __init__(self):
        self.chroma_manager = ChromaManager()
        self.intent_classifier = IntentClassifier()
        self.version_filter = VersionFilter()
        self.deprecation_detector = DeprecationDetector(self.chroma_manager)
        self.scoring_engine = ScoringEngine()
        
        logger.info("HybridRetriever initialized")
    
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Main retrieval method"""
        import time
        start_time = time.time()
        
        try:
            # Extract framework and version from query if not provided
            if not query.framework:
                query.framework = self.intent_classifier.extract_framework(query.query)
            
            if not query.version:
                query.version = self.intent_classifier.extract_version(query.query)
            
            # Classify intent if not provided
            if not query.intent:
                query.intent = self.intent_classifier.classify(query.query)
            
            # Check for deprecations first (if not explicitly excluded)
            deprecation_detected = False
            deprecation_info = None
            
            if not query.include_deprecated and query.framework:
                deprecation_detected, deprecation_info = self.deprecation_detector.check_deprecations(
                    query.query, query.framework.value
                )
            
            # Determine which collections to search
            collections_to_search = self._get_collections_for_intent(query.intent)
            
            # Build metadata filter
            metadata_filter = self._build_metadata_filter(
                query.framework, 
                query.version,
                query.include_deprecated
            )
            
            # Perform initial retrieval
            initial_results = self.chroma_manager.query(
                collection_names=collections_to_search,
                query_text=query.query,
                n_results=query.top_k * 2,  # Get more for reranking
                where=metadata_filter
            )
            
            if not initial_results:
                # Try without version filter if no results
                if metadata_filter and 'version' in metadata_filter:
                    logger.info("No results with version filter, trying without...")
                    metadata_filter.pop('version', None)
                    initial_results = self.chroma_manager.query(
                        collection_names=collections_to_search,
                        query_text=query.query,
                        n_results=query.top_k * 2,
                        where=metadata_filter
                    )
            
            # Score and rerank results
            scored_results = self.scoring_engine.score_documents(
                documents=initial_results,
                query=query.query,
                intent=query.intent,
                framework=query.framework,
                query_version=query.version
            )
            
            # Apply hallucination guard
            validated_results = self._apply_hallucination_guard(
                scored_results, query.query
            )
            
            # Prepare final results
            final_results = validated_results[:query.top_k]
            
            # Calculate query time
            query_time_ms = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                framework=query.framework,
                version=query.version,
                intent=query.intent,
                documents=final_results,
                deprecation_detected=deprecation_detected,
                deprecation_info=deprecation_info,
                query_time_ms=query_time_ms,
                total_results=len(final_results)
            )
            
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return RetrievalResult(
                framework=query.framework,
                version=query.version,
                intent=query.intent if query.intent else Intent.API_LOOKUP,
                documents=[],
                query_time_ms=(time.time() - start_time) * 1000,
                total_results=0
            )
    
    def _get_collections_for_intent(self, intent: Intent) -> List[str]:
        """Determine which collections to search based on intent"""
        intent_collections = {
            Intent.CODE_IMPLEMENTATION: [
                COLLECTIONS["code_examples"],
                COLLECTIONS["api_docs"],
                COLLECTIONS["tutorials"]
            ],
            Intent.ERROR_FIX: [
                COLLECTIONS["release_notes"],
                COLLECTIONS["deprecations"],
                COLLECTIONS["api_docs"]
            ],
            Intent.API_LOOKUP: [
                COLLECTIONS["api_docs"],
                COLLECTIONS["tutorials"]
            ],
            Intent.RESEARCH_THEORY: [
                COLLECTIONS["research"],
                COLLECTIONS["books"],
                COLLECTIONS["api_docs"]
            ],
            Intent.MIGRATION: [
                COLLECTIONS["release_notes"],
                COLLECTIONS["deprecations"],
                COLLECTIONS["api_docs"]
            ],
            Intent.PERFORMANCE_OPTIMIZATION: [
                COLLECTIONS["research"],
                COLLECTIONS["api_docs"],
                COLLECTIONS["code_examples"]
            ]
        }
        
        return intent_collections.get(intent, [COLLECTIONS["api_docs"]])
    
    def _build_metadata_filter(self, 
                              framework: Optional[Framework], 
                              version: Optional[str],
                              include_deprecated: bool) -> Optional[Dict]:
        """Build metadata filter for ChromaDB query"""
        filter_dict = {}
        
        if framework:
            filter_dict["framework"] = framework.value
            
            # Add version filter
            version_filter = self.version_filter.get_version_filter(framework, version)
            if version_filter:
                filter_dict.update(version_filter)
        
        if not include_deprecated:
            filter_dict["is_deprecated"] = False
        
        return filter_dict if filter_dict else None
    
    def _apply_hallucination_guard(self, 
                                   documents: List[Dict], 
                                   query: str) -> List[Dict]:
        """Apply hallucination prevention measures"""
        from config.settings import Settings
        
        validated_docs = []
        
        # Check for keyword overlap
        query_keywords = set(query.lower().split())
        official_sources_count = 0
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            content = doc.get('content', '').lower()
            
            # Check keyword overlap
            content_keywords = set(content.split())
            overlap = len(query_keywords & content_keywords) / max(1, len(query_keywords))
            
            if overlap < Settings.MIN_KEYWORD_OVERLAP:
                continue
            
            # Check for official sources
            if metadata.get('source_priority', 0) >= 4:
                official_sources_count += 1
            
            validated_docs.append(doc)
        
        # Ensure at least one official source
        if official_sources_count == 0 and validated_docs:
            # Try to find official sources
            for doc in validated_docs:
                if doc.get('metadata', {}).get('source_priority', 0) >= 4:
                    break
            else:
                # No official sources, but we have results - keep them but log warning
                logger.warning("No official sources found in results")
        
        return validated_docs
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        for collection_name in COLLECTIONS.values():
            stats[collection_name] = self.chroma_manager.get_collection_stats(collection_name)
        return stats