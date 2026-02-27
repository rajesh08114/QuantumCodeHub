from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

from config.constants import BASE_SCORING_WEIGHTS, INTENT_BIAS, Intent, Framework

class ScoringEngine:
    """Handles dynamic scoring of retrieved documents"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.base_weights = BASE_SCORING_WEIGHTS
    
    def score_documents(self, 
                       documents: List[Dict], 
                       query: str,
                       intent: Intent,
                       framework: Optional[Framework] = None,
                       query_version: Optional[str] = None) -> List[Dict]:
        """Score and rerank documents"""
        if not documents:
            return []
        
        try:
            # Calculate individual scores
            dense_scores = self._calculate_dense_scores(documents)
            bm25_scores = self._calculate_bm25_scores(documents, query)
            source_priority_scores = self._calculate_source_priority_scores(documents)
            version_match_scores = self._calculate_version_match_scores(
                documents, query_version, framework
            )
            intent_scores = self._calculate_intent_scores(documents, intent)
            
            # Get intent-specific weights
            weights = self._get_intent_weights(intent)
            
            # Calculate final scores
            final_scores = []
            for i in range(len(documents)):
                score = (
                    weights['dense'] * dense_scores[i] +
                    weights['bm25'] * bm25_scores[i] +
                    weights['source_priority'] * source_priority_scores[i] +
                    weights['version_match'] * version_match_scores[i] +
                    weights['intent'] * intent_scores[i]
                )
                final_scores.append(score)
            
            # Normalize scores
            final_scores = self._normalize_scores(final_scores)
            
            # Add scores to documents
            for i, doc in enumerate(documents):
                doc['score'] = final_scores[i]
                doc['score_components'] = {
                    'dense': dense_scores[i],
                    'bm25': bm25_scores[i],
                    'source_priority': source_priority_scores[i],
                    'version_match': version_match_scores[i],
                    'intent': intent_scores[i]
                }
            
            # Sort by score
            documents.sort(key=lambda x: x['score'], reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in score_documents: {e}")
            return documents
    
    def _calculate_dense_scores(self, documents: List[Dict]) -> List[float]:
        """Calculate scores based on embedding distance"""
        scores = []
        for doc in documents:
            # Convert distance to similarity (lower distance = higher score)
            distance = doc.get('distance', 1.0)
            similarity = 1.0 - min(distance, 1.0)
            scores.append(similarity)
        
        return self._normalize_scores(scores)
    
    def _calculate_bm25_scores(self, documents: List[Dict], query: str) -> List[float]:
        """Calculate BM25 lexical scores"""
        # Simplified BM25 simulation based on keyword overlap
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in documents:
            content = doc.get('content', '').lower()
            content_terms = set(content.split())
            
            # Calculate Jaccard similarity
            intersection = len(query_terms & content_terms)
            union = len(query_terms | content_terms)
            
            if union > 0:
                score = intersection / union
            else:
                score = 0
            
            scores.append(score)
        
        return self._normalize_scores(scores)
    
    def _calculate_source_priority_scores(self, documents: List[Dict]) -> List[float]:
        """Calculate scores based on source priority"""
        scores = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            priority = metadata.get('source_priority', 1)
            # Normalize to 0-1 range (priority 1-5)
            score = (priority - 1) / 4
            scores.append(score)
        
        return self._normalize_scores(scores)
    
    def _calculate_version_match_scores(self, 
                                       documents: List[Dict],
                                       query_version: Optional[str],
                                       framework: Optional[Framework]) -> List[float]:
        """Calculate version match scores"""
        from retrieval.version_filter import VersionFilter
        
        version_filter = VersionFilter()
        scores = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_version = metadata.get('version', '0.0.0')
            
            score = version_filter.calculate_version_match_score(
                doc_version, query_version, framework
            )
            scores.append(score)
        
        return self._normalize_scores(scores)
    
    def _calculate_intent_scores(self, documents: List[Dict], intent: Intent) -> List[float]:
        """Calculate scores based on intent relevance"""
        scores = []
        intent_config = INTENT_BIAS.get(intent, {})
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('doc_type')
            
            # Base score from document type
            if doc_type == 'api':
                score = 0.7
            elif doc_type == 'code_examples':
                score = 0.9
            elif doc_type == 'research':
                score = 0.8
            elif doc_type == 'tutorial':
                score = 0.6
            else:
                score = 0.5
            
            # Apply intent-specific boosts
            collection_boost = intent_config.get('collection_boost', {})
            for collection, boost in collection_boost.items():
                if collection in doc.get('collection', ''):
                    score += boost
            
            # Apply code weight threshold
            code_weight_min = intent_config.get('code_weight_min', 0)
            if metadata.get('code_weight', 0) >= code_weight_min:
                score += 0.1
            
            # Apply research weight threshold
            research_weight_min = intent_config.get('research_weight_min', 0)
            if metadata.get('research_weight', 0) >= research_weight_min:
                score += 0.1
            
            scores.append(min(1.0, score))
        
        return self._normalize_scores(scores)
    
    def _get_intent_weights(self, intent: Intent) -> Dict[str, float]:
        """Get weights adjusted for intent"""
        weights = self.base_weights.copy()
        
        # Adjust weights based on intent
        if intent == Intent.CODE_IMPLEMENTATION:
            weights['dense'] = 0.35
            weights['bm25'] = 0.30
            weights['source_priority'] = 0.15
            weights['version_match'] = 0.10
            weights['intent'] = 0.10
        elif intent == Intent.RESEARCH_THEORY:
            weights['dense'] = 0.30
            weights['bm25'] = 0.20
            weights['source_priority'] = 0.25
            weights['version_match'] = 0.05
            weights['intent'] = 0.20
        elif intent == Intent.ERROR_FIX:
            weights['dense'] = 0.30
            weights['bm25'] = 0.35
            weights['source_priority'] = 0.15
            weights['version_match'] = 0.10
            weights['intent'] = 0.10
        
        return weights
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score == 0:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]