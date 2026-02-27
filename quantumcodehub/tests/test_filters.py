import pytest
from unittest.mock import Mock, patch

from retrieval.scoring_engine import ScoringEngine
from config.constants import Intent

class TestScoringEngine:
    def setup_method(self):
        self.engine = ScoringEngine()
        
        # Sample documents
        self.documents = [
            {
                'id': 'doc1',
                'content': 'def quantum_circuit(): pass',
                'metadata': {
                    'source_priority': 5,
                    'version': '1.2.0',
                    'doc_type': 'api',
                    'code_weight': 0.9,
                    'research_weight': 0.1
                },
                'distance': 0.2,
                'collection': 'api_docs'
            },
            {
                'id': 'doc2',
                'content': 'research paper on quantum algorithms',
                'metadata': {
                    'source_priority': 4,
                    'version': '1.1.0',
                    'doc_type': 'research',
                    'code_weight': 0.1,
                    'research_weight': 0.9
                },
                'distance': 0.5,
                'collection': 'research'
            }
        ]
    
    def test_score_documents_code_intent(self):
        scored = self.engine.score_documents(
            self.documents.copy(),
            "quantum circuit implementation",
            Intent.CODE_IMPLEMENTATION
        )
        
        # First doc should score higher for code intent
        assert scored[0]['id'] == 'doc1'
        assert 'score' in scored[0]
        assert 'score_components' in scored[0]
    
    def test_score_documents_research_intent(self):
        scored = self.engine.score_documents(
            self.documents.copy(),
            "quantum algorithm complexity",
            Intent.RESEARCH_THEORY
        )
        
        # Second doc should score higher for research intent
        assert scored[0]['id'] == 'doc2'
    
    def test_version_match_scoring(self):
        scores = self.engine._calculate_version_match_scores(
            self.documents,
            "1.2.0",
            None
        )
        
        assert len(scores) == 2
        assert scores[0] > scores[1]  # doc1 version matches exactly
    
    def test_source_priority_scoring(self):
        scores = self.engine._calculate_source_priority_scores(self.documents)
        
        assert scores[0] == 1.0  # priority 5 -> max score
        assert scores[1] == 0.75  # priority 4 -> 0.75 after normalization