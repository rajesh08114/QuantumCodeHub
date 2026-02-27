import pytest
from unittest.mock import Mock, patch

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.intent_classifier import IntentClassifier
from retrieval.version_filter import VersionFilter
from models.document import RetrievalQuery
from config.constants import Framework, Intent

class TestIntentClassifier:
    def setup_method(self):
        self.classifier = IntentClassifier()
    
    def test_code_implementation_intent(self):
        query = "how to implement a quantum circuit in qiskit"
        intent = self.classifier.classify(query)
        assert intent == Intent.CODE_IMPLEMENTATION
    
    def test_api_lookup_intent(self):
        query = "what is the QuantumCircuit class"
        intent = self.classifier.classify(query)
        assert intent == Intent.API_LOOKUP
    
    def test_research_intent(self):
        query = "explain the quantum fourier transform algorithm complexity"
        intent = self.classifier.classify(query)
        assert intent == Intent.RESEARCH_THEORY
    
    def test_error_fix_intent(self):
        query = "getting error when running circuit: QiskitError"
        intent = self.classifier.classify(query)
        assert intent == Intent.ERROR_FIX
    
    def test_extract_framework(self):
        query = "how to use pennylane for variational circuits"
        framework = self.classifier.extract_framework(query)
        assert framework == Framework.PENNYLANE
    
    def test_extract_version(self):
        query = "using qiskit 1.2.0 features"
        version = self.classifier.extract_version(query)
        assert version == "1.2.0"

class TestVersionFilter:
    def setup_method(self):
        self.filter = VersionFilter()
    
    def test_version_filter_with_requested(self):
        filter_dict = self.filter.get_version_filter(
            Framework.QISKIT, "1.2.0"
        )
        assert filter_dict == {"version": {"$eq": "1.2.0"}}
    
    def test_version_filter_without_requested(self):
        filter_dict = self.filter.get_version_filter(Framework.QISKIT)
        assert "$gte" in filter_dict["version"]
        assert "$lte" in filter_dict["version"]
    
    def test_unsupported_version(self):
        filter_dict = self.filter.get_version_filter(
            Framework.QISKIT, "0.5.0"
        )
        # Should fall back to supported range
        assert "$gte" in filter_dict["version"]
        assert filter_dict["version"]["$gte"] == "1.0.0"

class TestHybridRetriever:
    def setup_method(self):
        self.retriever = HybridRetriever()
    
    @pytest.mark.asyncio
    async def test_retrieve_basic_query(self):
        query = RetrievalQuery(
            query="how to create a bell state in qiskit",
            framework=Framework.QISKIT
        )
        
        result = self.retriever.retrieve(query)
        
        assert result.framework == Framework.QISKIT
        assert result.intent is not None
        assert isinstance(result.documents, list)
    
    def test_deprecation_detection(self):
        # Mock deprecation detector
        self.retriever.deprecation_detector.check_deprecations = Mock(
            return_value=(True, {"replacement": "new_function"})
        )
        
        query = RetrievalQuery(
            query="using old_deprecated_function",
            framework=Framework.QISKIT
        )
        
        result = self.retriever.retrieve(query)
        
        assert result.deprecation_detected == True
        assert result.deprecation_info is not None