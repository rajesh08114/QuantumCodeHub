import re
from typing import Optional, List, Dict, Any
from loguru import logger
import spacy
from enum import Enum

from config.constants import Intent, Framework

class IntentClassifier:
    """Classifies user queries into intents"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found, downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Intent patterns
        self.intent_patterns = {
            Intent.CODE_IMPLEMENTATION: [
                r"how to (?:implement|write|create|build|code)",
                r"example of",
                r"sample code",
                r"implementation of",
                r"code for",
                r"using (\w+) in",
                r"show me how to",
                r"demonstrate"
            ],
            Intent.ERROR_FIX: [
                r"error",
                r"exception",
                r"fail",
                r"not working",
                r"issue",
                r"bug",
                r"fix",
                r"problem",
                r"doesn't work",
                r"does not work"
            ],
            Intent.API_LOOKUP: [
                r"what is",
                r"documentation for",
                r"api (?:for|reference)",
                r"function",
                r"class",
                r"method",
                r"parameter",
                r"argument",
                r"return (?:type|value)",
                r"signature"
            ],
            Intent.RESEARCH_THEORY: [
                r"algorithm",
                r"complexity",
                r"proof",
                r"theorem",
                r"lemma",
                r"theory",
                r"research",
                r"paper",
                r"citation",
                r"arxiv",
                r"quantum (?:algorithm|circuit|gate|operation)"
            ],
            Intent.MIGRATION: [
                r"migrate",
                r"upgrade",
                r"downgrade",
                r"version",
                r"deprecated",
                r"replace",
                r"instead of",
                r"change from",
                r"update from"
            ],
            Intent.PERFORMANCE_OPTIMIZATION: [
                r"performance",
                r"optimiz",
                r"fast",
                r"speed",
                r"efficient",
                r"throughput",
                r"latency",
                r"overhead",
                r"resource",
                r"memory",
                r"time"
            ]
        }
    
    def classify(self, query: str, context: Optional[Dict] = None) -> Intent:
        """Classify query intent"""
        try:
            # Process query with spaCy
            doc = self.nlp(query.lower())
            
            # Check for explicit intent indicators
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query.lower()):
                        logger.info(f"Classified intent as {intent} (pattern match)")
                        return intent
            
            # Use POS tagging and dependency parsing for better classification
            # Check for question words
            question_words = {"what", "how", "why", "when", "where", "which"}
            if any(token.text in question_words for token in doc):
                # Check if asking about code
                code_indicators = {"function", "method", "class", "code", "implement"}
                if any(token.text in code_indicators for token in doc):
                    if "how" in [token.text for token in doc]:
                        return Intent.CODE_IMPLEMENTATION
                    return Intent.API_LOOKUP
            
            # Check for research terms
            research_terms = {"algorithm", "complexity", "theorem", "proof", "paper"}
            if any(token.lemma_ in research_terms for token in doc):
                return Intent.RESEARCH_THEORY
            
            # Check for error terms
            error_terms = {"error", "exception", "fail", "issue", "bug"}
            if any(token.lemma_ in error_terms for token in doc):
                return Intent.ERROR_FIX
            
            # Check for performance terms
            performance_terms = {"performance", "optimize", "speed", "fast", "efficient"}
            if any(token.lemma_ in performance_terms for token in doc):
                return Intent.PERFORMANCE_OPTIMIZATION
            
            # Default to API lookup if no clear intent
            logger.info("No clear intent detected, defaulting to API_LOOKUP")
            return Intent.API_LOOKUP
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return Intent.API_LOOKUP
    
    def extract_framework(self, query: str) -> Optional[Framework]:
        """Extract framework from query"""
        query_lower = query.lower()
        
        framework_patterns = {
            Framework.QISKIT: [r"qiskit", r"ibm"],
            Framework.PENNYLANE: [r"pennylane", r"xanadu"],
            Framework.CIRQ: [r"cirq", r"google"],
            Framework.TORCHQUANTUM: [r"torchquantum", r"pytorch quantum"]
        }
        
        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return framework
        
        return None
    
    def extract_version(self, query: str) -> Optional[str]:
        """Extract version number from query"""
        version_pattern = r'(\d+\.\d+\.\d+)'
        match = re.search(version_pattern, query)
        
        if match:
            return match.group(1)
        
        return None
    
    def get_intent_boost(self, intent: Intent) -> Dict[str, float]:
        """Get collection boost weights for intent"""
        from config.constants import INTENT_BIAS
        
        return INTENT_BIAS.get(intent, {})