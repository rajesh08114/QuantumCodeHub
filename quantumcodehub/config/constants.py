from enum import Enum
from typing import Dict, List, Tuple

class Framework(Enum):
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    TORCHQUANTUM = "torchquantum"

class DocType(Enum):
    API = "api"
    TUTORIAL = "tutorial"
    RESEARCH = "research"
    BOOK = "book"
    RELEASE = "release"
    DEPRECATION = "deprecation"

class Intent(Enum):
    CODE_IMPLEMENTATION = "code_implementation"
    ERROR_FIX = "error_fix"
    API_LOOKUP = "api_lookup"
    RESEARCH_THEORY = "research_theory"
    MIGRATION = "migration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

# Supported version ranges
SUPPORTED_VERSIONS = {
    Framework.QISKIT: ("1.0.0", "1.2.0"),
    Framework.PENNYLANE: ("0.35.0", "0.36.0"),
    Framework.CIRQ: ("1.3.0", "1.4.0"),
    Framework.TORCHQUANTUM: ("0.2.0", "0.3.0"),
}

# Collection names
COLLECTIONS = {
    "api_docs": "quantum_api_docs",
    "release_notes": "quantum_release_notes",
    "books": "quantum_books",
    "research": "quantum_research",
    "code_examples": "quantum_code_examples",
    "deprecations": "quantum_deprecations"
}

# Source priority (1-5, 5 is highest)
SOURCE_PRIORITY = {
    "official_docs": 5,
    "official_tutorials": 5,
    "arxiv_papers": 4,
    "books": 4,
    "community_tutorials": 3,
    "blog_posts": 2,
    "stack_overflow": 2,
    "other": 1
}

# Intent to retrieval bias mapping
INTENT_BIAS = {
    Intent.CODE_IMPLEMENTATION: {
        "collection_boost": {"code_examples": 0.4, "api_docs": 0.3},
        "code_weight_min": 0.7
    },
    Intent.ERROR_FIX: {
        "collection_boost": {"release_notes": 0.3, "deprecations": 0.4, "api_docs": 0.2},
        "code_weight_min": 0.3
    },
    Intent.API_LOOKUP: {
        "collection_boost": {"api_docs": 0.7},
        "code_weight_min": 0.2
    },
    Intent.RESEARCH_THEORY: {
        "collection_boost": {"research": 0.5, "books": 0.3},
        "research_weight_min": 0.8
    },
    Intent.MIGRATION: {
        "collection_boost": {"release_notes": 0.4, "deprecations": 0.4, "api_docs": 0.2},
        "code_weight_min": 0.4
    },
    Intent.PERFORMANCE_OPTIMIZATION: {
        "collection_boost": {"research": 0.4, "api_docs": 0.3},
        "research_weight_min": 0.6
    }
}

# Scoring weights (base weights, modified by intent)
BASE_SCORING_WEIGHTS = {
    "dense": 0.40,
    "bm25": 0.25,
    "source_priority": 0.15,
    "version_match": 0.10,
    "intent": 0.10
}

# Deprecation keywords
DEPRECATION_KEYWORDS = [
    "deprecated", "will be removed", "no longer supported",
    "replaced by", "use instead", "obsolete", "legacy",
    "warning: .* deprecated", "deprecationwarning"
]