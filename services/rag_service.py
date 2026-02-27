"""
RAG pipeline service backed by ChromaDB.
Optimized retrieval uses multi-query recall + hybrid reranking + diversity filtering.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import logging
import re
import time
from urllib.parse import urlparse

import chromadb

from core.config import settings
from ml.embeddings import embedding_service

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation Service.

    Uses the shared Chroma collection created by `Rag_pipeline`.
    """

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "using",
        "with",
    }
    _LATEST_VERSION_LABELS = {"latest", "stable", "current", "main", "master", "head"}
    _FRAMEWORKS_WITH_VERSIONING = {
        "qiskit",
        "pennylane",
        "cirq",
        "torchquantum",
        "tensorflow_quantum",
    }
    _IBM_API_DATASET = "ibm_quantum_api_docs"

    def __init__(self):
        self.top_k = settings.RAG_TOP_K or 5
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.persist_dir = self._resolve_persist_dir(settings.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self._safe_get_collection(self.collection_name)
        self.traffic_profile = (settings.RAG_TRAFFIC_PROFILE or "auto").strip().lower()

        # Retrieval tuning
        self.fetch_multiplier = max(3, int(settings.RAG_FETCH_MULTIPLIER or 12))
        self.max_fetch_results = max(20, int(settings.RAG_MAX_FETCH_RESULTS or 200))
        self.max_query_variants = max(1, min(int(settings.RAG_QUERY_VARIANTS or 3), 5))
        self.max_docs_per_source = max(1, int(settings.RAG_MAX_DOCS_PER_SOURCE or 2))
        self.max_doc_chars = max(200, int(settings.RAG_MAX_DOC_CHARS or 1200))
        self.max_context_chars = max(1000, int(settings.RAG_MAX_CONTEXT_CHARS or 9000))

        # Hybrid scoring weights
        self.semantic_weight = max(0.0, float(settings.RAG_SEMANTIC_WEIGHT or 0.65))
        self.lexical_weight = max(0.0, float(settings.RAG_LEXICAL_WEIGHT or 0.25))
        self.rrf_weight = max(0.0, float(settings.RAG_RRF_WEIGHT or 0.10))
        self.framework_boost = max(0.0, float(settings.RAG_FRAMEWORK_BOOST or 0.08))
        self.rrf_k = max(1, int(settings.RAG_RRF_K or 60))
        self.default_to_latest_version = bool(
            getattr(settings, "RAG_DEFAULT_TO_LATEST_VERSION", True)
        )
        self.latest_version_cache_ttl_seconds = max(
            30, int(getattr(settings, "RAG_LATEST_VERSION_CACHE_TTL_SECONDS", 600) or 600)
        )
        self.legacy_mode_allow_all_versions = bool(
            getattr(settings, "RAG_LEGACY_MODE_ALLOW_ALL_VERSIONS", True)
        )
        self.strict_version_selection = bool(
            getattr(settings, "RAG_STRICT_VERSION_SELECTION", True)
        )
        self._framework_version_cache: Dict[str, Dict] = {}

    def _resolve_persist_dir(self, persist_dir: str) -> Path:
        base_dir = Path(__file__).resolve().parent.parent
        path = Path(persist_dir)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    def _safe_get_collection(self, collection_name: str):
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.warning("Chroma collection '%s' not available: %s", collection_name, e)
            return None

    def _get_or_create_collection(self, collection_name: str):
        collection = self._safe_get_collection(collection_name)
        if collection is not None:
            return collection
        logger.info("Creating missing Chroma collection '%s'", collection_name)
        return self.client.create_collection(name=collection_name)

    def _framework_aliases(self, framework: str) -> List[str]:
        normalized = (framework or "").strip().lower()
        aliases = {
            "qiskit": ["qiskit", "ibm"],
            "pennylane": ["pennylane", "xanadu"],
            "cirq": ["cirq", "quantumlib"],
            "torchquantum": ["torchquantum", "mit-han-lab"],
            "tensorflow_quantum": ["tensorflow_quantum", "tfq", "tensorflow/quantum"],
        }
        return aliases.get(normalized, [normalized] if normalized else [])

    def _looks_like_url(self, value: str) -> bool:
        candidate = (value or "").strip().lower()
        return candidate.startswith("http://") or candidate.startswith("https://")

    def _canonical_source_type(self, value: str) -> str:
        raw = str(value or "").strip().lower()
        if not raw or raw in {"unknown", "<unknown>", "none", "null"}:
            return ""
        mapping = {
            "github_doc": "github_docs",
            "github_docs": "github_docs",
            "official_doc": "official_docs",
            "official_docs": "official_docs",
            "html": "official_docs",
            "markdown": "official_docs",
            "text": "official_docs",
            "pdf": "official_docs",
            "jupyter_notebook": "official_docs",
            "notebook": "official_docs",
            "web": "official_docs",
            "framework_doc": "official_docs",
            "documentation": "official_docs",
            "api": "api_reference",
            "api_docs": "api_reference",
            "api_reference": "api_reference",
            "code": "api_reference",
            "qasm": "api_reference",
            "hardware": "hardware_docs",
            "hardware_docs": "hardware_docs",
        }
        return mapping.get(raw, raw)

    def _infer_source_type(self, metadata: Dict) -> str:
        existing = self._canonical_source_type(metadata.get("source_type", ""))
        if existing:
            return existing

        type_hint = str(metadata.get("type", "")).strip().lower()
        section_hint = str(metadata.get("section", "")).strip().lower()
        source = str(metadata.get("source", "")).strip().lower()
        url = str(metadata.get("url", "")).strip().lower()
        path = str(metadata.get("path", "")).strip().lower()
        haystack = " ".join([type_hint, section_hint, source, url, path])

        if any(token in haystack for token in ["hardware", "backend", "device", "sampler"]):
            return "hardware_docs"
        if any(token in haystack for token in ["/api/", "/stubs/", "api_reference", "reference"]):
            return "api_reference"
        if "github.com" in haystack:
            return "github_docs"
        if any(token in haystack for token in ["tutorial", "guide", "notebook", ".ipynb"]):
            return "official_docs"
        if any(token in haystack for token in ["/docs", "documentation", "readthedocs", "readme"]):
            return "official_docs"
        if type_hint in {"code", "qasm"}:
            return "api_reference"
        if type_hint in {"documentation", "readme", "framework_doc", "markdown", "text"}:
            return "official_docs"
        return "official_docs"

    def _normalize_metadata(self, metadata: Optional[Dict], requested_framework: str = "") -> Dict:
        normalized = dict(metadata or {})
        requested = self._normalize_framework_name(requested_framework)

        framework_value = self._normalize_framework_name(str(normalized.get("framework", "")))
        if not framework_value:
            infer_haystack = " ".join(
                [
                    str(normalized.get("source", "")),
                    str(normalized.get("url", "")),
                    str(normalized.get("path", "")),
                    str(normalized.get("section", "")),
                    str(normalized.get("type", "")),
                ]
            ).lower()
            for candidate in self._FRAMEWORKS_WITH_VERSIONING:
                aliases = self._framework_aliases(candidate)
                if any(alias in infer_haystack for alias in aliases):
                    framework_value = candidate
                    break
        if not framework_value and requested:
            framework_value = requested
        if framework_value:
            normalized["framework"] = framework_value

        version_value = str(normalized.get("version", "")).strip()
        if version_value:
            normalized["version"] = version_value

        url_value = str(normalized.get("url", "")).strip()
        source_value = str(normalized.get("source", "")).strip()
        path_value = str(normalized.get("path", "")).strip()

        if not url_value and self._looks_like_url(source_value):
            url_value = source_value
        if not source_value and url_value:
            source_value = url_value
        if not path_value:
            parsed_path = ""
            if url_value and self._looks_like_url(url_value):
                parsed_path = str(urlparse(url_value).path or "")
            elif source_value and self._looks_like_url(source_value):
                parsed_path = str(urlparse(source_value).path or "")
            if parsed_path:
                path_value = parsed_path

        if url_value:
            normalized["url"] = url_value
        if source_value:
            normalized["source"] = source_value
        if path_value:
            normalized["path"] = path_value

        source_type = self._infer_source_type(normalized)
        normalized["source_type"] = source_type

        section_value = str(normalized.get("section", "")).strip()
        if not section_value:
            section_value = source_type or str(normalized.get("type", "")).strip() or "general"
        normalized["section"] = section_value

        type_value = str(normalized.get("type", "")).strip()
        if not type_value:
            if source_type == "github_docs":
                type_value = "github_doc"
            elif source_type == "api_reference":
                type_value = "api_docs"
            elif source_type == "hardware_docs":
                type_value = "hardware_docs"
            else:
                type_value = "framework_doc"
        normalized["type"] = type_value

        if "layer" not in normalized:
            normalized["layer"] = 3 if source_type == "github_docs" else (2 if source_type == "api_reference" else 1)
        if "priority" not in normalized:
            normalized["priority"] = 3 if source_type in {"api_reference", "hardware_docs"} else 2

        return normalized

    def _metadata_quality_score(self, metadata: Dict) -> float:
        score = 0.0
        if str(metadata.get("source_type", "")).strip():
            score += 0.35
        if str(metadata.get("url", "")).strip() or str(metadata.get("source", "")).strip():
            score += 0.30
        if str(metadata.get("framework", "")).strip():
            score += 0.20
        if str(metadata.get("version", "")).strip():
            score += 0.10
        if str(metadata.get("path", "")).strip():
            score += 0.05
        return min(score, 1.0)

    def _matches_framework(self, metadata: Dict, aliases: List[str]) -> bool:
        metadata = self._normalize_metadata(metadata)
        if not aliases:
            return True

        framework_meta = str(metadata.get("framework", "")).lower()
        if framework_meta and any(alias in framework_meta for alias in aliases):
            return True

        source = str(metadata.get("source", "")).lower()
        path = str(metadata.get("path", "")).lower()
        section = str(metadata.get("section", "")).lower()
        haystack = f"{source} {path} {section}"
        return any(alias in haystack for alias in aliases)

    def _distance_to_similarity(self, distance: float) -> float:
        if distance is None:
            return 0.0
        similarity = 1.0 - float(distance)
        return max(min(similarity, 1.0), -1.0)

    def _query_preview(self, query: str, max_len: int = 120) -> str:
        text = " ".join((query or "").split())
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}..."

    def _contains_code_signals(self, text: str) -> bool:
        content = (text or "").lower()
        patterns = [
            r"```",
            r"\bimport\b",
            r"\bfrom\s+\w+\s+import\b",
            r"\bdef\b",
            r"\bclass\b",
            r"\bquantumcircuit\b",
            r"\bqc\.",
            r"\bqml\.",
            r"\bcirq\.",
            r"\btraceback\b",
            r"\berror\b",
        ]
        return any(re.search(pattern, content) for pattern in patterns)

    def _contains_concept_signals(self, text: str) -> bool:
        content = (text or "").lower()
        signals = [
            "explain",
            "what is",
            "concept",
            "theory",
            "intuition",
            "understand",
            "mathematical",
            "principle",
        ]
        return any(token in content for token in signals)

    def _resolve_active_profile(self, request_source: str, query: str) -> str:
        configured = self.traffic_profile
        allowed = {"auto", "code_heavy", "balanced", "conceptual"}
        if configured not in allowed:
            configured = "auto"

        if configured != "auto":
            return configured

        source = (request_source or "").lower()
        if any(token in source for token in ["/api/code/", "/api/transpile/", "/api/complete/", "/api/fix/"]):
            return "code_heavy"
        if any(token in source for token in ["#generate", "#fix"]):
            return "code_heavy"
        if any(token in source for token in ["/api/explain/", "#explain"]):
            return "conceptual"

        if self._contains_code_signals(query):
            return "code_heavy"
        if self._contains_concept_signals(query):
            return "conceptual"
        return "balanced"

    def _profile_config(self, profile: str) -> Dict:
        # Start from baseline env-tuned values.
        config = {
            "fetch_multiplier": self.fetch_multiplier,
            "max_fetch_results": self.max_fetch_results,
            "max_query_variants": self.max_query_variants,
            "max_docs_per_source": self.max_docs_per_source,
            "max_doc_chars": self.max_doc_chars,
            "max_context_chars": self.max_context_chars,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "rrf_weight": self.rrf_weight,
            "framework_boost": self.framework_boost,
            "rrf_k": self.rrf_k,
        }

        if profile == "code_heavy":
            config.update(
                {
                    "fetch_multiplier": max(config["fetch_multiplier"], 14),
                    "max_query_variants": min(max(config["max_query_variants"], 4), 5),
                    "max_docs_per_source": max(config["max_docs_per_source"], 3),
                    "max_doc_chars": max(config["max_doc_chars"], 1400),
                    "semantic_weight": 0.56,
                    "lexical_weight": 0.28,
                    "rrf_weight": 0.16,
                    "framework_boost": 0.14,
                }
            )
        elif profile == "conceptual":
            config.update(
                {
                    "fetch_multiplier": max(config["fetch_multiplier"], 10),
                    "max_query_variants": max(config["max_query_variants"], 3),
                    "max_docs_per_source": min(config["max_docs_per_source"], 2),
                    "max_doc_chars": max(config["max_doc_chars"], 1200),
                    "semantic_weight": 0.72,
                    "lexical_weight": 0.14,
                    "rrf_weight": 0.10,
                    "framework_boost": 0.04,
                }
            )
        else:
            # Balanced profile keeps env defaults, with safer floor values.
            config.update(
                {
                    "fetch_multiplier": max(config["fetch_multiplier"], 12),
                    "max_query_variants": max(config["max_query_variants"], 3),
                    "max_docs_per_source": max(config["max_docs_per_source"], 2),
                    "semantic_weight": max(config["semantic_weight"], 0.60),
                    "lexical_weight": max(config["lexical_weight"], 0.20),
                    "rrf_weight": max(config["rrf_weight"], 0.10),
                }
            )

        return config

    def _normalize_framework_name(self, framework: str) -> str:
        return (framework or "").strip().lower()

    def _normalize_runtime_mode(self, runtime_preferences: Optional[Dict]) -> str:
        prefs = runtime_preferences or {}
        mode = (prefs.get("mode") or "").strip().lower()
        if mode in {"auto", "modern", "legacy"}:
            return mode
        return "auto"

    def _extract_framework_version_spec(
        self,
        framework: str,
        runtime_preferences: Optional[Dict],
        version_constraint: Optional[str] = None,
    ) -> str:
        if version_constraint and str(version_constraint).strip():
            return str(version_constraint).strip()

        prefs = runtime_preferences or {}
        requested = (
            prefs.get("packages")
            or prefs.get("package_versions")
            or {}
        )
        safe_framework = self._normalize_framework_name(framework)
        if isinstance(requested, dict):
            for package_name, spec in requested.items():
                package_key = (package_name or "").strip().lower()
                if package_key == safe_framework and str(spec or "").strip():
                    return str(spec).strip()

        framework_spec = prefs.get("framework_version")
        if isinstance(framework_spec, str) and framework_spec.strip():
            return framework_spec.strip()

        return ""

    def _parse_version_tuple(self, value: str) -> Optional[tuple]:
        numbers = re.findall(r"\d+", value or "")
        if not numbers:
            return None
        return tuple(int(item) for item in numbers[:4])

    def _compare_versions(self, left: tuple, right: tuple) -> int:
        width = max(len(left), len(right))
        lhs = left + (0,) * (width - len(left))
        rhs = right + (0,) * (width - len(right))
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0

    def _is_version_in_spec(self, version_value: str, spec: str) -> Optional[bool]:
        installed = self._parse_version_tuple(version_value)
        if installed is None:
            return None

        clauses = [item.strip() for item in (spec or "").split(",") if item.strip()]
        if not clauses:
            return None

        for clause in clauses:
            range_match = re.match(
                r"^([0-9][0-9A-Za-z.\-+]*)\s*-\s*([0-9][0-9A-Za-z.\-+]*)$",
                clause,
            )
            if range_match:
                lower = self._parse_version_tuple(range_match.group(1))
                upper = self._parse_version_tuple(range_match.group(2))
                if lower is None or upper is None:
                    return None
                if self._compare_versions(installed, lower) < 0 or self._compare_versions(installed, upper) > 0:
                    return False
                continue

            match = re.match(r"^(>=|<=|==|!=|>|<)\s*([0-9][0-9A-Za-z.\-+]*)$", clause)
            operator = "=="
            target_text = clause
            if match:
                operator = match.group(1)
                target_text = match.group(2)
            target = self._parse_version_tuple(target_text)
            if target is None:
                return None

            cmp_value = self._compare_versions(installed, target)
            if operator == ">" and not (cmp_value > 0):
                return False
            if operator == ">=" and not (cmp_value >= 0):
                return False
            if operator == "<" and not (cmp_value < 0):
                return False
            if operator == "<=" and not (cmp_value <= 0):
                return False
            if operator == "==" and not (cmp_value == 0):
                return False
            if operator == "!=" and not (cmp_value != 0):
                return False
        return True

    def _version_sort_key(self, version: str):
        text = (version or "").strip()
        lowered = text.lower()
        prerelease = bool(re.search(r"(?:alpha|beta|rc|dev|preview|pre|a|b)\d*", lowered))
        # Prefer stable release over prerelease for same base version.
        core_text = re.split(r"(?:alpha|beta|rc|dev|preview|pre|a|b)", lowered, maxsplit=1)[0]
        core_parsed = self._parse_version_tuple(core_text or "")
        parsed = self._parse_version_tuple(text or "")

        if core_parsed is not None:
            return (
                3,
                core_parsed,
                1 if not prerelease else 0,
                parsed or tuple(),
                lowered,
            )
        if lowered in self._LATEST_VERSION_LABELS:
            return (2, tuple(), 0, lowered)
        if lowered:
            return (1, tuple(), 0, lowered)
        return (0, tuple(), 0, lowered)

    def _strip_prefix_operators(self, spec: str) -> str:
        return re.sub(r"^[\s<>=!~^]+", "", spec or "").strip()

    def _match_requested_version(self, versions: List[str], requested_spec: str) -> Optional[str]:
        if not versions:
            return None
        spec = (requested_spec or "").strip()
        if not spec:
            return None

        direct_lookup = {(item or "").strip().lower(): item for item in versions}
        if spec.lower() in direct_lookup:
            return direct_lookup[spec.lower()]

        cleaned = self._strip_prefix_operators(spec)
        if cleaned.lower() in direct_lookup:
            return direct_lookup[cleaned.lower()]

        normalized_versions = {self._strip_prefix_operators(item).lower(): item for item in versions}
        if cleaned.lower() in normalized_versions:
            return normalized_versions[cleaned.lower()]

        wildcard_match = re.search(r"\*", cleaned)
        if wildcard_match:
            prefix = cleaned[: wildcard_match.start()].rstrip(".").lower()
            matches = [
                item
                for item in versions
                if self._strip_prefix_operators(item).lower().startswith(prefix)
            ]
            if matches:
                return max(matches, key=self._version_sort_key)

        if any(token in spec for token in [">", "<", "=", "!", ","]):
            spec_matches = [
                item for item in versions if self._is_version_in_spec(item, spec) is True
            ]
            if spec_matches:
                return max(spec_matches, key=self._version_sort_key)

        parsed_spec = self._parse_version_tuple(cleaned)
        if parsed_spec is not None:
            prefix_matches = []
            for item in versions:
                parsed_item = self._parse_version_tuple(item)
                if parsed_item is None:
                    continue
                if parsed_item[: len(parsed_spec)] == parsed_spec:
                    prefix_matches.append(item)
            if prefix_matches:
                return max(prefix_matches, key=self._version_sort_key)

        return None

    def _scan_framework_versions(self, collection, framework: str) -> List[str]:
        safe_framework = self._normalize_framework_name(framework)
        if not safe_framework:
            return []

        discovered = set()
        batch = 4000
        offset = 0
        where_error = None
        try:
            while True:
                response = collection.get(
                    include=["metadatas"],
                    where={"framework": safe_framework},
                    limit=batch,
                    offset=offset,
                )
                metadatas = response.get("metadatas", [])
                if not metadatas:
                    break
                for metadata in metadatas:
                    normalized = self._normalize_metadata(metadata, requested_framework=safe_framework)
                    metadata_framework = self._normalize_framework_name(normalized.get("framework", ""))
                    if metadata_framework != safe_framework:
                        continue
                    version_value = str(normalized.get("version", "")).strip()
                    if version_value:
                        discovered.add(version_value)
                offset += batch
                if len(metadatas) < batch:
                    break
        except Exception as filter_error:
            where_error = filter_error

        # Fallback scan all records if where-filter failed or yielded no matches.
        if where_error or not discovered:
            if where_error:
                logger.warning(
                    "RAG version scan fallback activated framework=%s reason=%s",
                    safe_framework,
                    where_error,
                )
            total = collection.count()
            offset = 0
            while offset < total:
                response = collection.get(
                    include=["metadatas"],
                    limit=batch,
                    offset=offset,
                )
                metadatas = response.get("metadatas", [])
                for metadata in metadatas:
                    normalized = self._normalize_metadata(metadata, requested_framework=safe_framework)
                    metadata_framework = self._normalize_framework_name(normalized.get("framework", ""))
                    if metadata_framework != safe_framework:
                        continue
                    version_value = str(normalized.get("version", "")).strip()
                    if version_value:
                        discovered.add(version_value)
                offset += batch
        return sorted(discovered, key=self._version_sort_key)

    def _get_framework_versions(self, collection, framework: str) -> Dict:
        safe_framework = self._normalize_framework_name(framework)
        now = time.time()
        cached = self._framework_version_cache.get(safe_framework)
        if cached and float(cached.get("expires_at", 0)) > now:
            snapshot = dict(cached)
            snapshot["cache_hit"] = True
            return snapshot

        versions = self._scan_framework_versions(collection, safe_framework)
        latest = max(versions, key=self._version_sort_key) if versions else ""
        snapshot = {
            "framework": safe_framework,
            "versions": versions,
            "latest_version": latest,
            "version_count": len(versions),
            "expires_at": now + self.latest_version_cache_ttl_seconds,
            "cache_hit": False,
        }
        self._framework_version_cache[safe_framework] = snapshot
        return dict(snapshot)

    def get_framework_version_marker(self, framework: str) -> str:
        """
        Return a stable marker for cache keys that changes when framework docs version set changes.
        """
        safe_framework = self._normalize_framework_name(framework)
        if not safe_framework or safe_framework not in self._FRAMEWORKS_WITH_VERSIONING:
            return f"{safe_framework or 'unknown'}:unversioned"

        collection = self.collection or self._safe_get_collection(self.collection_name)
        if collection is None:
            return f"{safe_framework}:collection-missing"

        snapshot = self._get_framework_versions(collection, safe_framework)
        latest = str(snapshot.get("latest_version", "") or "none")
        count = int(snapshot.get("version_count", 0) or 0)
        return f"{safe_framework}:{latest}:{count}"

    def _resolve_version_filter(
        self,
        collection,
        framework: str,
        runtime_preferences: Optional[Dict],
        version_constraint: Optional[str],
        prefer_latest_version: Optional[bool],
    ) -> Dict:
        safe_framework = self._normalize_framework_name(framework)
        mode = self._normalize_runtime_mode(runtime_preferences)
        requested_spec = self._extract_framework_version_spec(
            framework=safe_framework,
            runtime_preferences=runtime_preferences,
            version_constraint=version_constraint,
        )
        should_prefer_latest = (
            self.default_to_latest_version
            if prefer_latest_version is None
            else bool(prefer_latest_version)
        )

        result = {
            "active": False,
            "framework": safe_framework,
            "mode": mode,
            "strict": self.strict_version_selection,
            "strategy": "none",
            "requested_spec": requested_spec,
            "selected_version": "",
            "latest_version": "",
            "available_version_count": 0,
            "available_versions": [],
            "cache_hit": False,
            "fallback_to_unfiltered": False,
            "error": "",
            "where": None,
        }

        if not safe_framework or safe_framework not in self._FRAMEWORKS_WITH_VERSIONING:
            result["strategy"] = "framework_not_version_targeted"
            return result

        if (
            mode == "legacy"
            and self.legacy_mode_allow_all_versions
            and not requested_spec
            and not self.strict_version_selection
        ):
            result["strategy"] = "legacy_all_versions"
            return result

        if not requested_spec and not should_prefer_latest:
            result["strategy"] = "latest_preference_disabled"
            return result

        version_snapshot = self._get_framework_versions(collection, safe_framework)
        result["available_version_count"] = int(version_snapshot.get("version_count", 0) or 0)
        result["available_versions"] = list(version_snapshot.get("versions", []))
        result["latest_version"] = str(version_snapshot.get("latest_version", "") or "")
        result["cache_hit"] = bool(version_snapshot.get("cache_hit", False))

        available_versions = result["available_versions"]
        if not available_versions:
            result["strategy"] = "no_versions_available"
            result["error"] = (
                f"No versioned documents found for framework '{safe_framework}' in collection "
                f"'{self.collection_name}'."
            )
            return result

        selected_version = ""

        if requested_spec:
            selected_version = self._match_requested_version(available_versions, requested_spec) or ""
            if selected_version:
                result["strategy"] = "requested_version_spec"
            else:
                result["strategy"] = "requested_spec_unmatched"
                result["error"] = (
                    f"Requested runtime version '{requested_spec}' for framework '{safe_framework}' "
                    f"was not found in ChromaDB."
                )
                if not self.strict_version_selection:
                    if mode == "legacy" and self.legacy_mode_allow_all_versions:
                        result["strategy"] = "requested_spec_unmatched_legacy_fallback"
                        return result
                    if should_prefer_latest and result["latest_version"]:
                        selected_version = result["latest_version"]
                        result["strategy"] = "requested_spec_unmatched_latest_fallback"
                        result["error"] = ""
                    else:
                        return result
                else:
                    return result
        elif should_prefer_latest and result["latest_version"]:
            selected_version = result["latest_version"]
            result["strategy"] = "latest_default"
        else:
            result["strategy"] = "no_version_selected"
            return result

        result["selected_version"] = selected_version
        result["active"] = bool(selected_version)
        if result["active"]:
            result["where"] = {
                "$and": [
                    {"framework": safe_framework},
                    {"version": selected_version},
                ]
            }
        return result

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-\.]{1,}", (text or "").lower())
        return [t for t in tokens if t not in self._STOPWORDS]

    def _extract_query_terms(self, query: str, framework: str) -> List[str]:
        aliases = self._framework_aliases(framework)
        seen = set()
        terms: List[str] = []

        for token in aliases + self._tokenize(query):
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            terms.append(token)
            if len(terms) >= 24:
                break
        return terms

    def _build_query_variants(self, query: str, framework: str, max_query_variants: int) -> List[str]:
        normalized = " ".join((query or "").split())
        terms = self._extract_query_terms(normalized, framework)

        variants: List[str] = []

        def add_variant(value: str):
            item = " ".join((value or "").split())
            if item and item not in variants:
                variants.append(item)

        add_variant(f"{framework} {normalized}".strip())
        add_variant(normalized)
        add_variant(" ".join(terms[:14]))
        add_variant(f"{framework} api usage best practices {normalized}".strip())

        return variants[: max_query_variants]

    def _doc_id(self, content: str, metadata: Dict) -> str:
        normalized = self._normalize_metadata(metadata)
        explicit = normalized.get("id") or normalized.get("doc_id")
        if explicit and str(explicit).strip():
            return str(explicit).strip()

        source = normalized.get("source") or normalized.get("url") or ""
        path = normalized.get("path", "")
        section = normalized.get("section", "") or normalized.get("type", "")
        framework = normalized.get("framework", "")
        version = normalized.get("version", "")
        chunk_token = normalized.get("chunk_hash") or normalized.get("chunk_id") or ""
        signature = (
            f"{framework}|{version}|{source}|{path}|{section}|{chunk_token}|"
            f"{self._query_preview(content, 180)}"
        )
        return hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()

    def _source_key(self, doc: Dict) -> str:
        metadata = self._normalize_metadata(doc.get("metadata") or {})
        return str(
            doc.get("source")
            or metadata.get("source")
            or metadata.get("url")
            or metadata.get("path")
            or metadata.get("source_type")
            or doc.get("section")
            or "unknown"
        ).lower()

    def _is_preferred_ibm_api_doc(self, doc: Dict) -> bool:
        metadata = self._normalize_metadata(doc.get("metadata") or {})
        dataset = str(metadata.get("dataset", "")).strip().lower()
        if dataset == self._IBM_API_DATASET:
            return True

        source = str(doc.get("source") or metadata.get("source") or "").strip().lower()
        if "quantum.cloud.ibm.com/docs/en/api" in source:
            return True

        url = str(metadata.get("url", "")).strip().lower()
        return "quantum.cloud.ibm.com/docs/en/api" in url

    def _lexical_score(self, query_terms: List[str], doc: Dict) -> float:
        if not query_terms:
            return 0.0
        query_set = set(query_terms)
        metadata = self._normalize_metadata(doc.get("metadata") or {})
        text = " ".join(
            [
                doc.get("content", ""),
                str(metadata.get("title", "")),
                doc.get("section", ""),
                doc.get("source", ""),
                str(metadata.get("url", "")),
                str(metadata.get("path", "")),
                str(metadata.get("source_type", "")),
                str(metadata.get("type", "")),
            ]
        )
        doc_terms = set(self._tokenize(text))
        if not doc_terms:
            return 0.0

        overlap = len(query_set.intersection(doc_terms))
        coverage = overlap / max(1, len(query_set))
        density = overlap / max(1, len(doc_terms))
        score = (coverage * 0.85) + (min(density * 20.0, 1.0) * 0.15)
        return max(0.0, min(score, 1.0))

    def _normalize_rrf(self, candidates: List[Dict], rrf_k: int):
        raw_scores = []
        for doc in candidates:
            ranks = doc.get("rank_positions", [])
            rrf_raw = 0.0
            for rank in ranks:
                rrf_raw += 1.0 / float(rrf_k + rank)
            doc["rrf_raw"] = rrf_raw
            raw_scores.append(rrf_raw)

        max_raw = max(raw_scores) if raw_scores else 0.0
        for doc in candidates:
            doc["rrf_score"] = (doc["rrf_raw"] / max_raw) if max_raw > 0 else 0.0

    def _hybrid_score(
        self,
        doc: Dict,
        semantic_weight: float,
        lexical_weight: float,
        rrf_weight: float,
        framework_boost: float,
    ) -> float:
        semantic = float(doc.get("semantic_score", 0.0))
        lexical = float(doc.get("lexical_score", 0.0))
        rrf = float(doc.get("rrf_score", 0.0))
        framework_affinity = 1.0 if doc.get("framework_match", False) else 0.0

        weighted_sum = (
            (semantic * semantic_weight)
            + (lexical * lexical_weight)
            + (rrf * rrf_weight)
            + (framework_affinity * framework_boost)
        )
        normalizer = semantic_weight + lexical_weight + rrf_weight + framework_boost
        if normalizer <= 0:
            return max(0.0, min(semantic, 1.0))
        return max(0.0, min(weighted_sum / normalizer, 1.0))

    def _clip_doc_content(self, content: str, max_doc_chars: int) -> str:
        text = (content or "").strip()
        if len(text) <= max_doc_chars:
            return text
        return f"{text[:max_doc_chars]}..."

    def _build_context_string(
        self,
        documents: List[Dict],
        max_doc_chars: int,
        max_context_chars: int,
    ) -> str:
        """Build formatted context string from retrieved documents with context budget."""
        if not documents:
            return ""

        context_parts = []
        consumed = 0
        for i, doc in enumerate(documents, 1):
            content = self._clip_doc_content(doc.get("content", ""), max_doc_chars=max_doc_chars)
            section = doc.get("section", "general")
            score = float(doc.get("score", 0.0))
            block = f"**Source {i}** ({section}, score={score:.3f})\n{content}\n"
            if consumed + len(block) > max_context_chars:
                remaining = max_context_chars - consumed
                if remaining < 200:
                    break
                block = f"{block[:remaining]}..."
            context_parts.append(block)
            consumed += len(block)
            if consumed >= max_context_chars:
                break

        return "\n".join(context_parts)

    async def retrieve_context(
        self,
        query: str,
        framework: str,
        top_k: int = None,
        score_threshold: Optional[float] = None,
        request_source: str = "unknown",
        runtime_preferences: Optional[Dict] = None,
        version_constraint: Optional[str] = None,
        prefer_latest_version: Optional[bool] = None,
    ) -> Dict:
        """
        Retrieve relevant documentation for a query.

        Strategy:
        1) Generate multiple query variants for recall.
        2) Retrieve larger candidate pool from Chroma.
        3) Hybrid rerank (semantic + lexical + reciprocal rank fusion + framework affinity).
        4) Apply diversity and context budget constraints.
        """
        start = time.perf_counter()
        normalized_query = (query or "").strip()
        limit = max(1, top_k or self.top_k)
        logger.info(
            "RAG request started source=%s framework=%s top_k=%s threshold=%s mode=%s query_preview=%s",
            request_source,
            framework,
            limit,
            score_threshold,
            self._normalize_runtime_mode(runtime_preferences),
            self._query_preview(normalized_query),
        )

        try:
            collection = self.collection or self._safe_get_collection(self.collection_name)
            if collection is None:
                logger.error(
                    "RAG request failed source=%s framework=%s reason=collection_missing collection=%s",
                    request_source,
                    framework,
                    self.collection_name,
                )
                return {
                    "documents": [],
                    "context": "",
                    "count": 0,
                    "average_score": 0,
                    "framework": framework,
                    "error": f"Collection '{self.collection_name}' not found",
                }
            self.collection = collection

            if not normalized_query:
                logger.warning(
                    "RAG request skipped source=%s framework=%s reason=empty_query",
                    request_source,
                    framework,
                )
                return {
                    "documents": [],
                    "context": "",
                    "count": 0,
                    "average_score": 0,
                    "framework": framework,
                }

            active_profile = self._resolve_active_profile(
                request_source=request_source,
                query=normalized_query,
            )
            profile = self._profile_config(active_profile)
            version_filter = self._resolve_version_filter(
                collection=collection,
                framework=framework,
                runtime_preferences=runtime_preferences,
                version_constraint=version_constraint,
                prefer_latest_version=prefer_latest_version,
            )
            if self.strict_version_selection and version_filter.get("error"):
                logger.error(
                    "RAG strict version selection blocked request source=%s framework=%s strategy=%s error=%s",
                    request_source,
                    framework,
                    version_filter.get("strategy"),
                    version_filter.get("error"),
                )
                return {
                    "documents": [],
                    "context": "",
                    "count": 0,
                    "average_score": 0,
                    "framework": framework,
                    "error": str(version_filter.get("error")),
                    "retrieval_metadata": {
                        "profile": active_profile,
                        "query_variants": [],
                        "candidate_count": 0,
                        "framework_candidate_count": 0,
                        "selected_count": 0,
                        "fetch_limit": 0,
                        "weights": {
                            "semantic": profile["semantic_weight"],
                            "lexical": profile["lexical_weight"],
                            "rrf": profile["rrf_weight"],
                            "framework_boost": profile["framework_boost"],
                        },
                        "version_filter": {
                            "active": version_filter.get("active", False),
                            "strict": version_filter.get("strict", self.strict_version_selection),
                            "strategy": version_filter.get("strategy"),
                            "mode": version_filter.get("mode"),
                            "requested_spec": version_filter.get("requested_spec"),
                            "selected_version": version_filter.get("selected_version"),
                            "latest_version": version_filter.get("latest_version"),
                            "available_version_count": version_filter.get("available_version_count", 0),
                            "fallback_to_unfiltered": version_filter.get("fallback_to_unfiltered", False),
                            "error": version_filter.get("error", ""),
                        },
                    },
                }

            query_variants = self._build_query_variants(
                normalized_query,
                framework,
                max_query_variants=profile["max_query_variants"],
            )
            query_terms = self._extract_query_terms(normalized_query, framework)
            fetch_limit = min(
                max(limit * profile["fetch_multiplier"], limit * 6),
                profile["max_fetch_results"],
            )

            logger.info(
                "RAG retrieval plan source=%s profile=%s variants=%s fetch_limit=%s query_terms=%s",
                request_source,
                active_profile,
                len(query_variants),
                fetch_limit,
                len(query_terms),
            )

            query_vectors = embedding_service.encode(query_variants).tolist()
            query_kwargs = {
                "query_embeddings": query_vectors,
                "n_results": fetch_limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if version_filter.get("where"):
                query_kwargs["where"] = version_filter["where"]

            result = collection.query(**query_kwargs)

            raw_docs_by_query = result.get("documents") or []
            if version_filter.get("where") and not any(raw_docs_by_query):
                if self.strict_version_selection:
                    version_filter["error"] = (
                        f"No documents found for strict version filter framework='{framework}' "
                        f"version='{version_filter.get('selected_version')}'."
                    )
                    logger.error(
                        "RAG strict version selection yielded zero documents source=%s framework=%s selected_version=%s",
                        request_source,
                        framework,
                        version_filter.get("selected_version"),
                    )
                    return {
                        "documents": [],
                        "context": "",
                        "count": 0,
                        "average_score": 0,
                        "framework": framework,
                        "error": str(version_filter["error"]),
                        "retrieval_metadata": {
                            "profile": active_profile,
                            "query_variants": query_variants,
                            "candidate_count": 0,
                            "framework_candidate_count": 0,
                            "selected_count": 0,
                            "fetch_limit": fetch_limit,
                            "weights": {
                                "semantic": profile["semantic_weight"],
                                "lexical": profile["lexical_weight"],
                                "rrf": profile["rrf_weight"],
                                "framework_boost": profile["framework_boost"],
                            },
                            "version_filter": {
                                "active": version_filter.get("active", False),
                                "strict": version_filter.get("strict", self.strict_version_selection),
                                "strategy": version_filter.get("strategy"),
                                "mode": version_filter.get("mode"),
                                "requested_spec": version_filter.get("requested_spec"),
                                "selected_version": version_filter.get("selected_version"),
                                "latest_version": version_filter.get("latest_version"),
                                "available_version_count": version_filter.get("available_version_count", 0),
                                "fallback_to_unfiltered": False,
                                "error": version_filter.get("error", ""),
                            },
                        },
                    }

                version_filter["fallback_to_unfiltered"] = True
                logger.warning(
                    "RAG version-filter query returned no results source=%s framework=%s selected_version=%s strategy=%s; retrying unfiltered",
                    request_source,
                    framework,
                    version_filter.get("selected_version"),
                    version_filter.get("strategy"),
                )
                fallback_kwargs = {
                    "query_embeddings": query_vectors,
                    "n_results": fetch_limit,
                    "include": ["documents", "metadatas", "distances"],
                }
                result = collection.query(**fallback_kwargs)

            raw_docs_by_query = result.get("documents") or []
            raw_meta_by_query = result.get("metadatas") or []
            raw_dist_by_query = result.get("distances") or []
            aliases = self._framework_aliases(framework)

            candidates_map: Dict[str, Dict] = {}
            for q_idx, _variant in enumerate(query_variants):
                raw_docs = raw_docs_by_query[q_idx] if q_idx < len(raw_docs_by_query) else []
                raw_metas = raw_meta_by_query[q_idx] if q_idx < len(raw_meta_by_query) else []
                raw_dists = raw_dist_by_query[q_idx] if q_idx < len(raw_dist_by_query) else []

                for rank, (content, metadata, distance) in enumerate(zip(raw_docs, raw_metas, raw_dists), start=1):
                    metadata = self._normalize_metadata(metadata, requested_framework=framework)
                    safe_content = content or ""
                    doc_id = self._doc_id(safe_content, metadata)
                    semantic_score = self._distance_to_similarity(distance)
                    source = metadata.get("source") or metadata.get("url") or ""
                    section = (
                        metadata.get("section")
                        or metadata.get("source_type")
                        or metadata.get("type")
                        or metadata.get("path", "general")
                    )
                    framework_match = self._matches_framework(metadata, aliases)

                    existing = candidates_map.get(doc_id)
                    if existing is None:
                        candidates_map[doc_id] = {
                            "id": doc_id,
                            "content": safe_content,
                            "source": source,
                            "section": section,
                            "source_type": metadata.get("source_type", ""),
                            "version": metadata.get("version", ""),
                            "score": 0.0,
                            "semantic_score": semantic_score,
                            "lexical_score": 0.0,
                            "rrf_score": 0.0,
                            "metadata_quality": self._metadata_quality_score(metadata),
                            "framework_match": framework_match,
                            "metadata": metadata,
                            "rank_positions": [rank],
                        }
                    else:
                        existing["rank_positions"].append(rank)
                        if semantic_score > existing["semantic_score"]:
                            existing["semantic_score"] = semantic_score
                        if framework_match:
                            existing["framework_match"] = True
                        existing["metadata_quality"] = max(
                            float(existing.get("metadata_quality", 0.0)),
                            self._metadata_quality_score(metadata),
                        )

            all_candidates = list(candidates_map.values())
            for doc in all_candidates:
                doc["lexical_score"] = self._lexical_score(query_terms, doc)

            self._normalize_rrf(all_candidates, rrf_k=profile["rrf_k"])

            for doc in all_candidates:
                doc["score"] = round(
                    self._hybrid_score(
                        doc,
                        semantic_weight=profile["semantic_weight"],
                        lexical_weight=profile["lexical_weight"],
                        rrf_weight=profile["rrf_weight"],
                        framework_boost=profile["framework_boost"],
                    ),
                    6,
                )
                doc["semantic_score"] = round(float(doc["semantic_score"]), 6)
                doc["lexical_score"] = round(float(doc["lexical_score"]), 6)
                doc["rrf_score"] = round(float(doc["rrf_score"]), 6)

            all_candidates.sort(
                key=lambda item: (
                    item.get("score", 0.0),
                    item.get("framework_match", False),
                    item.get("metadata_quality", 0.0),
                    item.get("semantic_score", 0.0),
                    item.get("lexical_score", 0.0),
                ),
                reverse=True,
            )

            framework_candidates = [doc for doc in all_candidates if doc.get("framework_match")]
            candidate_docs = framework_candidates if framework_candidates else all_candidates

            if score_threshold is not None:
                thresholded = [doc for doc in candidate_docs if doc.get("semantic_score", 0.0) >= score_threshold]
                if thresholded:
                    candidate_docs = thresholded

            preferred_candidates = [
                doc for doc in candidate_docs if self._is_preferred_ibm_api_doc(doc)
            ]
            non_preferred_candidates = [
                doc for doc in candidate_docs if not self._is_preferred_ibm_api_doc(doc)
            ]
            ordered_candidates = preferred_candidates + non_preferred_candidates

            selected: List[Dict] = []
            source_counts = defaultdict(int)
            seen_signatures = set()

            for doc in ordered_candidates:
                source_key = self._source_key(doc)
                signature = self._query_preview(doc.get("content", "").lower(), max_len=260)
                if signature in seen_signatures:
                    continue
                if source_counts[source_key] >= profile["max_docs_per_source"]:
                    continue

                selected.append(doc)
                seen_signatures.add(signature)
                source_counts[source_key] += 1

                if len(selected) >= limit:
                    break

            if not selected and candidate_docs:
                selected = candidate_docs[:limit]

            context = self._build_context_string(
                selected,
                max_doc_chars=profile["max_doc_chars"],
                max_context_chars=profile["max_context_chars"],
            )
            average_score = (
                sum(float(doc.get("score", 0.0)) for doc in selected) / len(selected)
                if selected
                else 0.0
            )

            payload = {
                "documents": selected,
                "context": context,
                "count": len(selected),
                "average_score": average_score,
                "framework": framework,
                "retrieval_metadata": {
                    "profile": active_profile,
                    "query_variants": query_variants,
                    "candidate_count": len(all_candidates),
                    "framework_candidate_count": len(framework_candidates),
                    "preferred_candidate_count": len(preferred_candidates),
                    "preferred_selected_count": len(
                        [doc for doc in selected if self._is_preferred_ibm_api_doc(doc)]
                    ),
                    "selected_count": len(selected),
                    "fetch_limit": fetch_limit,
                    "weights": {
                        "semantic": profile["semantic_weight"],
                        "lexical": profile["lexical_weight"],
                        "rrf": profile["rrf_weight"],
                        "framework_boost": profile["framework_boost"],
                    },
                    "version_filter": {
                        "active": version_filter.get("active", False),
                        "strict": version_filter.get("strict", self.strict_version_selection),
                        "strategy": version_filter.get("strategy"),
                        "mode": version_filter.get("mode"),
                        "requested_spec": version_filter.get("requested_spec"),
                        "selected_version": version_filter.get("selected_version"),
                        "latest_version": version_filter.get("latest_version"),
                        "available_version_count": version_filter.get("available_version_count", 0),
                        "fallback_to_unfiltered": version_filter.get("fallback_to_unfiltered", False),
                        "error": version_filter.get("error", ""),
                    },
                },
            }
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "RAG request completed source=%s framework=%s profile=%s docs=%s avg_score=%.4f candidates=%s latency_ms=%s",
                request_source,
                framework,
                active_profile,
                payload["count"],
                payload["average_score"],
                len(all_candidates),
                latency_ms,
            )
            return payload

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                "RAG request error source=%s framework=%s latency_ms=%s error=%s",
                request_source,
                framework,
                latency_ms,
                e,
            )
            return {
                "documents": [],
                "context": "",
                "count": 0,
                "average_score": 0,
                "framework": framework,
                "error": str(e),
            }

    async def add_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: Dict,
    ):
        """Add a new document to the vector database."""
        try:
            collection = self._get_or_create_collection(collection_name or self.collection_name)
            normalized_metadata = self._normalize_metadata(metadata)

            embedding = embedding_service.encode_single(content)
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[normalized_metadata],
            )
            framework = str(normalized_metadata.get("framework", "")).strip().lower()
            if framework:
                self._framework_version_cache.pop(framework, None)
            logger.info("Added document %s to %s", doc_id, collection_name)
        except Exception as e:
            logger.error("Error adding document: %s", e)
            raise

    async def search_with_filters(
        self,
        query: str,
        collection_name: str,
        filters: Dict,
        top_k: int = 5,
    ) -> List[Dict]:
        """Search with metadata filters."""
        try:
            collection = self._safe_get_collection(collection_name) if collection_name else self.collection
            if collection is None:
                return []

            query_vector = embedding_service.encode_single(query)
            query_kwargs = {
                "query_embeddings": [query_vector],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                query_kwargs["where"] = filters

            result = collection.query(**query_kwargs)
            raw_documents = (result.get("documents") or [[]])[0]
            raw_metadatas = (result.get("metadatas") or [[]])[0]
            raw_distances = (result.get("distances") or [[]])[0]

            hits = []
            for content, metadata, distance in zip(raw_documents, raw_metadatas, raw_distances):
                normalized_metadata = self._normalize_metadata(metadata)
                hits.append(
                    {
                        "content": content,
                        "metadata": normalized_metadata,
                        "score": round(self._distance_to_similarity(distance), 6),
                    }
                )
            return hits
        except Exception as e:
            logger.error("Filtered search error: %s", e)
            return []

    def health_check(self) -> Dict:
        """Basic Chroma health check for API readiness."""
        try:
            collection = self.collection or self._safe_get_collection(self.collection_name)
            if collection is None:
                return {
                    "status": "unhealthy",
                    "reason": f"collection '{self.collection_name}' not found",
                }
            count = collection.count()
            return {
                "status": "healthy",
                "collection": self.collection_name,
                "documents": count,
                "persist_dir": str(self.persist_dir),
                "retrieval": {
                    "traffic_profile": self.traffic_profile,
                    "fetch_multiplier": self.fetch_multiplier,
                    "max_fetch_results": self.max_fetch_results,
                    "max_query_variants": self.max_query_variants,
                    "default_to_latest_version": self.default_to_latest_version,
                    "strict_version_selection": self.strict_version_selection,
                },
            }
        except Exception as e:
            logger.error("Chroma health check failed: %s", e)
            return {
                "status": "unhealthy",
                "reason": str(e),
            }


# Singleton instance
rag_service = RAGService()
