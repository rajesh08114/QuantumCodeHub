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
        }
        return aliases.get(normalized, [normalized] if normalized else [])

    def _matches_framework(self, metadata: Dict, aliases: List[str]) -> bool:
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
        explicit = metadata.get("id") or metadata.get("doc_id") or metadata.get("chunk_id")
        if explicit:
            return str(explicit)
        source = metadata.get("source", "")
        path = metadata.get("path", "")
        section = metadata.get("section", "") or metadata.get("type", "")
        signature = f"{source}|{path}|{section}|{self._query_preview(content, 180)}"
        return hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()

    def _source_key(self, doc: Dict) -> str:
        metadata = doc.get("metadata") or {}
        return str(
            doc.get("source")
            or metadata.get("source")
            or metadata.get("path")
            or doc.get("section")
            or "unknown"
        ).lower()

    def _lexical_score(self, query_terms: List[str], doc: Dict) -> float:
        if not query_terms:
            return 0.0
        query_set = set(query_terms)
        metadata = doc.get("metadata") or {}
        text = " ".join(
            [
                doc.get("content", ""),
                doc.get("section", ""),
                doc.get("source", ""),
                str(metadata.get("path", "")),
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
            "RAG request started source=%s framework=%s top_k=%s threshold=%s query_preview=%s",
            request_source,
            framework,
            limit,
            score_threshold,
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
            result = collection.query(
                query_embeddings=query_vectors,
                n_results=fetch_limit,
                include=["documents", "metadatas", "distances"],
            )

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
                    metadata = metadata or {}
                    safe_content = content or ""
                    doc_id = self._doc_id(safe_content, metadata)
                    semantic_score = self._distance_to_similarity(distance)
                    source = metadata.get("source", "")
                    section = metadata.get("section") or metadata.get("type") or metadata.get("path", "general")
                    framework_match = self._matches_framework(metadata, aliases)

                    existing = candidates_map.get(doc_id)
                    if existing is None:
                        candidates_map[doc_id] = {
                            "id": doc_id,
                            "content": safe_content,
                            "source": source,
                            "section": section,
                            "score": 0.0,
                            "semantic_score": semantic_score,
                            "lexical_score": 0.0,
                            "rrf_score": 0.0,
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

            selected: List[Dict] = []
            source_counts = defaultdict(int)
            seen_signatures = set()

            for doc in candidate_docs:
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
                    "selected_count": len(selected),
                    "fetch_limit": fetch_limit,
                    "weights": {
                        "semantic": profile["semantic_weight"],
                        "lexical": profile["lexical_weight"],
                        "rrf": profile["rrf_weight"],
                        "framework_boost": profile["framework_boost"],
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

            embedding = embedding_service.encode_single(content)
            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}],
            )
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
                hits.append(
                    {
                        "content": content,
                        "metadata": metadata or {},
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
