"""
Unified conversational chatbot endpoint with session-aware memory.
"""
import logging
import re
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AliasChoices, BaseModel, Field

from core.config import settings
from core.security import get_current_active_user
from ml.prompts import ChatbotPrompts
from services.chat_memory_service import chat_memory_service
from services.llm_service import llm_service
from services.rag_service import rag_service
from services.runtime_compatibility import build_runtime_bundle_with_rag
from schemas.common import ClientContext, RuntimePreferences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chatbot"])

VALID_FRAMEWORKS = {"qiskit", "pennylane", "cirq", "torchquantum"}


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        validation_alias=AliasChoices("query", "message"),
        description="User query text.",
    )
    framework: Optional[str] = Field(
        None,
        description="Optional explicit framework preference: qiskit, pennylane, cirq, torchquantum.",
    )
    detail_level: str = Field("intermediate", description="beginner, intermediate, or advanced")
    session_id: Optional[str] = Field(None, description="Existing chat session id to continue")
    new_session: bool = Field(False, description="Force creating a new chat session")
    include_other_session_summaries: bool = Field(
        False,
        description="Include summaries from other sessions in context",
    )
    other_session_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of specific session ids to summarize",
    )
    client_context: Optional[ClientContext] = Field(
        None,
        description="Client metadata for version-aware retrieval/generation.",
    )
    runtime_preferences: Optional[RuntimePreferences] = Field(
        None,
        description="Optional explicit runtime target for version-aware chat code answers.",
    )


class ChatResponse(BaseModel):
    intent: str
    reply: str
    code: Optional[str] = None
    explanation: Optional[Dict[str, str]] = None
    session_id: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


def _extract_code_block(text: str) -> Optional[str]:
    match = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _detect_framework(query: str, explicit_framework: Optional[str]) -> str:
    if explicit_framework:
        normalized = explicit_framework.strip().lower()
        if normalized not in VALID_FRAMEWORKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid framework. Must be one of: {sorted(VALID_FRAMEWORKS)}",
            )
        return normalized

    text = (query or "").lower()
    framework_signals = [
        ("qiskit", ("qiskit", "ibm quantum", "quantumcircuit")),
        ("pennylane", ("pennylane", "qml.", "xanadu")),
        ("cirq", ("cirq", "quantumlib")),
        ("torchquantum", ("torchquantum", "quantum ml", "hybrid quantum")),
    ]
    for framework, signals in framework_signals:
        if any(signal in text for signal in signals):
            return framework

    return "general"


def _is_math_focused_query(query: str, detail_level: str) -> bool:
    if (detail_level or "").strip().lower() == "advanced":
        return True
    text = (query or "").lower()
    math_tokens = (
        "math",
        "mathematical",
        "equation",
        "formula",
        "derive",
        "proof",
        "state vector",
        "matrix",
        "hamiltonian",
        "bloch",
    )
    return any(token in text for token in math_tokens)


def _build_retrieval_query(
    query: str,
    framework: str,
    math_focus: bool,
    memory_aware_query: str,
    runtime_suffix: str,
) -> str:
    focus_line = (
        "Prioritize conceptual clarity, quantum intuition, and mathematical formulations."
        if math_focus
        else "Prioritize accurate conceptual explanation and practical guidance."
    )
    framework_hint = (
        f"Framework emphasis: {framework}."
        if framework in VALID_FRAMEWORKS
        else "Framework emphasis: general quantum computing concepts."
    )
    return (
        f"{memory_aware_query}\n\n"
        f"Original query: {query}\n"
        f"{framework_hint}\n"
        f"{focus_line}\n\n"
        f"{runtime_suffix}"
    )


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    current_user: dict = Depends(get_current_active_user),
):
    """
    Conversational QA endpoint.
    Accepts a plain query, retrieves supporting docs, and generates a high-quality response.
    """
    start = time.time()

    try:
        query = (request.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        detected_framework = _detect_framework(query, request.framework)
        rag_framework = detected_framework if detected_framework in VALID_FRAMEWORKS else "general"
        runtime_framework = detected_framework if detected_framework in VALID_FRAMEWORKS else "qiskit"
        math_focus = _is_math_focused_query(query, request.detail_level)

        runtime_bundle = await build_runtime_bundle_with_rag(
            framework=runtime_framework,
            client_context=request.client_context,
            runtime_preferences=request.runtime_preferences,
            request_source="/api/chat/message",
        )

        # Session memory bootstrap.
        user_id = current_user.get("user_id")
        memory_enabled = bool(settings.CHAT_ENABLE_SESSION_MEMORY and user_id)
        session_id = request.session_id
        session_context = ""
        cross_session_context = ""

        if memory_enabled:
            try:
                session_result = await chat_memory_service.get_or_create_session(
                    user_id=user_id,
                    framework=rag_framework,
                    requested_session_id=session_id,
                    force_new=request.new_session,
                    first_message=query,
                )
                session_id = session_result.get("session_id")
                if session_id:
                    session_context = await chat_memory_service.build_session_context(
                        user_id=user_id,
                        session_id=session_id,
                        limit=settings.CHAT_CONTEXT_MESSAGES,
                    )
                    if request.include_other_session_summaries:
                        cross_session_context = await chat_memory_service.build_cross_session_summary_context(
                            user_id=user_id,
                            current_session_id=session_id,
                            limit_sessions=settings.CHAT_CROSS_SESSION_SUMMARY_COUNT,
                            session_ids=request.other_session_ids,
                        )
                    await chat_memory_service.store_message(
                        user_id=user_id,
                        session_id=session_id,
                        role="user",
                        content=query,
                        intent="general",
                        framework=rag_framework,
                        metadata={"source": "/api/chat/message"},
                    )
            except Exception as mem_error:
                logger.warning("Chat memory unavailable, continuing without memory: %s", mem_error)
                memory_enabled = False
                session_id = None
                session_context = ""
                cross_session_context = ""

        memory_aware_query = chat_memory_service.build_memory_aware_query(
            user_query=query,
            session_context=session_context,
            cross_session_context=cross_session_context,
        )
        retrieval_query = _build_retrieval_query(
            query=query,
            framework=rag_framework,
            math_focus=math_focus,
            memory_aware_query=memory_aware_query,
            runtime_suffix=runtime_bundle["rag_query_suffix"],
        )

        rag_results = await rag_service.retrieve_context(
            query=retrieval_query,
            framework=rag_framework,
            top_k=6,
            request_source="/api/chat/message",
        )

        prompt = ChatbotPrompts.build_general_prompt(
            framework=rag_framework,
            user_question=query,
            rag_context=rag_results.get("context", ""),
            conversation_context=session_context,
            cross_session_summary=cross_session_context,
            compatibility_context=runtime_bundle["compatibility_context"],
            math_focus=math_focus,
        )

        max_tokens = settings.CHAT_GENERAL_MAX_TOKENS
        if math_focus:
            max_tokens = max(max_tokens, 1200)
        llm = await llm_service.generate_code(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.25,
        )
        logger.info(
            "Chat QA LLM response provider=%s model=%s tokens=%s fallback_used=%s",
            llm.get("provider"),
            llm.get("model"),
            llm.get("tokens_used"),
            llm.get("fallback_used", False),
        )

        reply = (llm.get("generated_text") or "").strip()
        if not reply:
            reply = "I could not generate a response. Please try rephrasing your question."

        code_block = _extract_code_block(reply)

        if memory_enabled and user_id and session_id:
            try:
                await chat_memory_service.store_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    content=reply,
                    intent="general",
                    framework=rag_framework,
                    metadata={
                        "provider": llm.get("provider"),
                        "model": llm.get("model"),
                        "tokens_used": llm.get("tokens_used", 0),
                        "source": "/api/chat/message",
                    },
                )
            except Exception as e:
                logger.warning("Failed to persist assistant message session_id=%s error=%s", session_id, e)

        return {
            "intent": "general",
            "reply": reply,
            "code": code_block,
            "session_id": session_id,
            "metadata": {
                "framework_detected": rag_framework,
                "framework_explicit": request.framework,
                "math_focus": math_focus,
                "detail_level": request.detail_level,
                "session_id": session_id,
                "chat_memory_enabled": memory_enabled,
                "session_context_loaded": bool(session_context),
                "cross_session_summary_loaded": bool(cross_session_context),
                "rag_documents": len(rag_results.get("documents", [])),
                "rag_average_score": rag_results.get("average_score", 0),
                "tokens_used": llm.get("tokens_used", 0),
                "llm_provider": llm.get("provider"),
                "llm_model": llm.get("model"),
                "llm_attempt": llm.get("attempt"),
                "llm_fallback_used": llm.get("fallback_used", False),
                "requested_runtime": runtime_bundle.get("requested_runtime", {}),
                "runtime_requirements": runtime_bundle.get("effective_runtime_target", {}),
                "runtime_recommendations": runtime_bundle["runtime_recommendations"],
                "version_conflicts": runtime_bundle["version_conflicts"],
                "runtime_validation": runtime_bundle["runtime_validation"],
                "latency_ms": int((time.time() - start) * 1000),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error: %s", e)
        raise HTTPException(500, f"Chatbot request failed: {str(e)}")
