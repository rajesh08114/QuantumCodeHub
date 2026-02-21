"""
Session-based chat memory service.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List, Optional

from core.config import settings
from core.database import get_db_connection, release_db_connection

logger = logging.getLogger(__name__)


class ChatMemoryService:
    """Persist and retrieve per-user chat sessions/messages."""

    def __init__(self):
        self.enabled = settings.CHAT_ENABLE_SESSION_MEMORY

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        value = " ".join((text or "").split())
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}..."

    @staticmethod
    def _normalize_session_title(text: str) -> str:
        normalized = " ".join((text or "").split())
        if not normalized:
            return "New Chat"
        if len(normalized) > 80:
            return f"{normalized[:80]}..."
        return normalized

    async def init_schema(self):
        """Create chat session/message tables when missing."""
        if not self.enabled:
            logger.info("Chat memory disabled by configuration")
            return

        conn = await get_db_connection()
        try:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id UUID NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    framework TEXT NOT NULL DEFAULT 'qiskit',
                    summary TEXT NOT NULL DEFAULT '',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    user_id UUID NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT,
                    framework TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at DESC);
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                ON chat_messages(session_id, created_at DESC);
                """
            )
            logger.info("Chat memory schema initialized")
        finally:
            await release_db_connection(conn)

    async def get_or_create_session(
        self,
        user_id: str,
        framework: str,
        requested_session_id: Optional[str] = None,
        force_new: bool = False,
        first_message: Optional[str] = None,
    ) -> Dict:
        """Resolve or create a chat session scoped to user."""
        if not self.enabled:
            return {"session_id": None, "created": False}
        if not user_id:
            return {"session_id": None, "created": False}

        user_uuid = uuid.UUID(user_id)
        conn = await get_db_connection()
        try:
            if requested_session_id and not force_new:
                existing = await conn.fetchrow(
                    """
                    SELECT id
                    FROM chat_sessions
                    WHERE id = $1 AND user_id = $2
                    """,
                    requested_session_id,
                    user_uuid,
                )
                if existing:
                    await conn.execute(
                        """
                        UPDATE chat_sessions
                        SET updated_at = NOW(), framework = $1
                        WHERE id = $2 AND user_id = $3
                        """,
                        framework,
                        requested_session_id,
                        user_uuid,
                    )
                    return {"session_id": requested_session_id, "created": False}

            await self._enforce_user_session_limit(conn, user_uuid)

            session_id = str(uuid.uuid4())
            title = self._normalize_session_title(first_message or "")
            await conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title, framework)
                VALUES ($1, $2, $3, $4)
                """,
                session_id,
                user_uuid,
                title,
                framework,
            )
            return {"session_id": session_id, "created": True}
        finally:
            await release_db_connection(conn)

    async def _enforce_user_session_limit(self, conn, user_uuid: uuid.UUID):
        max_sessions = max(1, int(settings.CHAT_MAX_SESSIONS_PER_USER or 100))
        await conn.execute(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY updated_at DESC, created_at DESC) AS rn
                FROM chat_sessions
                WHERE user_id = $1
            )
            DELETE FROM chat_sessions
            WHERE id IN (SELECT id FROM ranked WHERE rn > $2)
            """,
            user_uuid,
            max_sessions,
        )

    async def store_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        framework: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not self.enabled or not user_id or not session_id:
            return

        user_uuid = uuid.UUID(user_id)
        conn = await get_db_connection()
        try:
            safe_role = (role or "user").strip().lower()
            safe_content = (content or "").strip()
            if not safe_content:
                return

            await conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id, user_id, role, content, intent, framework, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                session_id,
                user_uuid,
                safe_role,
                safe_content,
                intent,
                framework,
                json.dumps(metadata or {}),
            )

            # Keep session metadata fresh.
            await conn.execute(
                """
                UPDATE chat_sessions
                SET
                    updated_at = NOW(),
                    framework = COALESCE($1, framework),
                    title = CASE
                        WHEN message_count = 0 THEN $2
                        ELSE title
                    END,
                    message_count = message_count + 1
                WHERE id = $3 AND user_id = $4
                """,
                framework,
                self._normalize_session_title(safe_content),
                session_id,
                user_uuid,
            )

            await self._enforce_session_message_limit(conn, session_id)

            if safe_role == "assistant":
                await self._refresh_session_summary(conn, user_uuid, session_id)
        finally:
            await release_db_connection(conn)

    async def _enforce_session_message_limit(self, conn, session_id: str):
        max_messages = max(10, int(settings.CHAT_MAX_MESSAGES_PER_SESSION or 120))
        await conn.execute(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY created_at DESC, id DESC) AS rn
                FROM chat_messages
                WHERE session_id = $1
            )
            DELETE FROM chat_messages
            WHERE id IN (SELECT id FROM ranked WHERE rn > $2)
            """,
            session_id,
            max_messages,
        )
        await conn.execute(
            """
            UPDATE chat_sessions
            SET message_count = (
                SELECT COUNT(*) FROM chat_messages WHERE session_id = $1
            )
            WHERE id = $1
            """,
            session_id,
        )

    async def load_recent_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int,
    ) -> List[Dict]:
        if not self.enabled or not user_id or not session_id:
            return []

        user_uuid = uuid.UUID(user_id)
        safe_limit = max(1, min(limit, 50))
        conn = await get_db_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT role, content, intent, framework, created_at
                FROM chat_messages
                WHERE session_id = $1
                  AND user_id = $2
                ORDER BY created_at DESC, id DESC
                LIMIT $3
                """,
                session_id,
                user_uuid,
                safe_limit,
            )
        finally:
            await release_db_connection(conn)

        ordered = list(reversed(rows))
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "intent": row["intent"],
                "framework": row["framework"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in ordered
        ]

    async def build_session_context(
        self,
        user_id: str,
        session_id: str,
        limit: int,
    ) -> str:
        messages = await self.load_recent_messages(user_id=user_id, session_id=session_id, limit=limit)
        if not messages:
            return ""

        lines: List[str] = []
        for item in messages:
            role = (item.get("role") or "user").lower()
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {self._clip(item.get('content', ''), 300)}")

        return "\n".join(lines)

    async def _refresh_session_summary(self, conn, user_uuid: uuid.UUID, session_id: str):
        summary = await self._summarize_session_from_messages(conn, user_uuid, session_id)
        if not summary:
            return
        max_chars = max(300, int(settings.CHAT_SESSION_SUMMARY_MAX_CHARS or 1200))
        summary = self._clip(summary, max_chars)

        await conn.execute(
            """
            UPDATE chat_sessions
            SET summary = $1, updated_at = NOW()
            WHERE id = $2 AND user_id = $3
            """,
            summary,
            session_id,
            user_uuid,
        )

    async def _summarize_session_from_messages(
        self,
        conn,
        user_uuid: uuid.UUID,
        session_id: str,
        limit: int = 24,
    ) -> str:
        rows = await conn.fetch(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = $1 AND user_id = $2
            ORDER BY created_at DESC, id DESC
            LIMIT $3
            """,
            session_id,
            user_uuid,
            max(4, limit),
        )
        if not rows:
            return ""

        ordered = list(reversed(rows))
        summary_lines: List[str] = []
        for row in ordered:
            role = (row["role"] or "user").lower()
            prefix = "User asked" if role == "user" else "Assistant answered"
            summary_lines.append(f"- {prefix}: {self._clip(row['content'], 180)}")
        return "\n".join(summary_lines)

    async def build_cross_session_summary_context(
        self,
        user_id: str,
        current_session_id: Optional[str],
        limit_sessions: int,
        session_ids: Optional[List[str]] = None,
    ) -> str:
        if not self.enabled or not user_id:
            return ""

        user_uuid = uuid.UUID(user_id)
        safe_limit = max(1, min(limit_sessions, 10))
        conn = await get_db_connection()
        try:
            if session_ids:
                rows = await conn.fetch(
                    """
                    SELECT id, title, summary, updated_at
                    FROM chat_sessions
                    WHERE user_id = $1
                      AND id = ANY($2::text[])
                      AND ($3::text IS NULL OR id <> $3)
                    ORDER BY updated_at DESC
                    LIMIT $4
                    """,
                    user_uuid,
                    session_ids,
                    current_session_id,
                    safe_limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, title, summary, updated_at
                    FROM chat_sessions
                    WHERE user_id = $1
                      AND ($2::text IS NULL OR id <> $2)
                    ORDER BY updated_at DESC
                    LIMIT $3
                    """,
                    user_uuid,
                    current_session_id,
                    safe_limit,
                )

            if not rows:
                return ""

            blocks: List[str] = []
            max_chars = max(200, int(settings.CHAT_SESSION_SUMMARY_MAX_CHARS or 1200))

            for row in rows:
                title = row["title"] or row["id"]
                summary = (row["summary"] or "").strip()
                if not summary:
                    summary = await self._summarize_session_from_messages(
                        conn=conn,
                        user_uuid=user_uuid,
                        session_id=row["id"],
                        limit=14,
                    )
                    if summary:
                        await conn.execute(
                            """
                            UPDATE chat_sessions
                            SET summary = $1, updated_at = NOW()
                            WHERE id = $2 AND user_id = $3
                            """,
                            self._clip(summary, max_chars),
                            row["id"],
                            user_uuid,
                        )

                summary = self._clip(summary, max_chars)
                if not summary:
                    continue

                blocks.append(f"Session {row['id']} ({title}):\n{summary}")

            return "\n\n".join(blocks)
        finally:
            await release_db_connection(conn)

    def build_prompt_memory_block(
        self,
        session_context: str,
        cross_session_context: str,
    ) -> str:
        blocks: List[str] = []
        if session_context:
            blocks.append(f"Current session recent turns:\n{session_context}")
        if cross_session_context:
            blocks.append(f"Other relevant session summaries:\n{cross_session_context}")
        return "\n\n".join(blocks).strip()

    def build_memory_aware_query(
        self,
        user_query: str,
        session_context: str,
        cross_session_context: str,
    ) -> str:
        base = (user_query or "").strip()
        memory = self.build_prompt_memory_block(session_context, cross_session_context)
        if not memory:
            return base

        combined = f"{base}\n\nContext:\n{memory}"
        max_chars = max(300, int(settings.CHAT_MEMORY_QUERY_MAX_CHARS or 1200))
        return self._clip(combined, max_chars)

    def health_check(self) -> Dict:
        return {
            "enabled": bool(self.enabled),
            "context_messages": settings.CHAT_CONTEXT_MESSAGES,
            "max_messages_per_session": settings.CHAT_MAX_MESSAGES_PER_SESSION,
            "max_sessions_per_user": settings.CHAT_MAX_SESSIONS_PER_USER,
        }


chat_memory_service = ChatMemoryService()
