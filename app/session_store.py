"""PostgreSQL 会话持久化，以及 MinIO 图片生命周期协调。"""

from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import quote
from uuid import uuid4

from langchain_core.messages import messages_from_dict
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.attachment_store import (
    AttachmentObjectStore,
    attachment_object_key,
    attachment_prefix,
)
from app.game_agent.multimodal import decode_data_url
from app.game_agent.models import AttachmentInput


class SessionStore:
    """保存完整 Transcript 和可压缩的 Agent State；图片原文交给 MinIO。"""

    def __init__(
        self,
        database_url: str,
        attachment_store: AttachmentObjectStore | None = None,
    ):
        self.attachment_store = attachment_store
        self.pool = AsyncConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )

    async def setup(self) -> None:
        """打开连接池并创建当前版本需要的三张表。"""
        await self.pool.open()
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_state (
                        session_id TEXT PRIMARY KEY
                            REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        state JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_transcript (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL
                            REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_transcript_session
                    ON chat_transcript(session_id, id)
                    """
                )

    async def close(self) -> None:
        """关闭 PostgreSQL 连接池。"""
        await self.pool.close()

    async def record_user_message(
        self,
        session_id: str,
        question: str,
        attachments: list[AttachmentInput] | None = None,
    ) -> list[dict]:
        """先保存图片，再原子写入会话和用户 Transcript；失败时回滚图片。"""
        if attachments and self.attachment_store is None:
            raise RuntimeError("附件存储未配置，图片必须保存到 MinIO")

        now = datetime.now().astimezone()
        refs: list[dict] = []
        uploaded_keys: list[str] = []
        try:
            for attachment in attachments or []:
                raw = decode_data_url(
                    attachment.data_url,
                    attachment.size,
                    attachment.mime_type,
                )
                attachment_id = str(uuid4())
                object_key = attachment_object_key(session_id, attachment_id)
                await self.attachment_store.put(
                    object_key,
                    raw,
                    attachment.mime_type,
                )
                uploaded_keys.append(object_key)
                refs.append({
                    "attachment_id": attachment_id,
                    "name": attachment.name,
                    "mime_type": attachment.mime_type,
                    "size": attachment.size,
                    "data_url": (
                        f"/ai/sessions/{quote(session_id, safe='')}/attachments/{attachment_id}"
                    ),
                })

            async with self.pool.connection() as connection:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        await cursor.execute(
                            """
                            INSERT INTO chat_sessions(session_id, title, created_at, updated_at)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT(session_id)
                            DO UPDATE SET updated_at = EXCLUDED.updated_at
                            """,
                            (session_id, self._title(question), now, now),
                        )
                        await cursor.execute(
                            """
                            INSERT INTO chat_transcript(
                                session_id, role, content, attachments, created_at
                            ) VALUES (%s, 'user', %s, %s::jsonb, %s)
                            """,
                            (session_id, question, json.dumps(refs, ensure_ascii=False), now),
                        )
        except Exception:
            if self.attachment_store is not None and uploaded_keys:
                await self.attachment_store.delete_many(uploaded_keys)
            raise
        return refs

    async def record_assistant_message(self, session_id: str, answer: str) -> None:
        """把最终回答追加到完整 Transcript。"""
        now = datetime.now().astimezone()
        async with self.pool.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
                        INSERT INTO chat_transcript(
                            session_id, role, content, attachments, created_at
                        ) VALUES (%s, 'assistant', %s, '[]'::jsonb, %s)
                        """,
                        (session_id, answer, now),
                    )
                    await cursor.execute(
                        "UPDATE chat_sessions SET updated_at = %s WHERE session_id = %s",
                        (now, session_id),
                    )

    async def get_attachment(self, attachment_id: str, session_id: str) -> dict | None:
        """根据会话和附件 ID 直接读取 MinIO，不查询 PostgreSQL 映射表。"""
        if self.attachment_store is None:
            raise RuntimeError("附件存储未配置，无法读取 MinIO 对象")
        try:
            object_key = attachment_object_key(session_id, attachment_id)
        except ValueError:
            return None
        stored = await self.attachment_store.get(object_key)
        if stored is None:
            return None
        content, mime_type = stored
        return {"content": content, "mime_type": mime_type}

    async def list_sessions(self) -> list[dict]:
        """按最近更新时间返回会话列表。"""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT session_id, title, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    """
                )
                return [dict(row) for row in await cursor.fetchall()]

    async def get_messages(self, session_id: str) -> list[dict]:
        """返回供前端展示的完整 Transcript，不受上下文压缩影响。"""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT role, content, attachments, created_at
                    FROM chat_transcript
                    WHERE session_id = %s
                    ORDER BY id
                    """,
                    (session_id,),
                )
                return [dict(row) for row in await cursor.fetchall()]

    async def rename_session(self, session_id: str, title: str) -> bool:
        """修改会话标题；不存在时返回 False。"""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "UPDATE chat_sessions SET title = %s WHERE session_id = %s RETURNING session_id",
                    (title, session_id),
                )
                return await cursor.fetchone() is not None

    async def delete_session(self, session_id: str) -> bool:
        """删除 PostgreSQL 会话，并清理该会话的全部 MinIO 图片。"""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM chat_sessions WHERE session_id = %s RETURNING session_id",
                    (session_id,),
                )
                deleted = await cursor.fetchone() is not None
        if deleted and self.attachment_store is not None:
            await self.attachment_store.delete_prefix(attachment_prefix(session_id))
        return deleted

    async def load_state(self, session_id: str) -> dict:
        """读取 Agent State，并把 JSON 消息恢复成 LangChain 消息对象。"""
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT state FROM agent_state WHERE session_id = %s",
                    (session_id,),
                )
                row = await cursor.fetchone()
        if not row:
            return {}
        data = dict(row["state"])
        data["active_messages"] = list(messages_from_dict(data.get("active_messages", [])))
        return data

    async def save_state(self, session_id: str, state: dict) -> None:
        """把当前近期消息和 ContextSummary 保存为可恢复的 Agent State。"""
        data = dict(state)
        data["active_messages"] = [
            message.model_dump() for message in data.get("active_messages", [])
        ]
        payload = json.dumps(data, ensure_ascii=False, default=str)
        async with self.pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO agent_state(session_id, state, updated_at)
                    VALUES (%s, %s::jsonb, %s)
                    ON CONFLICT(session_id) DO UPDATE
                    SET state = EXCLUDED.state, updated_at = EXCLUDED.updated_at
                    """,
                    (session_id, payload, datetime.now().astimezone()),
                )

    @staticmethod
    def _title(question: str) -> str:
        """从首条问题生成简短的默认会话标题。"""
        normalized = " ".join(question.split())
        return normalized[:28] + ("…" if len(normalized) > 28 else "")
