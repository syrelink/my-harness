"""MinIO 图片存储；对象键由 session_id 和 attachment_id 确定性生成。"""

from __future__ import annotations

import asyncio
import io
import os
from hashlib import sha256
from typing import Protocol
from uuid import UUID

from minio import Minio
from minio.error import S3Error


def attachment_prefix(session_id: str) -> str:
    """生成会话专属前缀，不把原始 session_id 暴露给对象存储。"""
    session_key = sha256(session_id.encode()).hexdigest()[:24]
    return f"sessions/{session_key}/"


def attachment_object_key(session_id: str, attachment_id: str) -> str:
    """用合法 UUID 生成对象键，客户端不能直接指定 MinIO 路径。"""
    normalized_id = str(UUID(attachment_id))
    return f"{attachment_prefix(session_id)}{normalized_id}"


class AttachmentObjectStore(Protocol):
    async def setup(self) -> None: ...
    async def put(self, object_key: str, content: bytes, mime_type: str) -> None: ...
    async def get(self, object_key: str) -> tuple[bytes, str] | None: ...
    async def delete_many(self, object_keys: list[str]) -> None: ...
    async def delete_prefix(self, prefix: str) -> None: ...


class MinioAttachmentStore:
    """通过 MinIO S3 API 保存图片原文；所有阻塞 SDK 调用都移出事件循环。"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    @classmethod
    def from_env(cls) -> "MinioAttachmentStore":
        secure = os.getenv("MINIO_SECURE", "false").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            bucket=os.getenv("MINIO_BUCKET", "game-rover-attachments"),
            secure=secure,
        )

    async def setup(self) -> None:
        def ensure_bucket() -> None:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)

        await asyncio.to_thread(ensure_bucket)

    async def put(self, object_key: str, content: bytes, mime_type: str) -> None:
        await asyncio.to_thread(
            self.client.put_object,
            self.bucket,
            object_key,
            io.BytesIO(content),
            len(content),
            content_type=mime_type,
        )

    async def get(self, object_key: str) -> tuple[bytes, str] | None:
        """读取对象正文及 MinIO 中保存的 Content-Type；不存在时返回 None。"""
        def download() -> tuple[bytes, str] | None:
            try:
                response = self.client.get_object(self.bucket, object_key)
            except S3Error as exc:
                if exc.code in {"NoSuchKey", "NoSuchObject"}:
                    return None
                raise
            try:
                content_type = response.headers.get("Content-Type") or "application/octet-stream"
                return response.read(), content_type
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(download)

    async def delete_many(self, object_keys: list[str]) -> None:
        for object_key in object_keys:
            await asyncio.to_thread(self.client.remove_object, self.bucket, object_key)

    async def delete_prefix(self, prefix: str) -> None:
        """删除一个会话前缀下的全部对象。"""
        def remove_all() -> None:
            for item in self.client.list_objects(self.bucket, prefix=prefix, recursive=True):
                self.client.remove_object(self.bucket, item.object_name)

        await asyncio.to_thread(remove_all)
