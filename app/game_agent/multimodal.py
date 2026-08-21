"""图片引用的持久化表示，以及模型调用前的临时 Hydration。"""

from __future__ import annotations

import base64
import re
from typing import Awaitable, Callable

from langchain_core.messages import AnyMessage, HumanMessage

from app.game_agent.models import AttachmentRef


AttachmentLoader = Callable[[str, str], Awaitable[dict | None]]


def render_attachment_references(refs: list[AttachmentRef | dict]) -> str:
    """把附件元数据写成消息中的轻量引用，绝不包含 Base64。"""
    # 入口统一校验，避免文件名、MIME 或大小不合法的数据进入持久消息。
    normalized = [AttachmentRef.model_validate(item) for item in refs]
    # attachment:// 是 Harness 内部引用协议，模型请求前才会解析为真实图片块。
    return "\n".join(
        f"- attachment://{item.attachment_id} name={item.name} "
        f"mime_type={item.mime_type} size={item.size}"
        for item in normalized
    )


async def hydrate_current_images(
    model_context: list[AnyMessage],
    refs: list[AttachmentRef | dict],
    *,
    session_id: str,
    loader: AttachmentLoader | None,
) -> list[AnyMessage]:
    """从 MinIO 读取本轮图片，生成临时模型消息，不修改持久化消息。"""
    normalized = [AttachmentRef.model_validate(item) for item in refs]
    if not normalized:
        return list(model_context)
    if loader is None:
        raise RuntimeError("存在图片引用，但没有配置附件读取器")

    # Provider 不认识 attachment_id，因此在请求边界读取原图并转换成 image_url。
    image_blocks = []
    for ref in normalized:
        stored = await loader(ref.attachment_id, session_id)
        if stored is None:
            raise ValueError(f"图片附件不存在或不属于当前会话：{ref.attachment_id}")
        mime_type = str(stored.get("mime_type") or ref.mime_type)
        if not mime_type.startswith("image/"):
            raise ValueError(f"附件不是图片：{ref.attachment_id}")
        raw = bytes(stored["content"])
        encoded = base64.b64encode(raw).decode("ascii")
        image_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        })

    # 复制列表和 HumanMessage；Base64 只进入这个临时副本，不会写回 Agent State。
    hydrated = list(model_context)
    # 图片属于本轮最新用户输入，因此从后向前寻找最近的 HumanMessage。
    for index in range(len(hydrated) - 1, -1, -1):
        message = hydrated[index]
        if not isinstance(message, HumanMessage):
            continue
        content = message.content if isinstance(message.content, list) else [
            {"type": "text", "text": str(message.content)}
        ]
        hydrated[index] = message.model_copy(update={"content": [*content, *image_blocks]})
        return hydrated
    raise ValueError("当前模型上下文中没有可挂载图片的用户消息")


def decode_data_url(
    data_url: str,
    declared_size: int,
    declared_mime_type: str,
) -> bytes:
    """校验前端 Data URL 并返回可安全写入 MinIO 的图片字节。"""
    # 只接受完整 Base64 Data URL，拒绝普通 URL、路径和非 Base64 内容。
    match = re.fullmatch(r"data:([^;,]+)?(?:;charset=[^;,]+)?;base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("附件必须使用 base64 Data URL")
    encoded_mime_type = str(match.group(1) or "").lower()
    # 同时比较请求字段与 Data URL 声明，避免伪造文件类型。
    if encoded_mime_type != declared_mime_type.lower():
        raise ValueError("附件 MIME Type 与 Data URL 不一致")
    try:
        # validate=True 会拒绝非法字符和错误 padding，而不是宽松解码。
        raw = base64.b64decode(match.group(2), validate=True)
    except ValueError as exc:
        raise ValueError("附件 base64 内容无效") from exc
    # 服务端再次执行大小限制，不能只相信浏览器校验。
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("单个附件不能超过 10MB")
    if len(raw) != declared_size:
        raise ValueError("附件大小与声明不一致")
    return raw
