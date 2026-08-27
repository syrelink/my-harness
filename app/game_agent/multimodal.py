"""图片引用的持久化表示，以及模型调用前的临时 Hydration。"""

from __future__ import annotations

import base64
import re
from typing import Awaitable, Callable

from langchain_core.messages import AnyMessage, HumanMessage

from app.game_agent.models import AttachmentRef


AttachmentLoader = Callable[[str, str], Awaitable[dict | None]]


def _image_ref_from_block(block: object) -> AttachmentRef | None:
    """从持久消息的 image block 中取出附件引用。"""
    if not isinstance(block, dict) or block.get("type") != "image":
        return None
    attachment = block.get("attachment")
    if not attachment:
        return None
    return AttachmentRef.model_validate(attachment)


def _image_ref_text(ref: AttachmentRef) -> dict:
    """把历史图片引用降级成普通文本，避免旧图片反复进入模型请求。"""
    return {
        "type": "text",
        "text": (
            f"[历史图片: attachment_id={ref.attachment_id}, "
            f"name={ref.name}, mime_type={ref.mime_type}, size={ref.size}]"
        ),
    }


async def _image_url_block(
    ref: AttachmentRef,
    *,
    session_id: str,
    loader: AttachmentLoader,
) -> dict:
    """按附件引用从 MinIO 读取图片，并转换为模型供应商支持的 image_url block。"""
    stored = await loader(ref.attachment_id, session_id)
    if stored is None:
        raise ValueError(f"图片附件不存在或不属于当前会话：{ref.attachment_id}")
    mime_type = str(stored.get("mime_type") or ref.mime_type)
    if not mime_type.startswith("image/"):
        raise ValueError(f"附件不是图片：{ref.attachment_id}")
    raw = bytes(stored["content"])
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


async def hydrate_current_images(
    model_context: list[AnyMessage],
    *,
    session_id: str,
    loader: AttachmentLoader | None,
) -> list[AnyMessage]:
    """把最新用户消息里的结构化图片引用临时转换成 image_url。

    持久状态里保存的是 ``{"type": "image", "attachment": ...}``。模型供应商不认识
    这个内部格式，所以只在请求模型前把最新 HumanMessage 的图片引用 hydrate 成
    ``image_url``；历史图片引用则保留为文本说明，不重新加载原图。
    """
    latest_human_index = next(
        (
            index
            for index in range(len(model_context) - 1, -1, -1)
            if isinstance(model_context[index], HumanMessage)
        ),
        None,
    )
    if latest_human_index is None:
        return list(model_context)

    hydrated = list(model_context)
    for index, message in enumerate(hydrated):
        if not isinstance(message, HumanMessage):
            continue
        if not isinstance(message.content, list):
            continue
        content = []
        changed = False
        for block in message.content:
            ref = _image_ref_from_block(block)
            if ref is None:
                content.append(block)
                continue
            changed = True
            if index == latest_human_index:
                if loader is None:
                    raise RuntimeError("存在图片引用，但没有配置附件读取器")
                content.append(await _image_url_block(ref, session_id=session_id, loader=loader))
            else:
                content.append(_image_ref_text(ref))
        if changed:
            hydrated[index] = message.model_copy(update={"content": content})
    return hydrated


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
