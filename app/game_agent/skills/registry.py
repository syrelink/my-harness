"""兼容 Agent Skills 目录规范的只读注册器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillMetadata:
    """启动时常驻上下文的轻量 Skill 元数据。"""

    name: str
    description: str


@dataclass(frozen=True)
class SkillDocument:
    """按需加载的 Skill 主体或 reference 文档。"""

    name: str
    resource: str | None
    content: str


class SkillRegistry:
    """扫描 Skill 元数据，并安全地按名称加载正文和一层 reference。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._metadata: dict[str, SkillMetadata] = {}
        self.refresh()

    @classmethod
    def default(cls) -> "SkillRegistry":
        return cls(Path(__file__).resolve().parent)

    def refresh(self) -> None:
        metadata = {}
        if not self.root.exists():
            self._metadata = metadata
            return
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            frontmatter, _ = self._parse_skill_file(skill_file)
            name = str(frontmatter.get("name", "")).strip()
            description = str(frontmatter.get("description", "")).strip()
            if not _SKILL_NAME.fullmatch(name):
                raise ValueError(f"无效 Skill 名称：{name or skill_file.parent.name}")
            if name != skill_file.parent.name:
                raise ValueError(f"Skill 名称必须与目录一致：{skill_file.parent.name}")
            if not description:
                raise ValueError(f"Skill 缺少 description：{name}")
            metadata[name] = SkillMetadata(name=name, description=description)
        self._metadata = metadata

    def catalog(self) -> list[SkillMetadata]:
        return list(self._metadata.values())

    def catalog_prompt(self) -> str:
        if not self._metadata:
            return "当前没有可用 Skill。"
        return "\n".join(
            f"- {item.name}: {item.description}"
            for item in self.catalog()
        )

    def load(self, name: str, resource: str | None = None) -> SkillDocument:
        if name not in self._metadata:
            raise ValueError(f"未知 Skill：{name}")
        skill_root = (self.root / name).resolve()
        if resource is None:
            _, content = self._parse_skill_file(skill_root / "SKILL.md")
            return SkillDocument(name=name, resource=None, content=content.strip())

        normalized = resource.strip().replace("\\", "/")
        if not re.fullmatch(r"references/[a-z0-9][a-z0-9-]*\.md", normalized):
            raise ValueError("resource 只能是 references/ 下的一层 Markdown 文件")
        resource_path = (skill_root / normalized).resolve()
        if skill_root not in resource_path.parents or not resource_path.is_file():
            raise ValueError(f"Skill Resource 不存在：{name}/{normalized}")
        return SkillDocument(name=name, resource=normalized, content=resource_path.read_text(encoding="utf-8").strip())

    @staticmethod
    def resource_key(name: str, resource: str) -> str:
        return f"{name}:{resource}"

    @staticmethod
    def split_resource_key(key: str) -> tuple[str, str]:
        name, resource = key.split(":", 1)
        return name, resource

    @staticmethod
    def _parse_skill_file(path: Path) -> tuple[dict, str]:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError(f"SKILL.md 缺少 YAML Frontmatter：{path}")
        try:
            frontmatter_text, body = text[4:].split("\n---\n", 1)
        except ValueError as exc:
            raise ValueError(f"SKILL.md Frontmatter 未闭合：{path}") from exc
        frontmatter = yaml.safe_load(frontmatter_text) or {}
        if not isinstance(frontmatter, dict):
            raise ValueError(f"SKILL.md Frontmatter 必须是对象：{path}")
        return frontmatter, body
