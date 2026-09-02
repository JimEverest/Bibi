# -*- coding: utf-8 -*-
"""
本地术语替换词典（方案2）

用法：
1. 识别完成后，按"最长优先"顺序对文本做字符串替换
2. 词典文件 terms.txt 放在项目根目录（或 config 中 dictionary.file 指定）
   格式：错误写法<TAB>正确写法，每行一条；# 开头为注释
3. 修改 terms.txt 后无需重启程序——每次转录都会重新加载（带 5 秒缓存）

默认内置一份常用错误映射（内置词典 + 用户词典双重生效，用户词典优先）。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 内置默认词典：识别输出 -> 期望输出（按实际错法持续补充）
DEFAULT_TERMS = {
    # -- 技术术语（来自实测日志的典型错误） --
    "as c note mcb": "MCP",
    "as c note": "MCP",
    "mcb": "MCP",
    "model l on its particle": "Model Context Protocol",
    "model contact protocol": "Model Context Protocol",
    "vs called copilots chant": "VS Code Copilot Chat",
    "copilots chant": "Copilot Chat",
    "jred divpts": "Jira、SharePoint",
    "jred": "Jira",
    "divpts": "SharePoint",
    "按卓住不控": "按住 Ctrl",
    "按装": "安装",
    # -- 常见同音/近音错字 --
    "杂次": "杂次",
}


class TermReplacer:
    """带缓存的术语替换器。词典文件每次转录时检查 mtime，5 秒内不重复加载。"""

    def __init__(self, dict_file: str | None = None, cache_seconds: float = 5.0):
        if dict_file:
            self.dict_file = Path(os.path.expanduser(dict_file))
        else:
            # 默认：项目根目录下 terms.txt
            self.dict_file = (
                Path(__file__).resolve().parent.parent / "terms.txt"
            )
        self.cache_seconds = cache_seconds
        self._cache_ts = 0.0
        self._terms: dict[str, str] = {}
        self._load(force=True)

    def _parse_lines(self, text: str, into: dict[str, str]) -> None:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                wrong, right = line.split("\t", 1)
            elif "," in line:
                wrong, right = line.split(",", 1)
            else:
                logger.warning("词典行格式错误（需 Tab 或逗号分隔）: %s", line)
                continue
            wrong, right = wrong.strip(), right.strip()
            if wrong and right:
                into[wrong] = right

    def _load(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._cache_ts) < self.cache_seconds:
            return
        self._cache_ts = now

        terms = dict(DEFAULT_TERMS)
        if self.dict_file.exists():
            try:
                user_text = self.dict_file.read_text(encoding="utf-8")
                self._parse_lines(user_text, terms)
            except Exception as exc:
                logger.error("加载用户词典失败 %s: %s", self.dict_file, exc)
        self._terms = terms
        logger.info("术语词典已加载: %d 条（含用户词典）", len(terms))

    def reload_if_stale(self) -> None:
        """外部每次转录前调用：缓存过期则重载词典。"""
        self._load(force=False)

    def replace(self, text: str) -> str:
        if not text:
            return text
        self.reload_if_stale()
        # 最长 key 优先，避免短词先替换破坏长词匹配
        for wrong in sorted(self._terms, key=len, reverse=True):
            if wrong in text:
                text = text.replace(wrong, self._terms[wrong])
        return text

    def vocab_hints(self, limit: int = 40) -> list[str]:
        """返回词典中的正确写法（value 端），供 LLM 润色做领域词汇提示。

        去重、按长度降序、截取前 limit 条；加载失败返回空列表。
        """
        try:
            self.reload_if_stale()
        except Exception:  # noqa: BLE001
            return []
        seen: set[str] = set()
        hints: list[str] = []
        for correct in sorted(self._terms.values(), key=len, reverse=True):
            c = correct.strip()
            if c and c not in seen:
                seen.add(c)
                hints.append(c)
                if len(hints) >= limit:
                    break
        return hints
