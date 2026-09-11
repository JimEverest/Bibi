# -*- coding: utf-8 -*-
"""
LLM 二次润色模块（方案5）

功能：ASR 转写 + 词典替换之后，可选地调用 LLM 对文本做"最小化润色"：
- 纠正 ASR 同音/近音错字（重点，结合领域词汇提示）
- 去除口癖/填充词（呃、嗯、就是说……）
- 合并无意义重复（"我我我觉得" -> "我觉得"）
- 整合自我修正（"周三开会，不对，是周四" -> "周四开会"）
- 保留说话人语气与风格，绝不改写句式

设计要点：
- 双协议：schema = "openai"（/chat/completions，兼容 vLLM/Ollama/OneAPI）
  或 "anthropic"（/v1/messages）
- 思考模式控制：DeepSeek V4 等新模型默认开启 thinking（effort=high），
  会导致润色延迟高达十几秒甚至超时。默认发送 disable 参数
  （OpenAI: thinking.type=disabled；Anthropic: reasoning.effort=none），
  可通过配置 llm.disable_thinking=false 关回（并可选 llm.reasoning_effort）
- 纯标准库实现（urllib），不引入 requests 依赖
- 失败永不阻塞主流程：超时/网络错误/HTTP错误 -> 返回 None，调用方降级用原文本
- prompt 参考了 ququ（蛐蛐）项目的 optimize prompt 设计（Apache 2.0），
  支持 {TEXT} 与 {VOCAB} 两个占位符
"""
from __future__ import annotations

import json
import logging
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class _LLMFailure(RuntimeError):
    def __init__(self, category: str, detail: str):
        self.category = category
        self.detail = detail
        super().__init__(detail)

    def __str__(self) -> str:
        return f"{self.category}: {self.detail}"


# 润色 prompt（结构参考 ququ 项目 optimize 模板，按中英混合 dictation 场景增补）
# 占位符：{TEXT} = 待润色文本；{VOCAB} = 领域词汇提示（无则整段为空）
DEFAULT_PROMPT = """# 角色与目标
你是一个专业的语音转录文本优化助手。输入文本由 ASR（自动语音识别）实时转写生成，说话人可能在口述技术内容、工作记录或日常事务。你的任务：做最小化润色，去掉语音转写噪音，100% 保留说话人的原始意图、用词和语气。文本可能是中文、英文或中英文混合。

# 最高优先级：ASR 同音/近音错字纠正
ASR 最常见的错误是同音字和近音词，必须结合上下文语义判断并纠正。典型模式：
- 专业术语被写成日常同音词：如"线程"误作"县城/县程/线成"，"跨线程"误作"跨线场"，"队列"误作"对列"，"推理"误作"推礼"。
- 英文技术词被音译或拆碎：如 "queue" 误作 "q/cue"，"route" 误作"路由/绕"，"ONNX" 误作"昂艾克斯"。
- 中英文交界处的粘连、漏字、错字：如 "连 root 的" 误作 "构件连 route的"。
纠正原则：先判断整句在说什么事，再把发音相同/相近、但语义不合的词换成语境下唯一合理的写法。若词汇提示表（若有）中存在发音相近的词条，优先采用。

# 核心原则
- **最小化修改**：只处理明确的、非内容性的言语错误。
- **保留原貌**：最大限度地保留用户的原始用词、句式和语气。
- **歧义时保守**：当不确定是否需要修改时，保持原样。

# 明确的优化指令 (Do's)
1.  **纠正同音错字与 ASR 误识别**（见上，这是最重要的任务）。
2.  **修正被错误拆分或拼错的英文单词**，中英文之间补空格。
3.  **移除无意义的填充词**：删除如"呃"、"嗯"、"啊这"、"那个"、"就是说"、"like"、"um"、"uh" 等不承载实际信息的词。
4.  **合并无意义重复**："我我我觉得" -> "我觉得"；"这个这个方案" -> "这个方案"。
5.  **整合自我修正**："会议定在周三，呃不对，是周四" -> "会议定在周四"。
6.  **补全标点**：按中英文各自习惯补全标点；英文单词内部绝不能加标点或空格。

# 严格的禁止项 (Don'ts)
1.  **禁止风格转换**：绝不能把口语化表达改成书面语。
2.  **禁止替换用词**：除明显的错别字/ASR误识别外，不能改变用户的用词选择。
3.  **禁止改变句式**：不能重组句子结构（如主动改被动）。
4.  **禁止增删情感语气词**："啊"、"呀"、"呢"、"吧"、"嘛"、"哦"等一律保留。
5.  **禁止主观臆断**：不添加原文不存在的信息。
6.  **禁止翻译**：中英文混合输入必须保持原有语言构成。
{VOCAB}
# 输出
直接返回优化后的文本，不要任何解释、前言、总结或代码块标记。

原始文本：
```
{TEXT}
```"""

_VOCAB_SECTION = """
# 词汇提示（说话人常用领域词，供同音纠错参考）
以下词语按发音/语义与原文对照时优先采用：
{VOCAB}
"""


class LLMPolisher:
    """调用 LLM 做二次润色。所有失败路径都返回 None（调用方降级）。"""

    def __init__(self, llm_config: dict):
        self._enabled_lock = threading.Lock()
        self._enabled = bool(llm_config.get("enabled", False))
        self.endpoint = str(llm_config.get("endpoint", "")).rstrip("/")
        self.schema = str(llm_config.get("schema", "openai")).lower()
        self.api_key = str(llm_config.get("api_key", ""))
        self.model = str(llm_config.get("model", ""))
        self.timeout = float(llm_config.get("timeout_seconds", 15))
        self.max_tokens = int(llm_config.get("max_tokens", 512))
        self.temperature = float(llm_config.get("temperature", 0.3))
        prompt = str(llm_config.get("prompt", "")).strip()
        self.prompt_template = prompt if prompt else DEFAULT_PROMPT
        # 思考模式控制：新模型（DeepSeek V4 等）默认 thinking=enabled + effort=high，
        # 润色场景要快，默认关闭
        self.disable_thinking = bool(llm_config.get("disable_thinking", True))
        self.reasoning_effort = str(llm_config.get("reasoning_effort", "")).strip()
        # 配置里的固定领域词汇（逗号/顿号分隔字符串或列表）
        vocab = llm_config.get("vocabulary", "")
        if isinstance(vocab, str):
            self.config_vocab = [w.strip() for w in re.split(r"[,，、;；\n]", vocab) if w.strip()]
        elif isinstance(vocab, (list, tuple)):
            self.config_vocab = [str(w).strip() for w in vocab if str(w).strip()]
        else:
            self.config_vocab = []
        # /v1/responses 风格（OpenAI 新 API）还是 /chat/completions（经典）
        self.style = str(llm_config.get("style", "chat")).lower()  # chat|responses
        # ---- 网络设置（内网代理/自签名证书环境）----
        self.http_proxy = str(llm_config.get("http_proxy", "")).strip()
        self.https_proxy = str(llm_config.get("https_proxy", "")).strip()
        self.disable_ssl_verify = bool(llm_config.get("disable_ssl_verify", False))

    # ---------- public ----------

    @property
    def enabled(self) -> bool:
        with self._enabled_lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._enabled_lock:
            self._enabled = bool(value)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def is_enabled(self) -> bool:
        return self.enabled

    def polish(self, text: str, extra_vocab: list[str] | None = None) -> str | None:
        """润色文本。任何失败返回 None。disabled 也返回 None。

        extra_vocab: 调用方（如术语词典）提供的领域词汇提示，与配置里的
        vocabulary 合并注入 prompt 的 {VOCAB} 占位符。
        """
        if not self.is_enabled() or not text or not text.strip():
            return None
        if not self.endpoint or not self.model:
            logger.warning("LLM 润色已启用但 endpoint/model 未配置，跳过润色")
            return None
        vocab_section = self._build_vocab_section(extra_vocab)
        prompt = self.prompt_template.replace("{VOCAB}", vocab_section).replace("{TEXT}", text)
        try:
            if self.schema == "anthropic":
                return self._call_anthropic(prompt)
            return self._call_openai(prompt)
        except _LLMFailure as exc:
            logger.warning(
                "LLM 润色失败（降级为原文本）: category=%s detail=%s",
                exc.category,
                exc.detail,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM 润色失败（降级为原文本）: category=unexpected "
                "error_type=%s detail=%s",
                type(exc).__name__,
                exc,
            )
            return None

    # ---------- protocols ----------

    def _build_vocab_section(self, extra_vocab: list[str] | None) -> str:
        seen: set[str] = set()
        items: list[str] = []
        for w in (extra_vocab or []) + self.config_vocab:
            w = w.strip()
            if w and w not in seen and len(w) >= 2:  # 排除单字符噪音
                seen.add(w)
                items.append(w)
        if not items:
            return ""
        return _VOCAB_SECTION.format(VOCAB="\n".join(f"- {w}" for w in items[:40]))

    def _http_post_json(self, url: str, headers: dict, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        safe_url = self._safe_url(url)
        started = time.monotonic()
        headers_received = False
        logger.info(
            "LLM HTTP请求开始: method=POST endpoint=%s timeout=%.1fs body_bytes=%d",
            safe_url,
            self.timeout,
            len(data),
        )
        # 代理：按 URL 协议选择 http_proxy / https_proxy（留空 = 直连）
        proxy = self.https_proxy if url.lower().startswith("https") else self.http_proxy
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        if self.disable_ssl_verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=ctx))
        opener = urllib.request.build_opener(*handlers) if handlers else urllib.request.build_opener()
        req = urllib.request.Request(url, data=data, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        req.add_header("Content-Type", "application/json")
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                headers_received = True
                status = getattr(resp, "status", None) or resp.getcode()
                logger.debug(
                    "LLM HTTP响应头已收到: endpoint=%s status=%s elapsed=%.2fs",
                    safe_url,
                    status,
                    time.monotonic() - started,
                )
                payload = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            payload = self._read_error_body(exc)
            elapsed = time.monotonic() - started
            logger.warning(
                "LLM HTTP请求失败: category=http endpoint=%s status=%s "
                "elapsed=%.2fs response=%s",
                safe_url,
                exc.code,
                elapsed,
                self._preview(payload),
            )
            raise _LLMFailure(
                "http",
                f"status={exc.code} reason={exc.reason!s} elapsed={elapsed:.2f}s "
                f"response={self._preview(payload)}",
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            elapsed = time.monotonic() - started
            phase = "response_body" if headers_received else "connect_or_response_headers"
            logger.warning(
                "LLM HTTP请求超时: category=timeout endpoint=%s phase=%s "
                "elapsed=%.2fs timeout=%.1fs error=%s",
                safe_url,
                phase,
                elapsed,
                self.timeout,
                exc,
            )
            raise _LLMFailure(
                "timeout",
                f"phase={phase} elapsed={elapsed:.2f}s limit={self.timeout:.1f}s",
            ) from exc
        except urllib.error.URLError as exc:
            elapsed = time.monotonic() - started
            logger.warning(
                "LLM HTTP请求失败: category=transport endpoint=%s elapsed=%.2fs "
                "reason=%s",
                safe_url,
                elapsed,
                exc.reason,
            )
            raise _LLMFailure(
                "transport",
                f"elapsed={elapsed:.2f}s reason={exc.reason!s}",
            ) from exc
        except OSError as exc:
            elapsed = time.monotonic() - started
            logger.warning(
                "LLM HTTP请求失败: category=os_error endpoint=%s elapsed=%.2fs "
                "error_type=%s detail=%s",
                safe_url,
                elapsed,
                type(exc).__name__,
                exc,
            )
            raise _LLMFailure(
                "os_error",
                f"elapsed={elapsed:.2f}s error_type={type(exc).__name__} detail={exc}",
            ) from exc

        elapsed = time.monotonic() - started
        logger.info(
            "LLM HTTP请求完成: endpoint=%s status=%s elapsed=%.2fs response_bytes=%d",
            safe_url,
            status,
            elapsed,
            len(payload.encode("utf-8")),
        )
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM响应解析失败: category=json endpoint=%s elapsed=%.2fs "
                "response=%s",
                safe_url,
                elapsed,
                self._preview(payload),
            )
            raise _LLMFailure(
                "json",
                f"elapsed={elapsed:.2f}s line={exc.lineno} column={exc.colno} "
                f"response={self._preview(payload)}",
            ) from exc
        if not isinstance(result, dict):
            raise _LLMFailure(
                "response",
                f"expected=object actual={type(result).__name__} elapsed={elapsed:.2f}s",
            )
        return result

    @staticmethod
    def _safe_url(url: str) -> str:
        parts = urlsplit(url)
        host = parts.hostname or "<invalid-host>"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme or '<invalid-scheme>'}://{host}{parts.path or '/'}"

    @staticmethod
    def _preview(payload: str, limit: int = 500) -> str:
        compact = " ".join(payload.split())
        return compact[:limit] + ("..." if len(compact) > limit else "")

    @staticmethod
    def _read_error_body(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return "<unavailable>"

    def _thinking_fields(self, schema: str) -> dict:
        """按 schema 生成思考模式控制字段。

        实测结论（2026-09，DeepSeek V4）：
        - 两个端点的关闭写法相同：thinking.type=disabled（官方文档中
          Anthropic 格式的 reasoning.effort=none 实测被网关拒绝 502）
        - V4 默认 thinking=enabled + effort=high，必须显式关闭
        - 开启状态：openai 端点用 reasoning_effort；anthropic 端点省略字段
          （端点默认即 thinking 开 + effort high）
        """
        if self.disable_thinking:
            return {"thinking": {"type": "disabled"}}
        effort = self.reasoning_effort or "high"
        if schema == "openai":
            return {"reasoning_effort": effort}
        return {}

    def _call_openai(self, prompt: str) -> str | None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.style == "responses":
            # OpenAI /v1/responses 风格
            body = {
                "model": self.model,
                "input": prompt,
                "temperature": self.temperature,
            }
            body.update(self._thinking_fields("openai"))
            data = self._http_post_json(self.endpoint, headers, body)
            # 兼容两种返回结构
            out = data.get("output_text")
            if out:
                return self._clean(str(out))
            for item in data.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and c.get("text"):
                        return self._clean(c["text"])
            raise _LLMFailure(
                "response",
                f"OpenAI responses 中没有 output_text: {self._preview_data(data)}",
            )
        # 经典 /chat/completions
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        body.update(self._thinking_fields("openai"))
        data = self._http_post_json(self.endpoint, headers, body)
        choices = data.get("choices") or []
        if not choices:
            raise _LLMFailure(
                "response",
                f"OpenAI 响应无 choices: {self._preview_data(data)}",
            )
        content = (choices[0].get("message") or {}).get("content", "")
        if not content:
            raise _LLMFailure(
                "response",
                f"OpenAI 响应 content 为空: {self._preview_data(data)}",
            )
        return self._clean(str(content))

    def _call_anthropic(self, prompt: str) -> str | None:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(self._thinking_fields("anthropic"))
        data = self._http_post_json(self.endpoint, headers, body)
        blocks = data.get("content") or []
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        joined = "".join(texts)
        if not joined:
            raise _LLMFailure(
                "response",
                f"Anthropic 响应无 text content: {self._preview_data(data)}",
            )
        return self._clean(joined)

    # ---------- helpers ----------

    @classmethod
    def _preview_data(cls, data: dict) -> str:
        try:
            payload = json.dumps(data, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            payload = str(data)
        return cls._preview(payload)

    @staticmethod
    def _clean(text: str) -> str:
        """去掉模型可能附带的代码块围栏与首尾空白。"""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()
