# -*- coding: utf-8 -*-
"""
设置对话框（tkinter）

两个配置分区：
- General：听写快捷键、录音指示浮窗开关、文本输出方式、单次录音上限（秒）
- LLM 润色：enabled / schema(openai|anthropic) / style(chat|responses) /
  endpoint / api_key / model / temperature / timeout / 思考模式开关 /
  领域词汇 / 润色 prompt（支持 {TEXT} 与 {VOCAB} 占位符）

保存写回 JSON 配置文件；快捷键等部分设置需重启程序生效（对话框内有提示）。
线程模型：Tk root 唯一且归主线程（app.ui_hub.UIHub）；open_settings_dialog
可从托盘线程调用，仅向主线程事件泵派发构建任务，绝不跨线程触碰 Tk。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _default_config_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "vocotype_config.json")


def load_settings(path: str | None = None) -> dict:
    """读配置文件（不存在则返回 {}，由 DEFAULT_CONFIG 合并兜底）。"""
    path = path or _default_config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.error("读取设置失败: %s", exc)
        return {}


def collect_hotkey_settings(
    toggle: str,
    push_to_talk: str,
    toggle_enabled: bool,
    push_to_talk_enabled: bool,
) -> dict:
    return {
        "toggle": toggle.strip() or "f2",
        "toggle_enabled": bool(toggle_enabled),
        "push_to_talk": push_to_talk.strip().lower() or "none",
        "push_to_talk_enabled": bool(push_to_talk_enabled),
    }


def save_settings(cfg: dict, path: str | None = None) -> bool:
    path = path or _default_config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("保存设置失败: %s", exc)
        return False


def open_settings_dialog(
    config_path: str | None = None,
    llm_enabled_callback: Callable[[bool], None] | None = None,
) -> None:
    """打开设置窗口。

    线程模型：Tk root 唯一且在主线程（UIHub）。本函数可在托盘线程调用，
    仅向主线程事件泵派发一个"构建窗口"任务（after），不直接碰 Tk。
    """
    from app.ui_hub import get_hub

    hub = get_hub()
    if hub.root is None:
        hub.create_root()  # main() 启动时已建；这里兜底
    path = config_path or _default_config_path()

    def _open():
        try:
            global _current_window
            # 防重复：已有设置窗口则前置聚焦
            if _current_window is not None:
                try:
                    if _current_window.winfo_exists():
                        _current_window.root.deiconify()
                        _current_window.root.lift()
                        _current_window.root.focus_force()
                        return
                except Exception:
                    pass
            _current_window = _SettingsWindow(
                path,
                master=hub.root,
                llm_enabled_callback=llm_enabled_callback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("设置窗口打开失败: %s", exc)

    if not hub.dispatch(_open):
        logger.error("设置窗口无法打开：主线程 UI 事件泵未运行")


_current_window: "_SettingsWindow | None" = None


class _SettingsWindow:
    def __init__(
        self,
        config_path: str,
        master=None,
        llm_enabled_callback: Callable[[bool], None] | None = None,
    ):
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self._path = config_path
        self._llm_enabled_callback = llm_enabled_callback
        self._cfg = load_settings(config_path)
        self._dirty = False

        self.root = tk.Toplevel(master) if master is not None else tk.Tk()
        from app.appinfo import APP_NAME, app_version

        self.root.title(f"{APP_NAME} 设置 v{app_version()}")
        self.root.geometry("560x820")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # ---------- General tab ----------
        g = ttk.Frame(nb, padding=12)
        nb.add(g, text="常规")

        hotkeys = self._cfg.get("hotkeys", {})
        ttk.Label(g, text="听写快捷键（如 f2 / ctrl+alt+v，重启生效）:").grid(row=0, column=0, sticky="w", pady=4)
        self.e_hotkey = ttk.Entry(g, width=18)
        self.e_hotkey.insert(0, str(hotkeys.get("toggle", "f2")))
        self.e_hotkey.grid(row=0, column=1, sticky="w", pady=4)
        self.var_toggle_enabled = tk.BooleanVar(value=bool(hotkeys.get("toggle_enabled", True)))
        ttk.Checkbutton(g, text="启用听写快捷键", variable=self.var_toggle_enabled).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(g, text="按住说话热键（按住录音、松开停止；none 禁用）:").grid(row=2, column=0, sticky="w", pady=4)
        self.e_ptt = ttk.Entry(g, width=18)
        self.e_ptt.insert(0, str(hotkeys.get("push_to_talk", "win+ctrl+alt")))
        self.e_ptt.grid(row=2, column=1, sticky="w", pady=4)
        self.var_ptt_enabled = tk.BooleanVar(value=bool(hotkeys.get("push_to_talk_enabled", True)))
        ttk.Checkbutton(g, text="启用按住说话快捷键", variable=self.var_ptt_enabled).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(g, text="悬浮胶囊按钮始终可用（按住说话或点击切换）", foreground="#555").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=4)

        self.var_indicator = tk.BooleanVar(value=bool(self._cfg.get("ui", {}).get("show_indicator", True)))
        ttk.Checkbutton(g, text="显示录音指示浮窗（立即生效）", variable=self.var_indicator).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(g, text="文本输出方式:").grid(row=6, column=0, sticky="w", pady=4)
        self.cb_output = ttk.Combobox(g, width=15, state="readonly",
                                      values=["auto", "type", "clipboard", "unicode"])
        self.cb_output.set(str(self._cfg.get("output", {}).get("method", "auto")))
        self.cb_output.grid(row=6, column=1, sticky="w", pady=4)

        ttk.Label(g, text="单次录音上限（秒，防误触长录音）:").grid(row=7, column=0, sticky="w", pady=4)
        self.e_maxsec = ttk.Entry(g, width=10)
        self.e_maxsec.insert(0, str(int(self._cfg.get("audio", {}).get("max_session_bytes", 2 * 1024 * 1024) / 32000)))
        self.e_maxsec.grid(row=7, column=1, sticky="w", pady=4)

        ttk.Label(g, text="麦克风输入增益（1~20，音量小调大）:").grid(row=8, column=0, sticky="w", pady=4)
        self.e_gain = ttk.Entry(g, width=10)
        self.e_gain.insert(0, str(float(self._cfg.get("audio", {}).get("gain", 12.0))))
        self.e_gain.grid(row=8, column=1, sticky="w", pady=4)

        # 麦克风设备选择（重启生效）
        from .audio_capture import list_input_devices
        self._mic_devices = list_input_devices()
        self._mic_values = ["系统默认输入设备（跟随 Windows 设置）"]
        self._mic_map = {}  # 显示名 -> 设备编号
        for d in self._mic_devices:
            tag = "（系统默认）" if d["default"] else ""
            display = f"#{d['index']} {d['name']}{tag}"
            self._mic_values.append(display)
            self._mic_map[display] = d["index"]
        cur = self._cfg.get("audio", {}).get("device")
        ttk.Label(g, text="当前麦克风（重启生效）:").grid(row=9, column=0, sticky="w", pady=4)
        self.cb_mic = ttk.Combobox(g, width=48, state="readonly", values=self._mic_values)
        if cur is None:
            self.cb_mic.set(self._mic_values[0])
        else:
            try:
                cur_idx = int(cur)
            except (TypeError, ValueError):
                cur_idx = None
            display_hit = next((v for v, i in self._mic_map.items() if i == cur_idx), None)
            if display_hit:
                self.cb_mic.set(display_hit)
            else:
                # 配置里的设备当前未检测到（如 USB 麦克风已拔出）——保留原值并提示
                missing = f"#{cur}（未检测到，启动时将回退）"
                self._mic_values.append(missing)
                self.cb_mic["values"] = self._mic_values
                self.cb_mic.set(missing)
        self.cb_mic.grid(row=9, column=1, sticky="w", pady=4)

        self.var_save_wav = tk.BooleanVar(value=bool(self._cfg.get("audio", {}).get("save_recordings", False)))
        ttk.Checkbutton(g, text="保存录音 WAV 到 logs/（默认关闭；仅供调试，开启后磁盘会持续增长，重启生效）",
                        variable=self.var_save_wav).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Button(g, text="一键清理日志与录音文件…", command=self._clean_house).grid(
            row=11, column=0, columnspan=2, sticky="w", pady=(10, 0))

        for c in range(2):
            g.columnconfigure(c, weight=1 if c else 0)

        # ---------- LLM tab ----------
        llm = self._cfg.get("llm", {})
        f = ttk.Frame(nb, padding=12)
        nb.add(f, text="LLM 润色")

        self.var_llm_enabled = tk.BooleanVar(value=bool(llm.get("enabled", False)))
        ttk.Checkbutton(f, text="启用 LLM 二次润色（立即生效；失败自动降级）",
                        variable=self.var_llm_enabled).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(f, text="协议 schema:").grid(row=1, column=0, sticky="w", pady=4)
        self.cb_schema = ttk.Combobox(f, width=15, state="readonly", values=["openai", "anthropic"])
        self.cb_schema.set(str(llm.get("schema", "openai")))
        self.cb_schema.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(f, text="协议 schema:").grid(row=1, column=0, sticky="w", pady=4)
        self.cb_schema = ttk.Combobox(f, width=15, state="readonly", values=["openai", "anthropic"])
        self.cb_schema.set(str(llm.get("schema", "openai")))
        self.cb_schema.grid(row=1, column=1, sticky="w", pady=4)

        # 端点风格：openai = chat(经典 /chat/completions) | responses(新 /v1/responses)；
        # anthropic = Messages API（/v1/messages），没有风格可选，自动置灰
        self._saved_style = str(llm.get("style", "chat"))
        self.lbl_style = ttk.Label(f, text="OpenAI 端点风格:")
        self.lbl_style.grid(row=2, column=0, sticky="w", pady=4)
        self.cb_style = ttk.Combobox(f, width=15, state="readonly", values=["chat", "responses"])
        self.cb_style.set(self._saved_style if self._saved_style in ("chat", "responses") else "chat")
        self.cb_style.grid(row=2, column=1, sticky="w", pady=4)
        self.cb_schema.bind("<<ComboboxSelected>>", self._update_style_state)
        self._update_style_state()  # 按当前 schema 初始化（anthropic 时置灰）

        ttk.Label(f, text="Endpoint URL:").grid(row=3, column=0, sticky="w", pady=4)
        self.e_endpoint = ttk.Entry(f, width=48)
        self.e_endpoint.insert(0, str(llm.get("endpoint", "")))
        self.e_endpoint.grid(row=3, column=1, sticky="we", pady=4)

        ttk.Label(f, text="API Key:").grid(row=4, column=0, sticky="w", pady=4)
        self.e_apikey = ttk.Entry(f, width=48, show="*")
        self.e_apikey.insert(0, str(llm.get("api_key", "")))
        self.e_apikey.grid(row=4, column=1, sticky="we", pady=4)

        ttk.Label(f, text="模型名称:").grid(row=5, column=0, sticky="w", pady=4)
        self.e_model = ttk.Entry(f, width=30)
        self.e_model.insert(0, str(llm.get("model", "")))
        self.e_model.grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(f, text="temperature (0~1):").grid(row=6, column=0, sticky="w", pady=4)
        self.e_temp = ttk.Entry(f, width=10)
        self.e_temp.insert(0, str(float(llm.get("temperature", 0.3))))
        self.e_temp.grid(row=6, column=1, sticky="w", pady=4)

        ttk.Label(f, text="超时（秒）:").grid(row=7, column=0, sticky="w", pady=4)
        self.e_timeout = ttk.Entry(f, width=10)
        self.e_timeout.insert(0, str(float(llm.get("timeout_seconds", 15))))
        self.e_timeout.grid(row=7, column=1, sticky="w", pady=4)

        self.var_no_thinking = tk.BooleanVar(value=bool(llm.get("disable_thinking", True)))
        ttk.Checkbutton(
            f, text="关闭模型思考模式（DeepSeek V4 默认开启 thinking，关闭后润色从十几秒降到 1~3 秒）",
            variable=self.var_no_thinking,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(f, text="领域词汇（逗号分隔，注入提示帮助纠同音错字）:").grid(row=9, column=0, sticky="w", pady=4)
        self.e_vocab = ttk.Entry(f, width=48)
        vocab_val = llm.get("vocabulary", "")
        if isinstance(vocab_val, (list, tuple)):
            vocab_val = ", ".join(str(x) for x in vocab_val)
        self.e_vocab.insert(0, str(vocab_val))
        self.e_vocab.grid(row=9, column=1, sticky="we", pady=4)

        ttk.Label(f, text="润色 Prompt（占位符：{TEXT}=原文本，{VOCAB}=词汇提示段）:").grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.txt_prompt = tk.Text(f, width=66, height=10, font=("Consolas", 9), wrap="word")
        self.txt_prompt.grid(row=11, column=0, columnspan=2, sticky="we", pady=4)
        prompt_val = str(llm.get("prompt", "")).strip()
        self.txt_prompt.insert("1.0", prompt_val if prompt_val else "（留空使用内置默认 Prompt）")
        ttk.Button(f, text="恢复默认 Prompt", width=16, command=self._reset_prompt).grid(
            row=12, column=1, sticky="e", pady=(0, 4))

        ttk.Label(f, text="DeepSeek 快捷填法：schema=openai（Chat Completions）→ https://api.deepseek.com/v1/chat/completions；"
                          "schema=anthropic（Messages API）→ https://api.deepseek.com/anthropic/v1/messages",
                  wraplength=460, foreground="#666").grid(row=13, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # ---------- 网络（内网代理 / 证书） ----------
        sep = ttk.Separator(f, orient="horizontal")
        sep.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(12, 6))
        ttk.Label(f, text="网络设置（代理 / SSL，重启程序生效）", font=("", 9, "bold")).grid(
            row=15, column=0, columnspan=2, sticky="w")

        ttk.Label(f, text="HTTP 代理（如 http://proxy.corp:8080）:").grid(row=16, column=0, sticky="w", pady=4)
        self.e_http_proxy = ttk.Entry(f, width=40)
        self.e_http_proxy.insert(0, str(llm.get("http_proxy", "")))
        self.e_http_proxy.grid(row=16, column=1, sticky="we", pady=4)

        ttk.Label(f, text="HTTPS 代理（如 http://proxy.corp:8080）:").grid(row=17, column=0, sticky="w", pady=4)
        self.e_https_proxy = ttk.Entry(f, width=40)
        self.e_https_proxy.insert(0, str(llm.get("https_proxy", "")))
        self.e_https_proxy.grid(row=17, column=1, sticky="we", pady=4)

        self.var_no_ssl = tk.BooleanVar(value=bool(llm.get("disable_ssl_verify", False)))
        ttk.Checkbutton(
            f, text="跳过 SSL 证书校验（自签名证书/内网网关场景；降低传输安全性，慎用）",
            variable=self.var_no_ssl,
        ).grid(row=18, column=0, columnspan=2, sticky="w", pady=4)

        for c in range(2):
            f.columnconfigure(c, weight=1 if c else 0)

        # ---------- buttons ----------
        btns = ttk.Frame(self.root, padding=(8, 4))
        btns.pack(fill="x")
        ttk.Button(btns, text="测试 LLM 连接", command=self._test_llm).pack(side="left", padx=4)
        ttk.Button(btns, text="保存", command=self._save).pack(side="right", padx=4)
        ttk.Button(btns, text="取消", command=self.root.destroy).pack(side="right", padx=4)

        self._msg = ttk.Label(self.root, text="", foreground="#0a7d32", padding=(10, 0))
        self._msg.pack(fill="x")

    # ---------- actions ----------

    def _update_style_state(self, *_args):
        """schema 联动：anthropic 使用其 Messages API，无端点风格可选（置灰）。"""
        if self.cb_schema.get() == "anthropic":
            self.lbl_style.config(text="端点风格（Anthropic Messages，无需选择）:")
            self.cb_style.configure(values=["messages"])
            self.cb_style.set("messages")
            self.cb_style.state(["disabled"])
        else:
            self.lbl_style.config(text="OpenAI 端点风格:")
            self.cb_style.state(["!disabled"])
            self.cb_style.configure(values=["chat", "responses"])
            if self.cb_style.get() not in ("chat", "responses"):
                self.cb_style.set(self._saved_style if self._saved_style in ("chat", "responses") else "chat")

    def winfo_exists(self) -> bool:
        try:
            return bool(self.root.winfo_exists())
        except Exception:
            return False

    def _on_close(self):
        global _current_window
        try:
            self.root.destroy()
        except Exception:
            pass
        _current_window = None

    def _reset_prompt(self):
        """清空 prompt 编辑框（保存时留空 = 使用内置默认）。"""
        self.txt_prompt.delete("1.0", "end")
        self.txt_prompt.insert("1.0", "（留空使用内置默认 Prompt）")

    def _clean_house(self):
        """一键清理 logs/ 下所有录音 WAV 与日志文件（当天的活跃日志被占用时改为清空）。"""
        from tkinter import messagebox

        log_dir = Path(_default_config_path()).parent / "logs"
        if not log_dir.is_dir():
            messagebox.showinfo("清理", f"目录不存在：{log_dir}")
            return

        wavs = sorted(log_dir.glob("*.wav"))
        logs = sorted(log_dir.glob("*.log*"))
        total = sum(p.stat().st_size for p in wavs + logs if p.exists())
        if not wavs and not logs:
            messagebox.showinfo("清理", "没有可清理的日志或录音文件。")
            return

        if not messagebox.askyesno(
            "确认清理",
            f"将删除 {len(wavs)} 个录音 WAV + {len(logs)} 个日志文件，"
            f"共 {total / 1024 / 1024:.1f} MB。\n\n"
            "此操作不可恢复，确定继续吗？",
        ):
            return

        # 删除可能耗时较长（文件被占用/杀软扫描），放后台线程执行，避免卡死 UI 主线程
        self._msg.config(text="正在清理…", foreground="#555")
        targets = wavs + logs

        def _worker():
            deleted, failed = 0, 0
            for p in targets:
                try:
                    p.unlink()
                    deleted += 1
                except OSError:
                    # 当天日志被运行中的程序占用，Windows 不允许删除 → 改为清空内容
                    try:
                        with open(p, "w", encoding="utf-8"):
                            pass
                        deleted += 1
                    except OSError:
                        failed += 1
            msg = f"已清理 {deleted} 个文件，释放 {total / 1024 / 1024:.1f} MB。"
            if failed:
                msg += f" {failed} 个文件清理失败（仍被占用）。"
            color = "#0a7d32" if not failed else "#b8860b"
            from app.ui_hub import get_hub
            get_hub().dispatch(lambda: self._msg.config(text=msg, foreground=color))

        threading.Thread(target=_worker, daemon=True, name="CleanHouse").start()

    def _collect(self) -> dict:
        cfg = dict(self._cfg)
        cfg["hotkeys"] = collect_hotkey_settings(
            self.e_hotkey.get(),
            self.e_ptt.get(),
            self.var_toggle_enabled.get(),
            self.var_ptt_enabled.get(),
        )
        cfg.setdefault("ui", {})["show_indicator"] = bool(self.var_indicator.get())
        cfg.setdefault("output", {})["method"] = self.cb_output.get()
        try:
            maxsec = max(10, min(300, int(float(self.e_maxsec.get()))))
        except Exception:
            maxsec = 60
        cfg.setdefault("audio", {})["max_session_bytes"] = maxsec * 32000
        try:
            gain = max(1.0, min(30.0, float(self.e_gain.get())))
        except Exception:
            gain = 12.0
        cfg["audio"]["gain"] = gain
        cfg["audio"]["save_recordings"] = bool(self.var_save_wav.get())
        # 麦克风选择：选了具体设备则存编号，选默认则存 None
        mic_sel = self.cb_mic.get()
        if mic_sel in self._mic_map:
            cfg["audio"]["device"] = self._mic_map[mic_sel]
        else:
            cfg["audio"]["device"] = None

        try:
            temp = max(0.0, min(1.0, float(self.e_temp.get())))
        except Exception:
            temp = 0.3
        try:
            timeout = max(3.0, min(120.0, float(self.e_timeout.get())))
        except Exception:
            timeout = 15.0
        prompt_text = self.txt_prompt.get("1.0", "end").strip()
        if "留空使用内置默认" in prompt_text:
            prompt_text = ""
        style = self.cb_style.get()
        if style not in ("chat", "responses"):
            # anthropic 下的占位值不落盘；保留原有效值或回退 chat
            style = self._saved_style if self._saved_style in ("chat", "responses") else "chat"
        cfg["llm"] = {
            "enabled": bool(self.var_llm_enabled.get()),
            "schema": self.cb_schema.get(),
            "style": style,
            "endpoint": self.e_endpoint.get().strip(),
            "api_key": self.e_apikey.get().strip(),
            "model": self.e_model.get().strip(),
            "temperature": temp,
            "timeout_seconds": timeout,
            "disable_thinking": bool(self.var_no_thinking.get()),
            "vocabulary": self.e_vocab.get().strip(),
            "prompt": prompt_text,
            "http_proxy": self.e_http_proxy.get().strip(),
            "https_proxy": self.e_https_proxy.get().strip(),
            "disable_ssl_verify": bool(self.var_no_ssl.get()),
        }
        return cfg

    def _save(self):
        cfg = self._collect()
        if save_settings(cfg, self._path):
            self._dirty = False
            if self._llm_enabled_callback is not None:
                self._llm_enabled_callback(bool(cfg["llm"]["enabled"]))
            self._msg.config(text="已保存。LLM 开关立即生效；快捷键等设置需重启程序生效。")
        else:
            self._msg.config(text="保存失败，请检查文件权限。", foreground="#c0392b")

    def _test_llm(self):
        cfg = self._collect().get("llm", {})
        cfg["enabled"] = True
        self._msg.config(text="正在测试连接…", foreground="#555")

        def _worker():
            from app.llm_polish import LLMPolisher

            p = LLMPolisher(cfg)
            result = p.polish("ui 县城架构 tcl 构件连 root的 after 都不允许跨线场调用")
            ok = result is not None

            def _apply():
                if ok:
                    self._msg.config(text=f"连接成功！返回: {result[:40]}", foreground="#0a7d32")
                else:
                    self._msg.config(text="连接失败（endpoint/key/网络），详见日志。", foreground="#c0392b")

            # 后台线程不碰 Tk：经 UIHub 队列派发到主线程执行
            from app.ui_hub import get_hub
            get_hub().dispatch(_apply)

        threading.Thread(target=_worker, daemon=True).start()

    def run(self):
        self.root.mainloop()
