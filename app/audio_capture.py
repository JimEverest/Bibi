"""Audio capture utilities built on sounddevice."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


logger = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    """枚举可用输入设备，每个物理设备只保留一条（去重 + 过滤噪音）。

    Windows 会把同一支麦克风通过多套音频接口（MME/DirectSound/WASAPI/WDM-KS）
    各枚举一次；这里按 WASAPI > WDM-KS > MME > DirectSound 的优先级去重，
    并过滤掉 Sound Mapper 别名与扬声器回采（loopback）条目。
    每项: {"index": int, "name": str, "default": bool, "hostapi": str}
    index 可直接写入 config audio.device；default 标记系统默认输入设备。
    """
    try:
        try:
            default_input_idx = sd.query_devices(kind="input")["index"]
        except Exception:
            di = sd.default.device
            default_input_idx = di[0] if isinstance(di, (list, tuple)) else di
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
    except Exception as exc:
        logger.error("枚举音频设备失败: %s", exc)
        return []

    # 各 Host API 的优先级（数值越小越优先）
    api_prio: dict[int, int] = {}
    for rank, pref in enumerate(("WASAPI", "WDM-KS", "MME", "DirectSound")):
        for hai, h in enumerate(hostapis):
            if pref in str(h.get("name", "")):
                api_prio[hai] = rank
                break

    cands = []
    for idx, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        name = str(info.get("name", f"设备 #{idx}"))
        # 还原 Windows 驱动的间接字符串为友好名：
        # "Input (@System32\drivers\xxx.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iPhone))"
        #   → "Hands-Free HF Audio (iPhone)"
        if "@" in name and ";" in name:
            try:
                inner = name[name.index("@"):].rstrip(")")
                parts = inner.split(";")
                template = parts[1].replace("%1", "").replace("%0", "").strip()
                params = " ".join(p.strip().strip("()") for p in parts[2:] if p.strip().strip("()"))
                name = f"{template} ({params})" if params else template
            except Exception:
                pass
        low = name.lower()
        # 过滤：Sound Mapper / Primary Sound Capture 是默认设备别名（已有首项代替）；
        # Speaker 条目是扬声器回采，不能当麦克风；空名/无名条目不可辨认，跳过
        if "sound mapper" in low or "speaker" in low or "输出" in name:
            continue
        if "primary sound capture" in low or not name.strip() or name.strip().endswith("()"):
            continue
        hai = info.get("hostapi")
        cands.append({
            "index": idx,
            "name": name,
            "default": idx == default_input_idx,
            "hostapi": str(hostapis[hai]["name"]) if hai is not None else "",
            "_prio": api_prio.get(hai, 99),
        })

    # 按名称前缀归组去重（MME 会把设备名截断到 31 字符，取前缀可跨接口对上同一物理设备），
    # DirectSound 属遗留兼容层，直接排除；默认标记在组内传递（默认设备常是 MME 条目）
    groups: dict[str, dict] = {}
    for d in cands:
        if "DirectSound" in d["hostapi"]:
            continue
        key = d["name"][:16]
        cur = groups.get(key)
        if cur is None:
            groups[key] = d
        else:
            merged_default = cur["default"] or d["default"]
            if d["_prio"] < cur["_prio"]:
                groups[key] = d
            groups[key]["default"] = merged_default

    result = sorted(groups.values(), key=lambda d: (not d["default"], d["index"]))
    for d in result:
        d.pop("_prio", None)
    return result


def describe_device(device) -> str:
    """把 config 中的 device 值（None/编号）翻译成 user-friendly 描述。"""
    if device is None:
        return "系统默认输入设备"
    try:
        idx = int(device)
    except (TypeError, ValueError):
        return "系统默认输入设备"
    for d in list_input_devices():
        if d["index"] == idx:
            suffix = "（系统默认）" if d["default"] else ""
            return f"#{idx} {d['name']}{suffix}"
    # 不在精选列表里（可能是被过滤的接口别名）——查全量表并注明接口
    try:
        info = sd.query_devices(idx)
        if info and info.get("max_input_channels", 0) > 0:
            return f"#{idx} {info.get('name', '?')}（接口别名，建议重新选择）"
    except Exception:
        pass
    return f"#{idx}（未检测到该设备，将回退到可用输入设备）"


class AudioCaptureError(RuntimeError):
    """Raised when the audio capture stream cannot be started."""


class AudioCapture:
    """Capture audio frames from the default (or configured) microphone."""

    def __init__(
        self,
        sample_rate: int,
        block_ms: int,
        device: Optional[str] = None,
        queue_size: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self._stream: Optional[sd.RawInputStream] = None
        self._lock = threading.Lock()
        self._running = False

        self._block_size = int(self.sample_rate * self.block_ms / 1000)
        if self._block_size <= 0:
            raise ValueError("block_ms too small for selected sample rate")

    @property
    def queue(self) -> "queue.Queue[np.ndarray]":
        return self._queue

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

            self.flush()
            self._stream = self._create_stream(self.device)
            try:
                self._stream.start()
            except Exception:
                self._stream.close()
                self._stream = self._create_stream(self._fallback_device())
                self._stream.start()

            self._running = True
            logger.info(
                "音频采集已启动，采样率=%sHz，块大小=%s样本，设备=%s",
                self.sample_rate,
                self._block_size,
                self._stream.device,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return

            assert self._stream is not None
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._running = False
            logger.info("音频采集已停止")

    def flush(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _create_stream(self, device: Optional[str]) -> sd.RawInputStream:
        try:
            return sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self._block_size,
                dtype="int16",
                channels=1,
                callback=self._callback,
                device=device,
            )
        except Exception as exc:
            msg = f"无法创建音频输入流: {exc}"
            logger.error(msg)
            raise AudioCaptureError(msg) from exc

    def _fallback_device(self) -> Optional[int]:
        try:
            devices = sd.query_devices()
            for idx, info in enumerate(devices):
                if info.get("max_input_channels", 0) > 0:
                    logger.warning(
                        "回退至输入设备 #%s (%s)", idx, info.get("name", "unknown")
                    )
                    return idx
        except Exception as exc:
            logger.error("查询音频设备失败: %s", exc)
        return None

    def _callback(self, in_data, frames, time, status):  # type: ignore[override]
        if status:
            logger.warning("音频流状态: %s", status)

        frame = np.frombuffer(in_data, dtype=np.int16)
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            logger.warning("音频队列已满，丢弃音频帧")


