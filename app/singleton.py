# -*- coding: utf-8 -*-
"""
单实例锁（防止多实例并发导致重复录音/重复注入文字）

原理：绑定本地回环地址的固定端口。若端口已被占用说明已有实例在跑，
第二个实例直接退出。Windows/Linux 通用，零依赖。
"""
from __future__ import annotations

import logging
import socket
import sys

logger = logging.getLogger(__name__)

LOCK_HOST = "127.0.0.1"
LOCK_PORT = 57612  # Bibi 单实例锁端口，避开常用端口段

_sock: socket.socket | None = None


def acquire_single_instance_lock() -> bool:
    """尝试获取单实例锁。成功返回 True；已有实例运行时返回 False。"""
    global _sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((LOCK_HOST, LOCK_PORT))
        s.listen(1)  # 占住端口
        _sock = s
        logger.info("单实例锁获取成功 (port=%d)", LOCK_PORT)
        return True
    except OSError:
        logger.error(
            "检测到另一个 Bibi 实例已在运行（端口 %d 被占用）。"
            "请先关闭旧实例（任务管理器结束 python.exe / main.py），再启动。",
            LOCK_PORT,
        )
        print(
            "[Bibi] 已有实例在运行！请先关闭旧的 Bibi 窗口再启动新实例。",
            file=sys.stderr,
        )
        return False
