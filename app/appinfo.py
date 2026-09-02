# -*- coding: utf-8 -*-
"""应用全局信息：名称与版本号（读项目根 VERSION 文件，带缓存）。"""
from __future__ import annotations

import os

APP_NAME = "Bibi"

_VERSION_CACHE: str | None = None


def app_version() -> str:
    """读取项目根目录 VERSION 文件（缺失时回退 0.0.1）。"""
    global _VERSION_CACHE
    if _VERSION_CACHE is None:
        try:
            vfile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
            with open(vfile, "r", encoding="utf-8") as f:
                _VERSION_CACHE = f.read().strip() or "0.0.1"
        except Exception:
            _VERSION_CACHE = "0.0.1"
    return _VERSION_CACHE
