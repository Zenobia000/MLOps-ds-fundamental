"""結構化 logging 工具。

提供全專案統一的 logger 設定，避免各模組各自呼叫 ``logging.basicConfig``
造成格式不一致。預設輸出「時間 等級 模組 訊息」單行格式，方便 grep 與
之後接 log 聚合（如 Loki / CloudWatch）。

用法::

    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("training started", extra={"model": "xgboost"})

設計：
    - 等級可由環境變數 ``LOG_LEVEL`` 覆蓋（預設 INFO），不寫死。
    - 以 module 名為 logger 名，繼承同一 handler，避免重複輸出。
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 模組級旗標，確保 root handler 只設定一次（多次 import 不重複加 handler）。
_CONFIGURED = False


def _configure_root() -> None:
    """為 root logger 安裝一個 stdout StreamHandler（僅一次）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    # 清掉既有 handler，避免與第三方（如 mlflow）重複輸出。
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """取得已套用統一格式的 logger。

    Args:
        name: logger 名稱，慣例傳入 ``__name__``；None 時回傳 root logger。

    Returns:
        設定好的 :class:`logging.Logger`。
    """
    _configure_root()
    return logging.getLogger(name)
