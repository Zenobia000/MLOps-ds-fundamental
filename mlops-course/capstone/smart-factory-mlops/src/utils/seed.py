"""亂數種子統一設定（可重現性的單一入口）。

一次把 ``random`` / ``numpy`` / ``torch``（含 CUDA）的種子設好，
確保「同 seed → 同結果」。torch 為選用依賴：未安裝時自動略過其設定，
讓純表格 / 時序流程在無 torch 環境也能跑。

用法::

    from src.utils.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """設定全域亂數種子。

    Args:
        seed: 種子值（通常來自 ``conf/config.yaml`` 的 ``seed``）。
        deterministic: 為 True 時，要求 torch 使用確定性演算法
            （可能略降效能，但確保結果可重現）。

    Returns:
        實際套用的 seed（方便呼叫端記錄 / log）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("未安裝 torch，略過 torch 種子設定（表格 / 時序流程不受影響）。")

    logger.info("已設定亂數種子 seed=%d（deterministic=%s）", seed, deterministic)
    return seed
