"""BentoML runner 定義：把兩種模型包成可獨立排程的 runner。

設計重點：
    - 一個 runner = 一個可獨立縮放 / 批次化的推論單元。
    - 表格（XGBoost）與影像（ONNX ResNet）各一個 runner，service.py 再把兩者
      組成單一 API（見 service.py）。
    - **dynamic batching**：BentoML 會在毫秒級時間窗內把多個請求合併成一個 batch
      餵給模型，提升吞吐。下方以註解說明建議參數，實際值依壓測調整。

教學備註：
    在真實專案中，runner 通常由 ``bentoml.xgboost.get(...)`` /
    ``bentoml.onnx.get(...)`` 從 BentoML Model Store 取得已 ``save_model`` 的版本；
    這裡提供 builder 函式 + 清楚 TODO，讓骨架在「尚未 save_model」時仍可 import。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- dynamic batching 建議參數（依壓測調整）---------------------------------
# max_batch_size：單一 batch 最多合併幾筆請求。表格輕量可大；影像較重宜小。
# max_latency_ms：等待湊滿 batch 的最長時間（毫秒），超過即送出。
TABULAR_BATCHING = {"max_batch_size": 128, "max_latency_ms": 20}
VISION_BATCHING = {"max_batch_size": 16, "max_latency_ms": 50}

# BentoML Model Store 內的模型標籤（由訓練端 bentoml.*.save_model 寫入）
TABULAR_MODEL_TAG = "smart_factory_tabular:latest"
VISION_MODEL_TAG = "smart_factory_vision:latest"


def build_tabular_runner() -> Any:
    """建立預測性維護（XGBoost）runner。

    Returns:
        已套用 dynamic batching 設定的 BentoML runner。

    Raises:
        RuntimeError: BentoML 未安裝，或 Model Store 內找不到對應模型標籤。
    """
    try:
        import bentoml
    except ImportError as exc:  # pragma: no cover - 環境缺套件
        raise RuntimeError("未安裝 bentoml，無法建立 runner") from exc

    try:
        model_ref = bentoml.xgboost.get(TABULAR_MODEL_TAG)
    except Exception as exc:  # noqa: BLE001
        # TODO: 訓練流程需呼叫 bentoml.xgboost.save_model("smart_factory_tabular", booster)
        raise RuntimeError(
            f"Model Store 內找不到 {TABULAR_MODEL_TAG}；請先在訓練端 save_model。"
        ) from exc

    runner = model_ref.to_runner(
        name="tabular_runner",
        max_batch_size=TABULAR_BATCHING["max_batch_size"],
        max_latency_ms=TABULAR_BATCHING["max_latency_ms"],
    )
    logger.info("建立 tabular_runner（batching=%s）", TABULAR_BATCHING)
    return runner


def build_vision_runner() -> Any:
    """建立瑕疵檢測（ONNX ResNet）runner。

    Returns:
        已套用 dynamic batching 設定的 BentoML runner。

    Raises:
        RuntimeError: BentoML 未安裝，或 Model Store 內找不到對應模型標籤。
    """
    try:
        import bentoml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未安裝 bentoml，無法建立 runner") from exc

    try:
        model_ref = bentoml.onnx.get(VISION_MODEL_TAG)
    except Exception as exc:  # noqa: BLE001
        # TODO: 視覺訓練端需呼叫 bentoml.onnx.save_model("smart_factory_vision", onnx_proto)
        raise RuntimeError(
            f"Model Store 內找不到 {VISION_MODEL_TAG}；請先在訓練端 save_model。"
        ) from exc

    runner = model_ref.to_runner(
        name="vision_runner",
        max_batch_size=VISION_BATCHING["max_batch_size"],
        max_latency_ms=VISION_BATCHING["max_latency_ms"],
    )
    logger.info("建立 vision_runner（batching=%s）", VISION_BATCHING)
    return runner
