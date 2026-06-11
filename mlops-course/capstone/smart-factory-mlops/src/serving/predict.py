"""推論邏輯：前處理 → 模型推論 → 後處理（純函式、無框架依賴）。

設計重點：
    - **config-driven**：門檻等參數來自 ``conf/config.yaml``（由呼叫端傳入）。
    - **模型載入分離**：「如何拿到模型」在 :mod:`src.serving.model_loader`；
      本模組只關心「拿到模型後怎麼算」（單一職責、易測試）。
    - **無狀態前後處理**：所有轉換都建立新物件（不可變），方便並行與測試。

此模組刻意「不依賴 BentoML」，純函式 / 純類別，讓 FastAPI、BentoML、批次腳本
都能重用同一套推論邏輯（高內聚、低耦合）。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# 為向後相容，從 model_loader re-export 載入函式（舊呼叫端仍可用）
from src.serving.model_loader import load_tabular_model, load_vision_session
from src.serving.schemas import (
    DefectPrediction,
    MaintenancePrediction,
    SensorReading,
)

__all__ = [
    "TABULAR_FEATURES",
    "load_tabular_model",
    "load_vision_session",
    "predict_defect",
    "predict_maintenance",
    "preprocess_image",
    "readings_to_matrix",
]

# 與 toy_sensors.csv 一致的特徵順序（務必與訓練端對齊）
TABULAR_FEATURES: tuple[str, ...] = ("temperature", "vibration", "current")
# 影像模型輸入尺寸（ResNet 微調預設）
IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# 前處理
# ---------------------------------------------------------------------------


def readings_to_matrix(readings: Sequence[SensorReading]) -> np.ndarray:
    """把感測讀數列表轉成 ``(n_samples, n_features)`` 的 float32 矩陣。

    特徵順序固定為 :data:`TABULAR_FEATURES`，確保與訓練端欄位順序一致。
    """
    rows = [[getattr(r, name) for name in TABULAR_FEATURES] for r in readings]
    return np.asarray(rows, dtype=np.float32)


def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """影像前處理：假設已 resize，這裡做 0-1 縮放 + ImageNet 標準化 + CHW。

    Args:
        image_array: 形狀 ``(H, W, 3)``、值域 0-255 的 RGB 影像。

    Returns:
        形狀 ``(1, 3, H, W)`` 的 float32 batch，供 ONNX / PyTorch runner 使用。
    """
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"預期 (H, W, 3) RGB 影像，實際為 {image_array.shape}")
    arr = image_array.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr[np.newaxis, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# 推論 + 後處理
# ---------------------------------------------------------------------------


def _to_proba(raw: np.ndarray) -> np.ndarray:
    """把模型原始輸出統一成「正類機率」一維陣列。"""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:  # predict_proba 兩欄
        return arr[:, 1]
    return arr.reshape(-1)


def _model_proba(model: Any, features: np.ndarray) -> np.ndarray:
    """依模型介面（sklearn / xgboost Booster / mlflow pyfunc）取得正類機率。"""
    if hasattr(model, "predict_proba"):
        return _to_proba(model.predict_proba(features))
    try:
        import xgboost as xgb

        if isinstance(model, xgb.Booster):
            return _to_proba(model.predict(xgb.DMatrix(features)))
    except ImportError:
        pass
    return _to_proba(model.predict(features))


def predict_maintenance(
    model: Any,
    readings: Sequence[SensorReading],
    threshold: float,
) -> list[MaintenancePrediction]:
    """執行預測性維護推論並組裝結構化結果。"""
    features = readings_to_matrix(readings)
    proba = _model_proba(model, features)
    return [
        MaintenancePrediction(
            machine_id=r.machine_id,
            failure_probability=float(p),
            will_fail=bool(p >= threshold),
        )
        for r, p in zip(readings, proba)
    ]


def predict_defect(raw_logits: np.ndarray, threshold: float) -> DefectPrediction:
    """把影像模型 logits 後處理成瑕疵判定。

    Args:
        raw_logits: 模型輸出，形狀 ``(1, 2)``（good / defect）或 ``(1, 1)``。
    """
    logits = np.asarray(raw_logits, dtype=np.float32).reshape(-1)
    if logits.size == 2:  # softmax 取 defect 機率
        exp = np.exp(logits - logits.max())
        prob_defect = float((exp / exp.sum())[1])
    else:  # 單一 sigmoid 輸出
        prob_defect = float(1.0 / (1.0 + np.exp(-logits[0])))
    label = "defect" if prob_defect >= threshold else "good"
    return DefectPrediction(label=label, defect_probability=prob_defect)
