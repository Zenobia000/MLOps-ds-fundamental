"""模型載入：從 MLflow Model Registry 取模型，並提供本地產物回退。

把「載入」與「推論」分離（單一職責）：predict.py 只關心數學，model_loader.py
只關心「如何拿到一個可推論的物件」。兩者都吃 ``conf/config.yaml`` 的 dict。

回退策略：
    registry 不可用（離線 / 教學環境）時，改讀本地產物，避免服務完全無法啟動；
    皆失敗才拋 RuntimeError，由呼叫端決定降級（回 503）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_tabular_model(cfg: dict[str, Any]) -> tuple[Any, str, str]:
    """從 MLflow Model Registry 載入表格（XGBoost）模型，registry 失敗則本地回退。

    Returns:
        ``(model, model_name, model_version)``；model 具備 ``predict_proba``
        或 ``predict`` 介面。

    Raises:
        RuntimeError: registry 與本地回退皆失敗時拋出。
    """
    serving_cfg = cfg.get("serving", {})
    model_name = serving_cfg.get("tabular_model_name", "smart_factory_tabular")
    stage = serving_cfg.get("tabular_model_stage", "Production")
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri")

    try:
        import mlflow  # 延遲匯入：未裝 mlflow 的環境仍可 import 本模組

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(uri)
        logger.info("已從 registry 載入表格模型：%s", uri)
        return model, model_name, stage
    except Exception as exc:  # noqa: BLE001 — 回退路徑需攔截所有載入錯誤
        logger.warning("registry 載入失敗（%s），改用本地產物回退", exc)

    local_path = Path(cfg.get("paths", {}).get("models", "models")) / "tabular" / "model.xgb"
    if not local_path.exists():
        raise RuntimeError(
            f"表格模型載入失敗：registry 不可用且本地產物不存在 {local_path}。"
            "請先執行 'python -m src.training.train' 產生模型。"
        )
    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(local_path))
        logger.info("已從本地載入表格模型：%s", local_path)
        return booster, model_name, "local"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"本地表格模型載入失敗：{exc}") from exc


def load_vision_session(cfg: dict[str, Any]) -> tuple[Any, str, str]:
    """載入視覺瑕疵檢測模型（ONNX Runtime session）。

    Returns:
        ``(session, model_name, model_version)``。

    Raises:
        RuntimeError: ONNX 產物不存在或 onnxruntime 未安裝時拋出。
    """
    serving_cfg = cfg.get("serving", {})
    model_name = serving_cfg.get("vision_model_name", "smart_factory_vision")
    onnx_path = Path(cfg.get("paths", {}).get("models", "models")) / "vision" / "model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(
            f"影像模型載入失敗：ONNX 產物不存在 {onnx_path}。"
            "請先執行 'python -m src.training.train model=resnet' 並匯出 ONNX。"
        )
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        logger.info("已載入影像模型（ONNX）：%s", onnx_path)
        return session, model_name, "onnx"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ONNX 影像模型載入失敗：{exc}") from exc
