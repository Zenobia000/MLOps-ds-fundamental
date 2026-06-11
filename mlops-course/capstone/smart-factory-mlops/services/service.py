"""BentoML 服務（BentoML 1.2+ 新式 ``@bentoml.service`` API）。

把「預測性維護（表格 XGBoost）+ 瑕疵檢測（影像 ONNX ResNet）」包成單一服務：

對外端點：
    POST /predict_maintenance  ─ 表格批次推論（JSON: MaintenanceRequest）
    POST /predict_defect       ─ 影像單張推論（multipart image）
    POST /healthz              ─ 健康檢查

設計重點：
    - 模型於 ``__init__`` 載入（透過 :mod:`src.serving.model_loader`，registry →
      本地產物回退）；推論數學重用 :mod:`src.serving.predict`（無框架依賴、易測）。
    - **優雅降級**：模型載入失敗時對應端點回 503，而非整個服務崩潰。
    - **config-driven**：門檻來自 ``conf/config.yaml`` 的 ``serving`` 區段。

啟動方式（repo 根目錄）：
    bentoml serve services.service:SmartFactoryService --reload
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.serving.model_loader import load_tabular_model, load_vision_session
from src.serving.predict import (
    IMAGE_SIZE,
    predict_defect,
    predict_maintenance,
    preprocess_image,
)
from src.serving.schemas import (
    DefectResponse,
    HealthResponse,
    MaintenanceRequest,
    MaintenanceResponse,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _safe(builder: Callable[[], Any]) -> Any | None:
    """嘗試載入模型；失敗回 None（對應端點之後回 503）。"""
    try:
        return builder()
    except Exception as exc:  # noqa: BLE001 — 降級需攔截所有載入錯誤
        logger.warning("模型載入失敗，相關端點將回 503：%s", exc)
        return None


try:
    import bentoml
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover — 無 bentoml 環境仍可 import 本模組做單元測試
    bentoml = None  # type: ignore[assignment]


if bentoml is not None:

    @bentoml.service(name="smart_factory")
    class SmartFactoryService:
        """智慧工廠統一推論服務（表格 + 影像雙模型）。"""

        def __init__(self) -> None:
            try:
                self.cfg = load_config()
            except Exception as exc:  # noqa: BLE001
                logger.warning("載入 config 失敗（%s），改用預設門檻", exc)
                self.cfg = {}

            serving_cfg = self.cfg.get("serving", {})
            self.maint_threshold = float(serving_cfg.get("tabular_threshold", 0.5))
            self.defect_threshold = float(serving_cfg.get("vision_threshold", 0.5))

            # registry → 本地回退；任一失敗則該端點降級回 503。
            self.tabular = _safe(lambda: load_tabular_model(self.cfg))
            self.vision = _safe(lambda: load_vision_session(self.cfg))

        @bentoml.api
        def predict_maintenance(self, request: MaintenanceRequest) -> MaintenanceResponse:
            """預測性維護批次推論（一次多筆感測讀數）。"""
            if self.tabular is None:
                raise bentoml.exceptions.ServiceUnavailable("表格模型尚未就緒")
            model, name, version = self.tabular
            predictions = predict_maintenance(model, request.readings, self.maint_threshold)
            return MaintenanceResponse(
                model_name=name,
                model_version=version,
                threshold=self.maint_threshold,
                predictions=predictions,
            )

        @bentoml.api
        def predict_defect(self, image: PILImage.Image) -> DefectResponse:
            """瑕疵檢測單張影像推論。"""
            if self.vision is None:
                raise bentoml.exceptions.ServiceUnavailable("影像模型尚未就緒")
            session, name, version = self.vision
            arr = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
            batch = preprocess_image(arr)
            input_name = session.get_inputs()[0].name
            logits = session.run(None, {input_name: batch})[0]
            prediction = predict_defect(np.asarray(logits), self.defect_threshold)
            return DefectResponse(
                model_name=name,
                model_version=version,
                threshold=self.defect_threshold,
                prediction=prediction,
            )

        @bentoml.api
        def healthz(self) -> HealthResponse:
            """健康檢查：回報兩個模型是否就緒。"""
            return HealthResponse(
                status="ok" if (self.tabular and self.vision) else "degraded",
                tabular_model_loaded=self.tabular is not None,
                vision_model_loaded=self.vision is not None,
            )

else:  # pragma: no cover — 純文件用途，提醒安裝 bentoml
    SmartFactoryService = None
    logger.warning("未安裝 bentoml；services.service 僅供 import，無法 serve")
