"""BentoML 服務：把「預測性維護（表格）+ 瑕疵檢測（影像）」包成單一 API。

對外端點：
    POST /predict_maintenance  ─ 表格批次推論（JSON）
    POST /predict_defect       ─ 影像單張推論（multipart / image）
    GET  /healthz              ─ 健康檢查（runner 是否就緒）

設計重點：
    - **runner 共用**：runner 由 services/runners.py 定義，service 只負責 I/O、
      schema 驗證與後處理；推論的數學邏輯重用 src/serving/predict.py。
    - **優雅降級**：教學 / CI 環境若 Model Store 尚無模型，service 仍可 import 與
      啟動，相關端點回 503，而非整個服務崩潰。
    - **config-driven**：門檻、registry URI 來自 conf/config.yaml。

啟動方式（repo 根目錄）：
    bentoml serve services.service:svc --reload
"""

from __future__ import annotations

import logging

import numpy as np

from src.serving.predict import predict_defect, preprocess_image
from src.serving.schemas import (
    DefectResponse,
    HealthResponse,
    MaintenanceRequest,
    MaintenanceResponse,
)
from src.utils.config import load_config

try:
    import bentoml
    from bentoml.io import JSON, Image
except ImportError:  # pragma: no cover - 讓無 bentoml 環境仍可 import 本模組做單元測試
    bentoml = None  # type: ignore[assignment]
    JSON = Image = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --- 設定載入（失敗時用安全預設，不讓服務啟動崩潰）--------------------------
try:
    _CFG = load_config()
except Exception as exc:  # noqa: BLE001
    logger.warning("載入 config 失敗（%s），改用預設門檻", exc)
    _CFG = {}

_SERVING_CFG = _CFG.get("serving", {})
MAINT_THRESHOLD: float = float(_SERVING_CFG.get("tabular_threshold", 0.5))
DEFECT_THRESHOLD: float = float(_SERVING_CFG.get("vision_threshold", 0.5))


def _safe_build(builder) -> object | None:
    """嘗試建立 runner；缺模型時記錄並回 None（端點之後回 503）。"""
    try:
        return builder()
    except Exception as exc:  # noqa: BLE001
        logger.warning("runner 建立失敗，相關端點將回 503：%s", exc)
        return None


if bentoml is not None:
    from services.runners import build_tabular_runner, build_vision_runner

    _tabular_runner = _safe_build(build_tabular_runner)
    _vision_runner = _safe_build(build_vision_runner)
    _runners = [r for r in (_tabular_runner, _vision_runner) if r is not None]

    svc = bentoml.Service("smart_factory", runners=_runners)

    @svc.api(input=JSON(pydantic_model=MaintenanceRequest), output=JSON(pydantic_model=MaintenanceResponse))
    async def predict_maintenance_api(req: MaintenanceRequest) -> MaintenanceResponse:
        """預測性維護批次推論端點。"""
        if _tabular_runner is None:
            raise bentoml.exceptions.ServiceUnavailable("表格模型尚未就緒")
        features = np.asarray(
            [[r.temperature, r.vibration, r.current] for r in req.readings],
            dtype=np.float32,
        )
        proba = await _tabular_runner.async_run(features)
        # 重用後處理：以 runner 回傳的機率組裝結構化結果
        from src.serving.predict import _to_proba  # noqa: PLC0415 — 局部匯入避免循環
        from src.serving.schemas import MaintenancePrediction  # noqa: PLC0415

        scores = _to_proba(np.asarray(proba))
        predictions = [
            MaintenancePrediction(
                machine_id=r.machine_id,
                failure_probability=float(s),
                will_fail=bool(s >= MAINT_THRESHOLD),
            )
            for r, s in zip(req.readings, scores)
        ]
        return MaintenanceResponse(
            model_name="smart_factory_tabular",
            model_version="bento",
            threshold=MAINT_THRESHOLD,
            predictions=predictions,
        )

    @svc.api(input=Image(), output=JSON(pydantic_model=DefectResponse))
    async def predict_defect_api(img) -> DefectResponse:  # noqa: ANN001 — PIL.Image
        """瑕疵檢測單張影像推論端點。"""
        if _vision_runner is None:
            raise bentoml.exceptions.ServiceUnavailable("影像模型尚未就緒")
        rgb = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.uint8)
        batch = preprocess_image(rgb)
        logits = await _vision_runner.async_run(batch)
        prediction = predict_defect(np.asarray(logits), DEFECT_THRESHOLD)
        return DefectResponse(
            model_name="smart_factory_vision",
            model_version="bento",
            threshold=DEFECT_THRESHOLD,
            prediction=prediction,
        )

    @svc.api(input=JSON(), output=JSON(pydantic_model=HealthResponse))
    async def healthz(_: dict) -> HealthResponse:
        """健康檢查：回報兩個 runner 是否就緒。"""
        ok = _tabular_runner is not None and _vision_runner is not None
        return HealthResponse(
            status="ok" if ok else "degraded",
            tabular_model_loaded=_tabular_runner is not None,
            vision_model_loaded=_vision_runner is not None,
        )

else:  # pragma: no cover - 純文件用途，提醒安裝 bentoml
    svc = None
    logger.warning("未安裝 bentoml；services.service:svc 僅供 import 測試，無法 serve")
