"""推論 API 的請求 / 回應結構（Pydantic schema）。

本模組定義「系統邊界」的資料契約：所有外部輸入在進入推論邏輯前，
都必須先通過這些 schema 的驗證（fail-fast、清晰錯誤訊息）。

契約對齊：
    - 實體（entity）欄位 = ``machine_id``
    - 時間戳欄位         = ``event_timestamp``
    - 預測維護特徵       = temperature / vibration / current（與 toy_sensors.csv 一致）
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 預測性維護（結構化 / 時序）─ 表格模型
# ---------------------------------------------------------------------------


class SensorReading(BaseModel):
    """單一設備在某時間點的感測器讀數（表格模型的一筆輸入）。"""

    machine_id: str = Field(..., description="設備實體 ID", examples=["machine_01"])
    event_timestamp: datetime | None = Field(
        default=None, description="事件時間戳（ISO8601），可選，缺省時不影響純表格推論"
    )
    temperature: float = Field(..., description="溫度感測值", ge=-50.0, le=500.0)
    vibration: float = Field(..., description="振動感測值", ge=0.0)
    current: float = Field(..., description="電流感測值", ge=0.0)


class MaintenanceRequest(BaseModel):
    """預測性維護批次請求：一次可送多筆讀數。"""

    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class MaintenancePrediction(BaseModel):
    """單筆預測性維護結果。"""

    machine_id: str
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="故障機率")
    will_fail: bool = Field(..., description="是否判定為將故障（依門檻）")


class MaintenanceResponse(BaseModel):
    """預測性維護批次回應。"""

    model_name: str = Field(..., description="實際命中的 registry 模型名稱")
    model_version: str = Field(..., description="模型版本 / stage")
    threshold: float = Field(..., description="判定門檻")
    predictions: list[MaintenancePrediction]


# ---------------------------------------------------------------------------
# 視覺瑕疵檢測（影像）─ 影像模型
# ---------------------------------------------------------------------------


class DefectPrediction(BaseModel):
    """單張影像瑕疵檢測結果。"""

    label: Literal["good", "defect"] = Field(..., description="預測類別")
    defect_probability: float = Field(..., ge=0.0, le=1.0)


class DefectResponse(BaseModel):
    """影像瑕疵檢測回應（單張影像）。"""

    model_name: str
    model_version: str
    threshold: float
    prediction: DefectPrediction


# ---------------------------------------------------------------------------
# 健康檢查
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """服務健康狀態。"""

    status: Literal["ok", "degraded"] = "ok"
    tabular_model_loaded: bool = False
    vision_model_loaded: bool = False
