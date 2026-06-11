"""服務化層（M4）：推論前後處理、輸入驗證 schema、registry 模型載入。

對外公開最常用的 schema 與推論函式，讓 services/ 與測試可簡潔匯入：

    from src.serving import MaintenanceRequest, predict_maintenance
"""

from src.serving.model_loader import load_tabular_model, load_vision_session
from src.serving.predict import (
    predict_defect,
    predict_maintenance,
    preprocess_image,
    readings_to_matrix,
)
from src.serving.schemas import (
    DefectPrediction,
    DefectResponse,
    HealthResponse,
    MaintenancePrediction,
    MaintenanceRequest,
    MaintenanceResponse,
    SensorReading,
)

__all__ = [
    "DefectPrediction",
    "DefectResponse",
    "HealthResponse",
    "MaintenancePrediction",
    "MaintenanceRequest",
    "MaintenanceResponse",
    "SensorReading",
    "load_tabular_model",
    "load_vision_session",
    "predict_defect",
    "predict_maintenance",
    "preprocess_image",
    "readings_to_matrix",
]
