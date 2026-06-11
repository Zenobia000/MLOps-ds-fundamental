"""監控子套件（M6）。

匯集兩大能力：
- :mod:`drift`：以 Evidently 進行資料漂移偵測（ref vs current）。
- :mod:`metrics`：以 Prometheus 匯出延遲 / QPS / 預測分布等線上指標。

設計原則：與模型型態無關（tabular / timeseries / vision 共用同一套監控介面），
所有閾值與欄位皆 config-driven，不在程式中硬編碼。
"""

from __future__ import annotations

__all__ = [
    "run_data_drift",
    "DriftResult",
    "PredictionMetrics",
    "build_registry",
]

from src.monitoring.drift import DriftResult, run_data_drift  # noqa: E402
from src.monitoring.metrics import PredictionMetrics, build_registry  # noqa: E402
