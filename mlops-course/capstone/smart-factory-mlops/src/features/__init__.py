"""特徵層（features layer）。

把已驗證的原始資料轉成「可訓練 / 可上線」的特徵：

- :mod:`src.features.build_features`：感測器時序特徵（滾動均值、振動 std…），
  以 ``groupby(machine_id)`` + ``shift`` 嚴格避免時間穿越（leakage）。
- :mod:`src.features.feast_io`：Feast 取數薄封裝
  （``get_historical_features`` / ``get_online_features``）。

特徵層只負責「造特徵」，不碰模型；造好的特徵集落 ``data/processed`` 後
由 Feast 與訓練層共用。
"""

from src.features.build_features import (
    FEATURE_VIEW_COLUMNS,
    build_sensor_features,
)
from src.features.feast_io import (
    get_historical_features,
    get_online_features,
)

__all__ = [
    "build_sensor_features",
    "FEATURE_VIEW_COLUMNS",
    "get_historical_features",
    "get_online_features",
]
