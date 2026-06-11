"""Feast 特徵視圖與服務（features）。

定義感測器時序 FeatureView（滾動均值 / 振動 std…）與對外的
FeatureService。特徵欄位與 :data:`src.features.build_features.FEATURE_VIEW_COLUMNS`
保持單一定義一致，避免線上 / 離線特徵漂移（training-serving skew）。
"""

from datetime import timedelta

from feast import FeatureService, FeatureView, Field
from feast.types import Float32

from data_sources import sensor_stats_source
from entities import machine

# 感測器時序特徵視圖：以 machine 為實體，TTL 7 天（線上特徵新鮮度上限）。
sensor_stats_fv = FeatureView(
    name="sensor_stats",
    entities=[machine],
    ttl=timedelta(days=7),
    schema=[
        Field(name="temp_roll_mean", dtype=Float32),
        Field(name="vib_roll_std", dtype=Float32),
        Field(name="current_roll_mean", dtype=Float32),
        Field(name="temp_delta", dtype=Float32),
    ],
    online=True,
    source=sensor_stats_source,
    description="每台機台的感測器滾動統計特徵（避免時間穿越）。",
)

# 對外特徵服務：訓練與服務皆引用此名稱（見 src/features/feast_io.py）。
predictive_maintenance_v1 = FeatureService(
    name="predictive_maintenance_v1",
    features=[sensor_stats_fv],
    description="預測性維護模型（XGBoost）所需的感測器時序特徵集。",
)
