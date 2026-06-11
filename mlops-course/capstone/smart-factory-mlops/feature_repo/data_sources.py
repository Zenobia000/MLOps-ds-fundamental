"""Feast 資料來源定義（data_sources）。

感測器時序特徵的離線來源為本地 parquet（FileSource）。該檔由
``src/features/build_features.py`` 造好特徵後寫出（見 data/README 與
notebooks/03）。``timestamp_field`` 對齊全專案契約 ``event_timestamp``，
Feast 以此欄做 point-in-time join。
"""

from pathlib import Path

from feast import FileSource
from feast.data_format import ParquetFormat

# 特徵檔位於 feature_repo/data/sensor_features.parquet（相對本檔）。
_FEATURES_PATH = Path(__file__).resolve().parent / "data" / "sensor_features.parquet"

# 感測器時序特徵來源。
sensor_stats_source = FileSource(
    name="sensor_stats_source",
    path=str(_FEATURES_PATH),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    description="每台機台每小時的感測器滾動統計特徵（離線 parquet）。",
)
