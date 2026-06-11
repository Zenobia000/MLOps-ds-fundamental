"""資料層（data layer）。

負責「把外部資料安全地讀進記憶體並驗證契約」：

- :mod:`src.data.loaders`：載入感測器時序、瑕疵影像、產能需求三型態資料，
  缺真實資料時自動後援到課程共用玩具資料（toy_sensors）。
- :mod:`src.data.validation`：以 schema / 分布檢查把關，快速失敗並給清晰錯誤。

設計原則：純函式、回傳新物件（不可變）、不在此層做特徵工程
（特徵工程屬 :mod:`src.features`），維持高內聚低耦合。
"""

from src.data.loaders import load_demand, load_images, load_sensors
from src.data.validation import (
    DataValidationError,
    validate_image_manifest,
    validate_sensors,
)

__all__ = [
    "load_sensors",
    "load_images",
    "load_demand",
    "validate_sensors",
    "validate_image_manifest",
    "DataValidationError",
]
