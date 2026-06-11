"""產生 Feast 離線特徵 parquet（offline materialize）。

從感測器原始資料造好時序特徵後，寫成 parquet 供 feature_repo 的
FileSource 讀取。供教學 / CI 重現使用：

    python -m src.features.materialize_offline

產出 ``feature_repo/data/sensor_features.parquet``，接著即可在
feature_repo 下執行 ``feast apply`` / ``feast materialize ...``。

> 刻意放在 ``src/features`` 而非 feature_repo：因為 ``feast apply`` 會掃描
> feature_repo 內所有 ``.py`` 當成特徵定義，若放含 ``src`` import 的腳本會壞掉。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.data.loaders import load_sensors
from src.data.validation import validate_sensors
from src.features.build_features import FEATURE_VIEW_COLUMNS, build_sensor_features

logger = logging.getLogger(__name__)

# 輸出落點：feature_repo/data/sensor_features.parquet（與 data_sources.py 對齊）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT = _REPO_ROOT / "feature_repo" / "data" / "sensor_features.parquet"


def materialize_offline(out_path: Path = _OUT) -> Path:
    """造特徵並寫出 parquet；回傳輸出路徑。"""
    df = load_sensors()
    validate_sensors(df)
    feats = build_sensor_features(df)

    # 只保留 Feast 需要的欄位：entity + timestamp + 特徵（+ 標籤供 spine 用）。
    keep = ["machine_id", "event_timestamp", *FEATURE_VIEW_COLUMNS]
    if "failure" in feats.columns:
        keep.append("failure")
    out_df = feats[keep].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("寫出 %d 列特徵到 %s", len(out_df), out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    materialize_offline()
