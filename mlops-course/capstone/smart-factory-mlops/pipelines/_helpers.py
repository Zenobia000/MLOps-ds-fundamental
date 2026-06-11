"""管線共用小工具：軟相依 import、config 載入、小樣本後援資料。

把「sibling 模組可能尚未建好」的彈性集中在一處，避免三條 pipeline 重複
try/except，維持高內聚低耦合。
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("pipelines")

ENTITY_COL = "machine_id"
TIMESTAMP_COL = "event_timestamp"

# datasets/ 玩具資料（從本檔往上推導 repo 結構）
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOY_SENSORS = _REPO_ROOT / "mlops-course" / "datasets" / "toy_sensors.csv"


def soft_import(module: str, attr: str) -> Callable[..., Any] | None:
    """嘗試取得 ``module.attr``；任何失敗都回 None 並記錄，不拋例外。"""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - 軟相依設計
        logger.info("軟相依模組 %s 不可用（%s），改用後援。", module, exc)
        return None
    fn = getattr(mod, attr, None)
    if fn is None:
        logger.info("模組 %s 未提供 %s，改用後援。", module, attr)
    return fn


def load_config(conf_dir: str | None = None) -> dict:
    """載入 config：優先用 ``src.utils.config.load_config``，否則回最小預設。

    與全專案契約一致（project/seed/paths/mlflow/active_model）。
    ``conf_dir`` 為設定**目錄**（非單一檔案），與 src.utils.config 契約一致。
    """
    loader = soft_import("src.utils.config", "load_config")
    if loader is not None:
        try:
            return dict(loader(conf_dir) if conf_dir else loader())
        except Exception as exc:  # noqa: BLE001
            logger.warning("src.utils.config.load_config 失敗（%s），用預設。", exc)

    return {
        "project": "smart-factory-mlops",
        "seed": 42,
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "models": "models",
        },
        "mlflow": {
            "tracking_uri": "file:./mlruns",
            "experiment": "smart-factory",
        },
        "active_model": "xgboost",
    }


def load_sensors(config: dict) -> pd.DataFrame:
    """載入感測資料：優先 src.data.loaders，否則讀玩具檔，再否則合成。"""
    loader = soft_import("src.data.loaders", "load_sensors")
    if loader is not None:
        try:
            return loader(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("src.data.loaders.load_sensors 失敗（%s），用後援。", exc)

    if _TOY_SENSORS.exists():
        return pd.read_csv(_TOY_SENSORS, parse_dates=[TIMESTAMP_COL])

    import numpy as np

    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            ENTITY_COL: ["machine_01"] * n,
            TIMESTAMP_COL: pd.date_range("2024-01-01", periods=n, freq="h"),
            "temperature": rng.normal(62, 3, n),
            "vibration": rng.normal(3, 0.4, n),
            "current": rng.normal(9.5, 0.5, n),
        }
    )
    score = (df["temperature"] - 62) / 3 + (df["vibration"] - 3) / 0.4
    df["failure"] = (score > 1.0).astype(int)
    return df
