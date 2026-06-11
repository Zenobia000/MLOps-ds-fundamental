"""pytest 全域 fixtures 與設定。

提供整個測試套件共用的小樣本資料與暫存設定，讓單元 / 整合 / 資料契約
測試都能在「不依賴外部大資料、不寫死路徑」的前提下快速跑完。

設計原則
--------
- 所有 fixture 皆回傳「不可變的全新物件」（DataFrame 用 ``.copy()``），
  避免測試之間互相污染。
- 實體鍵固定為 ``machine_id``、時間鍵固定為 ``event_timestamp``，
  與全專案契約一致。
- 不在這裡 import 任何 ``src.*`` 模組，避免「sibling 模組尚未建好」時
  整個收集階段就崩潰；各測試檔自行用 ``pytest.importorskip`` 守護。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# --- 全專案共同契約常數 -------------------------------------------------------
ENTITY_COL = "machine_id"
TIMESTAMP_COL = "event_timestamp"
SENSOR_COLS = ["temperature", "vibration", "current"]
LABEL_COL = "failure"

# datasets/ 共用玩具資料（相對 repo 結構推導，找不到時退回合成資料）
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOY_SENSORS = _REPO_ROOT / "mlops-course" / "datasets" / "toy_sensors.csv"


def _synthetic_sensors(n_machines: int = 3, n_steps: int = 40) -> pd.DataFrame:
    """以固定種子生成合成感測時序，作為玩具資料的離線後援。

    故障規則刻意與 toy_sensors 一致：溫度高 + 振動大 → 故障機率上升，
    確保特徵與標籤之間有可被測試斷言的真實關聯。
    """
    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for m in range(1, n_machines + 1):
        temp = rng.normal(62.0, 3.0, n_steps).cumsum() / np.arange(1, n_steps + 1)
        vib = rng.normal(3.0, 0.4, n_steps)
        cur = rng.normal(9.5, 0.5, n_steps)
        score = (temp - 62.0) / 3.0 + (vib - 3.0) / 0.4
        fail = (score > 1.0).astype(int)
        for i in range(n_steps):
            rows.append(
                {
                    ENTITY_COL: f"machine_{m:02d}",
                    TIMESTAMP_COL: base + pd.Timedelta(hours=i),
                    "temperature": round(float(temp[i]), 3),
                    "vibration": round(float(vib[i]), 3),
                    "current": round(float(cur[i]), 3),
                    LABEL_COL: int(fail[i]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sensors_df() -> pd.DataFrame:
    """回傳小樣本感測 DataFrame（優先真實玩具檔，否則合成後援）。"""
    if _TOY_SENSORS.exists():
        df = pd.read_csv(_TOY_SENSORS, parse_dates=[TIMESTAMP_COL])
        return df.copy()
    return _synthetic_sensors()


@pytest.fixture()
def mini_sensors_df(sensors_df: pd.DataFrame) -> pd.DataFrame:
    """單一機台、前 24 筆的極小切片，給快速單元測試用。"""
    one = sensors_df[sensors_df[ENTITY_COL] == sensors_df[ENTITY_COL].iloc[0]]
    return one.sort_values(TIMESTAMP_COL).head(24).reset_index(drop=True).copy()


@pytest.fixture()
def feature_columns() -> list[str]:
    """回傳結構化模型的原始特徵欄位清單。"""
    return list(SENSOR_COLS)


@pytest.fixture()
def tmp_config(tmp_path: Path) -> dict[str, object]:
    """寫出一份最小但符合契約的 config.yaml，回傳 dict 與其路徑。

    結構對齊 ``conf/config.yaml`` 契約：project / seed / paths / mlflow / active_model。
    讓需要 config 的測試不必依賴 repo 內真實設定檔。
    """
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (tmp_path / "models").mkdir()

    cfg: dict[str, object] = {
        "project": "smart-factory-mlops-test",
        "seed": 42,
        "paths": {
            "data_raw": str(data_dir / "raw"),
            "data_processed": str(data_dir / "processed"),
            "models": str(tmp_path / "models"),
        },
        "mlflow": {
            "tracking_uri": f"file://{tmp_path / 'mlruns'}",
            "experiment": "test-experiment",
        },
        "active_model": "xgboost",
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cfg["__path__"] = str(cfg_path)  # 方便測試取回檔案位置
    return cfg


@pytest.fixture()
def sensors_csv(tmp_path: Path, sensors_df: pd.DataFrame) -> str:
    """把小樣本資料落地成 csv，回傳路徑，給需要讀檔的整合測試用。"""
    path = tmp_path / "sensors.csv"
    sensors_df.to_csv(path, index=False)
    return str(path)
