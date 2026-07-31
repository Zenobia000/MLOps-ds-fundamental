"""
workspace/prepare_features.py — M3 填空：把玩具資料補成 Feast 能吃的 parquet

對照：modules/m3-feature-store/sandbox/00_prepare_data.py
怎麼跑：python workspace/prepare_features.py
產出：workspace/feature_repo/data/*.parquet
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
COURSE_ROOT = HERE.parent
CSV_PATH = COURSE_ROOT / "datasets" / "diabetes.csv"
OUT_DIR = HERE / "feature_repo" / "data"

PREDICTOR_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COL = "Outcome"


def main() -> None:
    # TODO(M3-1): 讀 diabetes.csv；若不存在 raise FileNotFoundError
    raise NotImplementedError("TODO(M3-1): 讀 CSV")

    # TODO(M3-2): 加 patient_id（0..n-1）與 event_timestamp（對照 sandbox 補時間戳手法）
    raise NotImplementedError("TODO(M3-2): patient_id + event_timestamp")

    # TODO(M3-3): 拆成 predictors / target 兩張表，寫入 OUT_DIR 的 parquet
    #   predictors_df.parquet / target_df.parquet
    raise NotImplementedError("TODO(M3-3): to_parquet")

    print(f"寫入完成：{OUT_DIR}")


if __name__ == "__main__":
    main()
