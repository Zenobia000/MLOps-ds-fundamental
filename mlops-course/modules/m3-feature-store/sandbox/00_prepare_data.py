"""
00_prepare_data.py — 把 diabetes.csv 轉成 Feast 能吃的 parquet。

這個檔示範什麼:
    Feast 的離線資料來源(FileSource)需要兩樣東西才能做 point-in-time join:
      1. 一個 entity 欄位(誰?) → 這裡是 patient_id
      2. 一個事件時間欄位(何時發生?) → 這裡是 event_timestamp
    原始 diabetes.csv 兩者都沒有,所以我們在這裡「補上」,
    並拆成 predictors(特徵)與 target(標籤)兩張表,模擬真實世界
    「特徵」與「標籤」常常來自不同表、不同時間的情境。

怎麼跑:
    cd 到本資料夾(sandbox/)後執行:
        python 00_prepare_data.py
    產出:
        feature_repo/data/predictors_df.parquet
        feature_repo/data/target_df.parquet
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 設定 seed:本檔會用亂數產生時間戳,固定 seed 才能每次跑出一樣結果
SEED = 42
np.random.seed(SEED)

# --- 路徑(全部相對本檔位置,確保在任何地方執行都正確)---
HERE = Path(__file__).resolve().parent                      # .../m3-feature-store/sandbox
DATASETS_DIR = HERE.parents[2] / "datasets"                 # .../mlops-course/datasets
CSV_PATH = DATASETS_DIR / "diabetes.csv"
OUT_DIR = HERE / "feature_repo" / "data"                    # FileSource 會用相對 feature_repo 的 data/

# diabetes.csv 的 8 個特徵欄位(predictors)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] 讀取 {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 1) 補上 entity:給每位病患一個穩定的整數 ID(0..N-1)
    df = df.reset_index(drop=True)
    df["patient_id"] = df.index.astype("int64")

    # 2) 補上事件時間戳:
    #    真實情境裡每筆紀錄的時間都不同。這裡用「過去 30 天內隨機灑點」模擬。
    #    重點:同一個 patient_id 的「特徵」與「標籤」要落在合理的先後關係,
    #    Feast 才能依時間正確 join(下一支 demo 會詳細示範)。
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    n = len(df)
    # 特徵在病患就診當下量測 → 隨機分布在過去 30 天
    predictor_offsets = np.random.randint(0, 30 * 24, size=n)  # 單位:小時
    predictor_ts = [base_time + timedelta(hours=int(h)) for h in predictor_offsets]

    # 標籤(是否罹病)是「事後」才確診的 → 比特徵時間晚 1 小時
    # 這樣 point-in-time join 時,訓練時刻拿得到特徵,也對得到正確標籤。
    target_ts = [t + timedelta(hours=1) for t in predictor_ts]

    # 3) 拆成兩張表
    # event_timestamp 一律寫成「帶 UTC 時區」。Feast 內部以 UTC 處理時間,
    # 來源與查詢兩邊都用 UTC,才不會在 point-in-time 比較時踩到 tz-naive/tz-aware 衝突。
    predictors_df = df[["patient_id"] + PREDICTOR_COLS].copy()
    predictors_df["event_timestamp"] = pd.to_datetime(predictor_ts, utc=True)

    target_df = df[["patient_id", TARGET_COL]].copy()
    target_df["event_timestamp"] = pd.to_datetime(target_ts, utc=True)

    # 4) 寫出 parquet(Feast FileSource 預設吃 parquet)
    pred_path = OUT_DIR / "predictors_df.parquet"
    tgt_path = OUT_DIR / "target_df.parquet"
    predictors_df.to_parquet(pred_path, index=False)
    target_df.to_parquet(tgt_path, index=False)

    print(f"[write] {pred_path}  ({len(predictors_df)} 列)")
    print(f"[write] {tgt_path}  ({len(target_df)} 列)")
    print("\n[預覽] predictors_df 前 3 列:")
    print(predictors_df.head(3).to_string(index=False))
    print("\n[預覽] target_df 前 3 列:")
    print(target_df.head(3).to_string(index=False))
    print("\n完成。下一步:cd feature_repo && feast apply,再跑 ../01_point_in_time_demo.py")


if __name__ == "__main__":
    main()
