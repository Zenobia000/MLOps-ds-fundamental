"""產生本課程共用的玩具資料集（Layer 1 沙盒專用）。

這個檔示範什麼：
    用「最小、可重現」的玩具資料，餵給後面每一個工具沙盒。
    一律設好 seed，讓每次重跑都得到同一份資料，避免「資料會變」干擾學習。

產生哪些檔（都寫到本資料夾 datasets/ 下）：
    1. iris.csv        —— sklearn 內建鳶尾花，分類沙盒（m1 baseline / MLflow / Optuna）
    2. toy_sensors.csv —— 自製極簡時序，時序 / 漂移沙盒（Feast / Evidently）
    diabetes.csv 已沿用既有檔，不在此重產。

怎麼跑：
    cd datasets
    python make_datasets.py

設計原則（玩具資料先行）：
    新工具的初次接觸，注意力要留給「工具怎麼用」，不要被複雜資料分心。
    所以這裡的資料刻意小、欄位刻意少、關係刻意直白。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# ── 全域固定 seed：確保可重現（同 seed → 同資料） ──────────────────
SEED = 42

# 本檔所在資料夾，輸出一律寫到這裡，跟「從哪裡呼叫」無關
DATASETS_DIR = Path(__file__).resolve().parent


def make_iris() -> Path:
    """把 sklearn 內建 iris 存成帶欄位名與 target 的 CSV（分類沙盒）。

    iris 是最經典的玩具分類資料：4 個數值特徵、3 個花種類別、共 150 列。
    乾淨、平衡、無缺值，最適合「先把工具跑起來」。
    """
    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()  # 不可變：操作副本，不動原物件

    # 欄位名換成簡潔好打的版本（去掉空白與單位括號）
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )

    # 額外補一欄「花種名稱」，方便人眼閱讀；target 仍保留數值標籤供建模
    target_names = {i: name for i, name in enumerate(bunch.target_names)}
    df["target_name"] = df["target"].map(target_names)

    out = DATASETS_DIR / "iris.csv"
    df.to_csv(out, index=False)
    return out


def make_toy_sensors(n_machines: int = 5, n_steps: int = 150) -> Path:
    """自製極簡時序感測資料：機台 × 時間，含故障標籤（時序 / 漂移沙盒）。

    欄位：
        machine_id       —— 機台代號（entity，給 Feast 當實體鍵）
        event_timestamp  —— 事件時間（每台每小時一筆，給 Feast 當時間鍵）
        temperature      —— 溫度（°C）
        vibration        —— 振動（mm/s）
        current          —— 電流（A）
        failure          —— 是否故障（0/1 標籤）

    列數 = n_machines * n_steps（預設 5 * 150 = 750 列，落在 500–1000）。

    故障規則（刻意簡單、好解釋）：
        溫度高 + 振動大時，故障機率上升。學生能一眼看出「特徵 → 標籤」的關係。

    可注入的漂移選項（給 m6 Evidently 沙盒用）：
        若要製造「資料漂移」情境，可在呼叫端對「後半段時間」的特徵動手腳，例如：
            mask = df["event_timestamp"] >= df["event_timestamp"].quantile(0.5)
            df.loc[mask, "temperature"] += 8.0      # 溫度整體偏移（covariate drift）
            df.loc[mask, "vibration"] *= 1.5        # 振動放大（scale drift）
        這樣參考期 vs 當前期的分布就不同，Evidently 能偵測到漂移。
        本函式預設「不注入漂移」，保持一份乾淨基準；漂移由沙盒範例自行注入。
    """
    rng = np.random.default_rng(SEED)  # numpy 固定 seed，取代裸 random

    # 每台機台正常運轉的「基準值」略有差異，讓資料更真實一點
    base_temp = rng.uniform(55, 65, size=n_machines)      # 各機台基準溫度
    base_vib = rng.uniform(2.0, 3.0, size=n_machines)     # 各機台基準振動
    base_cur = rng.uniform(9.0, 11.0, size=n_machines)    # 各機台基準電流

    # 時間軸：從固定起點起算，每台每小時一筆（時間鍵）
    start = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(start=start, periods=n_steps, freq="h")

    rows = []
    for m in range(n_machines):
        machine_id = f"machine_{m + 1:02d}"  # machine_01 ... machine_05
        for t in range(n_steps):
            # 加上常態雜訊，模擬感測讀數的隨機波動
            temperature = base_temp[m] + rng.normal(0, 2.0)
            vibration = base_vib[m] + rng.normal(0, 0.3)
            current = base_cur[m] + rng.normal(0, 0.5)

            # 故障機率：溫度與振動的「絕對值」越高越危險，且兩者同時偏高有加乘
            # 效應（交互項）。係數刻意調得讓 temperature / vibration 能「直接」預測
            # failure（模型 roc_auc≈0.9），確保這是一份「學得起來」的教學資料；
            # offset 控制故障維持在約 1/4 的少數類。
            t_excess = temperature - 60.0
            v_excess = vibration - 2.5
            risk = 0.8 * t_excess + 3.0 * v_excess + 0.6 * t_excess * v_excess - 3.4
            prob = 1.0 / (1.0 + np.exp(-risk))  # sigmoid 壓到 0~1
            failure = int(rng.random() < prob)

            rows.append(
                {
                    "machine_id": machine_id,
                    "event_timestamp": timestamps[t],
                    "temperature": round(float(temperature), 3),
                    "vibration": round(float(vibration), 3),
                    "current": round(float(current), 3),
                    "failure": failure,
                }
            )

    df = pd.DataFrame(rows)

    out = DATASETS_DIR / "toy_sensors.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    iris_path = make_iris()
    iris_df = pd.read_csv(iris_path)
    print(f"[OK] {iris_path.name}: {len(iris_df)} 列, 欄位 = {list(iris_df.columns)}")

    sensors_path = make_toy_sensors()
    sensors_df = pd.read_csv(sensors_path)
    print(
        f"[OK] {sensors_path.name}: {len(sensors_df)} 列, "
        f"故障數 = {int(sensors_df['failure'].sum())}, "
        f"欄位 = {list(sensors_df.columns)}"
    )

    diabetes_path = DATASETS_DIR / "diabetes.csv"
    status = "已存在" if diabetes_path.exists() else "缺檔（請放回 datasets/diabetes.csv）"
    print(f"[--] {diabetes_path.name}: {status}（沿用既有檔，不重產）")


if __name__ == "__main__":
    main()
