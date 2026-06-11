"""
02_params_metrics.py —— 多組 params/metrics + 指定追蹤位置與實驗（技能階梯 階 1）

這個檔示範什麼：
    在 01 的基礎上多學三件事：
      1. set_tracking_uri：明確指定「紀錄要存去哪」（這裡用本地資料夾 ./mlruns）。
      2. set_experiment：把相關的 run 收進同一個「實驗」資料夾，方便分組比較。
      3. 一個 run 可以一次記「多個」params 與 metrics（log_params / log_metrics 複數版）。
    我們連跑三組不同超參，每組各一個 run，全部進同一個實驗。

怎麼跑：
    python 02_params_metrics.py
    mlflow ui          # 在本資料夾開 UI，到 experiment「iris-logreg-sweep」比較三個 run
"""

import os
from pathlib import Path

# 新版 MLflow 對「本地檔案紀錄（./mlruns）」預設會擋；教學要的就是這種零設定的本地紀錄，
# 因此開一個環境變數明確允許它（等於告訴 MLflow：我知道，就是要用檔案後端）。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[4] / "datasets" / "iris.csv"

# 本檔所在資料夾，紀錄就存在這裡的 ./mlruns（用 as_uri() 轉成 file:// 合法網址）
TRACKING_URI = (Path(__file__).resolve().parent / "mlruns").as_uri()


def main() -> None:
    # iris 欄位：四個數值特徵 + target（數值標籤）+ target_name（花種名，僅供人讀）
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target", "target_name"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 動詞：set_tracking_uri —— 告訴 MLflow 紀錄存哪裡
    mlflow.set_tracking_uri(TRACKING_URI)
    print("tracking uri =", mlflow.get_tracking_uri())

    # 動詞：set_experiment —— 之後所有 run 都自動進這個實驗
    mlflow.set_experiment("iris-logreg-sweep")

    # 三組不同超參，逐一各開一個 run（手動掃參數；下一階 Optuna 會把這件事自動化）
    param_grid = [
        {"C": 0.1, "max_iter": 200},
        {"C": 1.0, "max_iter": 200},
        {"C": 10.0, "max_iter": 200},
    ]

    for params in param_grid:
        with mlflow.start_run(run_name=f"C={params['C']}"):
            model = LogisticRegression(random_state=SEED, **params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1_macro": f1_score(y_test, preds, average="macro"),
            }

            # 複數版：一次記多個 params / 多個 metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            print(f"  run C={params['C']:>4} -> "
                  f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

    print("三個 run 已寫入實驗 iris-logreg-sweep，用 `mlflow ui` 並排比較")


if __name__ == "__main__":
    main()
