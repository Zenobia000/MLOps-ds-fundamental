"""
01_first_run.py —— MLflow 最小可用單元（技能階梯 階 1）

這個檔示範什麼：
    用「一個 run」把一次訓練的「超參數」與「結果指標」記錄到 MLflow。
    只用三個動詞：start_run / log_param / log_metric。
    跑完後 MLflow 會在本資料夾產生 ./mlruns/，那就是你的實驗紀錄。

怎麼跑：
    python 01_first_run.py

看結果（在本資料夾開 UI，瀏覽器進 http://127.0.0.1:5000）：
    mlflow ui
"""

import os
from pathlib import Path

# 新版 MLflow 對「本地檔案紀錄（./mlruns）」預設會擋；教學要的就是這種零設定的本地紀錄，
# 因此開一個環境變數明確允許它（等於告訴 MLflow：我知道，就是要用檔案後端）。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 固定隨機種子，確保每次切分與訓練都可重現
SEED = 42

# 玩具資料只放在共用的 datasets/，模組內不複製資料檔
# 本檔在 .../sandbox/mlflow/，往上四層回到 mlops-course/ 再進 datasets/
DATA_PATH = Path(__file__).resolve().parents[4] / "datasets" / "iris.csv"


def main() -> None:
    # 讀玩具資料：iris 四個數值特徵 + target 標籤
    # 欄位 = sepal_length / sepal_width / petal_length / petal_width / target / target_name
    # target 是數值標籤（0/1/2）；target_name 只是給人看的花種名，建模時兩個都從特徵拿掉
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target", "target_name"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 這次要試的超參數（之後你會改它、再跑一次、在 UI 比較）
    C = 1.0
    max_iter = 200

    # start_run 是一切的開始：這個 with 區塊內記錄的東西都屬於同一個 run
    with mlflow.start_run(run_name="first-run"):
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=SEED)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        # 動詞 1：log_param —— 記錄「輸入」（這次用了什麼超參）
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        # 動詞 2：log_metric —— 記錄「輸出」（這次跑出什麼成績）
        mlflow.log_metric("accuracy", acc)

        print(f"已記錄一個 run：C={C}, max_iter={max_iter}, accuracy={acc:.4f}")
        print("執行 `mlflow ui` 後到 http://127.0.0.1:5000 看這個 run")


if __name__ == "__main__":
    main()
