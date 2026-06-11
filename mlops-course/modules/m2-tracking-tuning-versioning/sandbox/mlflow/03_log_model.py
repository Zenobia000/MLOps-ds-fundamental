"""
03_log_model.py —— 記錄「模型本身」+ signature，再讀回來預測（技能階梯 階 1）

這個檔示範什麼：
    前兩個檔只記了 params/metrics（數字）。這裡記「模型物件」本身，讓它能被部署、被重用。
    新動詞：
      1. infer_signature：從 (輸入, 輸出) 自動推導「模型的輸入輸出長相」（schema）。
         signature 讓之後載入的人一眼知道要餵什麼欄位、會吐什麼。
      2. mlflow.sklearn.log_model：把訓練好的模型存進這個 run 的 artifacts。
      3. mlflow.sklearn.load_model：用 run 的 URI 把模型讀回來，驗證真的能預測。

怎麼跑：
    python 03_log_model.py
    mlflow ui          # 點進 run，左側 Artifacts 會看到 model/ 與它的 signature
"""

import os
from pathlib import Path

# 新版 MLflow 對「本地檔案紀錄（./mlruns）」預設會擋；教學要的就是這種零設定的本地紀錄，
# 因此開一個環境變數明確允許它（等於告訴 MLflow：我知道，就是要用檔案後端）。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[4] / "datasets" / "iris.csv"
TRACKING_URI = (Path(__file__).resolve().parent / "mlruns").as_uri()


def main() -> None:
    # iris 欄位：四個數值特徵 + target（數值標籤）+ target_name（花種名，僅供人讀）
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target", "target_name"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("iris-logreg-model")

    with mlflow.start_run(run_name="logreg-with-model") as run:
        model = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", acc)

        # 動詞 1：infer_signature —— 從輸入 X 與輸出 preds 推導 schema
        signature = infer_signature(X_test, preds)

        # 動詞 2：log_model —— 把模型存進這個 run，附上 signature 與一筆範例輸入
        #   新版 MLflow 用關鍵字 name=（舊版叫 artifact_path=，兩者擇一）
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.iloc[:3],
        )

        print(f"模型已記錄：accuracy={acc:.4f}")
        print("signature =", signature)

    # 動詞 3：load_model —— 用剛剛的 model_uri 把模型讀回來，確認真的能用
    loaded = mlflow.sklearn.load_model(model_info.model_uri)
    reloaded_acc = accuracy_score(y_test, loaded.predict(X_test))
    print(f"讀回模型後重新預測 accuracy={reloaded_acc:.4f}（應與上面相同）")


if __name__ == "__main__":
    main()
