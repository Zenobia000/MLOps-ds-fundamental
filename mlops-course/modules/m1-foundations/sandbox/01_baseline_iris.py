"""
01_baseline_iris.py — 階 0：純 sklearn baseline（無任何 MLOps 工具）

這個檔示範什麼：
    用最熟悉的 sklearn，在 iris 玩具資料上訓一個 LogisticRegression baseline，
    印出 test accuracy。重點不是模型多強，而是建立「可重現」心智——
    固定 seed，今天跑、明天跑、別人跑，都得到一模一樣的數字。
    這支腳本「完全不碰 MLflow / Optuna / DVC」，是後續所有模組的起點。

怎麼跑：
    cd 到本檔所在資料夾後，直接：
        python 01_baseline_iris.py

    或從 repo 任意位置用絕對/相對路徑：
        python modules/m1-foundations/sandbox/01_baseline_iris.py

預期輸出（會穩定重現）：
    讀到 150 筆資料、3 個類別，最後印出一行 test accuracy。
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 固定亂數種子：可重現的第一步。任何用到隨機性的地方都綁同一顆 seed。
SEED = 42


def find_iris_csv() -> Path:
    """定位 datasets/iris.csv。

    用「相對本檔位置往上找」而非寫死絕對路徑，
    這樣不管從哪個工作目錄執行都找得到資料。
    本檔在 modules/m1-foundations/sandbox/，往上三層即 mlops-course/。
    """
    course_root = Path(__file__).resolve().parents[3]
    iris_path = course_root / "datasets" / "iris.csv"
    if not iris_path.exists():
        raise FileNotFoundError(
            f"找不到玩具資料：{iris_path}\n"
            "請確認 mlops-course/datasets/iris.csv 已存在。"
        )
    return iris_path


def load_iris(iris_path: Path):
    """讀 iris.csv，回傳特徵 X 與標籤 y。

    不假設欄位的確切命名：把「最後一欄」當作標籤（iris 的慣例），
    其餘數值欄當特徵。這樣對 sepal_length 或 SepalLengthCm 等不同
    header 都能照跑，降低初學者卡在欄名的機率。
    """
    df = pd.read_csv(iris_path)
    label_col = df.columns[-1]            # 最後一欄是花種類別
    X = df.drop(columns=[label_col])      # 其餘為特徵
    y = df[label_col]
    print(f"讀到資料：{len(df)} 筆、{y.nunique()} 個類別（標籤欄＝{label_col}）")
    return X, y


def main() -> None:
    iris_path = find_iris_csv()
    X, y = load_iris(iris_path)

    # train_test_split 帶 random_state＝SEED：切分方式可重現。
    # stratify=y 讓三個花種在 train/test 的比例一致，baseline 更穩。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # LogisticRegression 也綁 random_state，確保每次訓練結果一致。
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)

    # 在「沒看過的」test set 上評估，這才是誠實的 baseline 分數。
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Baseline test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
