"""
workspace/train.py — M1 主線填空模板（階 0）

怎麼用：
    1. 在 mlops-course/ 執行：make workspace-m1
    2. 搜尋 TODO(M1- 依序把 raise NotImplementedError 換成真正程式
    3. 對照：modules/m1-foundations/sandbox/01_baseline_iris.py
    4. 跑：python workspace/train.py（兩次 accuracy 必須一樣）

驗收：可重現的 iris LogisticRegression baseline。
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# TODO(M1-1): 把 SEED 設成固定整數（建議 42），之後所有 random_state 都用它
SEED = None  # ← 改這裡


def find_iris_csv() -> Path:
    """定位 mlops-course/datasets/iris.csv。

    提示：本檔在 workspace/，course root 是 Path(__file__).resolve().parents[?]
    對照 sandbox 的 parents[3]——那邊多了 modules/m1/sandbox 兩層。
    """
    # TODO(M1-2): 算出 course_root，回傳 course_root / "datasets" / "iris.csv"
    # 若檔案不存在，raise FileNotFoundError（訊息寫清楚路徑）
    raise NotImplementedError("TODO(M1-2): find_iris_csv — 見上方 docstring")


def load_iris(iris_path: Path):
    """讀 CSV，回傳 (X, y)。

    提示：最後一欄當標籤、其餘當特徵（與 sandbox 相同，相容不同 header）。
    對照 sandbox load_iris()。
    """
    # TODO(M1-3): pd.read_csv → 拆 X / y → print 筆數與類別數 → return X, y
    raise NotImplementedError("TODO(M1-3): load_iris")


def main() -> None:
    if SEED is None:
        raise NotImplementedError("TODO(M1-1): 先設定 SEED")

    iris_path = find_iris_csv()
    X, y = load_iris(iris_path)

    # TODO(M1-4): train_test_split
    #   test_size=0.2, random_state=SEED, stratify=y
    # 對照 sandbox main() 裡的 train_test_split 呼叫
    raise NotImplementedError("TODO(M1-4): train_test_split")
    # X_train, X_test, y_train, y_test = ...

    # TODO(M1-5): 建立 LogisticRegression(max_iter=1000, random_state=SEED)、fit、predict
    # 對照 sandbox：model.fit / model.predict
    raise NotImplementedError("TODO(M1-5): 訓練與預測")
    # model = ...
    # preds = ...

    # TODO(M1-6): 用 accuracy_score 算 test accuracy 並 print
    # 格式建議：print(f"Baseline test accuracy: {acc:.4f}")
    raise NotImplementedError("TODO(M1-6): 印出 test accuracy")

    # M2 預留接點（先別刪；解鎖後用 make workspace-m2）
    # M2: 之後在這裡接 MLflow log_*


if __name__ == "__main__":
    main()
