"""
sandbox/prefect/flow.py — 階 9 Prefect 最小可用單元

這個檔示範什麼：
    只用 Prefect 的三個核心動詞 —— @task / @flow / 本地 run —— 把
    「兩支既有函式」串成一個有向流程（flow）。
    重點：你不需要 server、不需要 deployment、不需要 schedule，
    一個 `python flow.py` 就能在本地跑完整條 pipeline，並由 Prefect
    自動記錄每個 task 的執行狀態（成功/失敗/重試）。

    資料與模型刻意沿用 M1 的 iris baseline——**這裡沒有任何新的 ML**。
    你已經會這段了，正因為如此，你才能把注意力全放在
    「加上 @task / @flow 之後，多了什麼」這件事上。

延後（之後需要時再回來）：
    - deployment（把 flow 註冊到 Prefect server / Cloud）
    - schedule（定時觸發）
    - blocks（外部資源連線設定）
    - 品質 gate（accuracy 不到門檻就讓 flow 失敗）—— 這是 workspace 的練習

怎麼跑：
    # 依賴不用另外裝，課程統一環境已含 prefect / pandas / scikit-learn。
    # 在本資料夾執行：
    python flow.py

    跑完你會在終端看到 Prefect 印出每個 task 的狀態流轉
    （Pending -> Running -> Completed），最後印出評估結果。
"""

from pathlib import Path

import pandas as pd
from prefect import flow, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 固定亂數種子：可重現的第一步，與 M1 baseline 綁同一顆 seed，
# 所以這條 flow 的 accuracy 應該跟你在 M1 跑出來的一致。
SEED = 42


def find_iris_csv() -> Path:
    """定位 mlops-course/datasets/iris.csv。

    用「相對本檔位置往上找」而非寫死絕對路徑，
    這樣不管從哪個工作目錄執行都找得到資料。
    本檔在 modules/m5-automation/sandbox/prefect/，往上四層即 mlops-course/。

    注意這支**刻意不加 @task**：它是純路徑計算，沒有 I/O 成本、
    不會失敗到需要重試，包成 task 只會讓執行圖多一個沒資訊量的節點。
    「什麼該是 task」的判準是「值不值得被單獨追蹤與重試」，不是「是不是函式」。
    """
    course_root = Path(__file__).resolve().parents[4]
    iris_path = course_root / "datasets" / "iris.csv"
    if not iris_path.exists():
        raise FileNotFoundError(
            f"找不到玩具資料：{iris_path}\n"
            "請確認 mlops-course/datasets/iris.csv 已存在。"
        )
    return iris_path


@task
def load_data() -> pd.DataFrame:
    """第一支 task：載入 iris 資料集。

    這正是「既有函式加個裝飾器就被 Prefect 追蹤」的示範——
    函式本體跟 M1 的載入邏輯是同一套，只是多了 @task。
    """
    df = pd.read_csv(find_iris_csv())
    print(f"[load_data] 已載入 {len(df)} 筆 iris 資料，{df['target'].nunique()} 個類別")
    return df


@task
def train_eval(df: pd.DataFrame) -> float:
    """第二支 task：訓練 LogisticRegression 並在 test set 上評估。

    沿用 M1 baseline 的作法：iris.csv 有 target（數值）與 target_name（文字），
    訓練只用數值 target，文字欄丟掉。回傳值 = test accuracy。
    """
    y = df["target"]
    X = df.drop(columns=["target", "target_name"])

    # stratify=y 讓三個花種在 train/test 的比例一致；random_state 綁 SEED 才可重現。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)

    # 在「沒看過的」test set 上評估，這才是誠實的分數。
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"[train_eval] 評估完成，accuracy = {accuracy:.3f}")
    return accuracy


@flow(name="iris-train-flow")
def main() -> float:
    """@flow：把上面兩支 task 串成一條 pipeline。

    flow 內部呼叫 task 時，Prefect 會自動建立執行圖、追蹤狀態、
    並把上一個 task 的輸出當作下一個 task 的輸入（這裡 df -> train_eval）。
    """
    df = load_data()
    accuracy = train_eval(df)
    print(f"[flow] pipeline 結束，最終 accuracy = {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    # 本地 run：直接呼叫 flow 函式即可在本機跑完整條流程。
    # 這就是 Prefect「最小可用」的用法——先把 flow 跑起來，
    # deployment / schedule 等到你真的要排程時再學。
    main()
