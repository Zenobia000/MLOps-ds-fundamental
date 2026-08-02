"""
workspace/flow.py — M5 填空：Prefect 編排 + 品質 gate

怎麼用：
    1. 在 mlops-course/ 執行：make workspace-m5
    2. 搜尋 TODO(M5- 依序把 raise NotImplementedError 換成真正程式
    3. 對照：modules/m5-automation/sandbox/prefect/flow.py
    4. 跑：python workspace/flow.py

驗收：一條會因為模型太爛而「失敗」的 flow。
    sandbox 教你把兩支 task 串起來；這裡多兩件 sandbox 沒教的事——
    先驗資料（validate_schema）、後擋爛模型（quality_gate）。
    這兩個 task 一旦 raise，Prefect 會把整條 flow 標成 Failed，
    這正是「自動化」的重點：壞掉要停下來，而不是安靜地產出爛結果。
"""

from prefect import flow, task

SEED = 42
# TODO(M5-1): 設定 accuracy 門檻（例如 0.8）；低於門檻要讓 flow 失敗
ACCURACY_THRESHOLD = None


@task
def load_features():
    """TODO(M5-2): 載入特徵（可先讀 datasets/iris.csv，之後改接 Feast）。

    提示：作法與 sandbox 的 load_data() 相同，差別只在路徑深度——
    本檔在 workspace/，course root 是 Path(__file__).resolve().parents[?]，
    對照 sandbox 的 parents[4]（那邊多了 modules/m5/sandbox/prefect 幾層）。
    M1 的 workspace/train.py 你已經算過同一件事，可直接沿用。
    """
    raise NotImplementedError("TODO(M5-2): load_features")


@task
def validate_schema(data):
    """TODO(M5-3): 檢查欄位 / 值域；不通過就 raise ValueError。

    提示：iris.csv 的欄位是 sepal_length / sepal_width / petal_length /
    petal_width / target / target_name。至少檢查兩件事——
    該有的欄位都在、數值特徵沒有 NaN 且為正數。
    這支 sandbox 沒有，是你要自己想的：資料先驗過，訓練才有意義。
    """
    raise NotImplementedError("TODO(M5-3): validate_schema")


@task
def train_and_evaluate(data) -> float:
    """TODO(M5-4): 訓練 + 回傳 test accuracy（可呼叫既有 train 邏輯）。

    提示：對照 sandbox 的 train_eval()——y 取 target 欄，
    X 丟掉 target 與 target_name，train_test_split 帶 stratify=y 與
    random_state=SEED，再用 LogisticRegression。這段你在 M1 就寫過了。
    """
    raise NotImplementedError("TODO(M5-4): train_and_evaluate")


@task
def quality_gate(accuracy: float) -> None:
    """TODO(M5-5): accuracy < ACCURACY_THRESHOLD 時 raise，擋住爛模型。

    提示：iris 太簡單，正常會落在 0.95 以上、擋不下來。
    想驗證 gate 真的有效，把 ACCURACY_THRESHOLD 暫時調到 0.99，
    確認 flow 真的被標成 Failed，再調回合理值。
    「沒失敗過的 gate 等於沒有 gate。」
    """
    raise NotImplementedError("TODO(M5-5): quality_gate")


@flow(name="workspace-train-flow")
def train_flow():
    # TODO(M5-6): load → validate → train → quality_gate 串起來
    raise NotImplementedError("TODO(M5-6): 串 task")


if __name__ == "__main__":
    train_flow()
