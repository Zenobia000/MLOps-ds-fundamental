"""
workspace/flow.py — M5 填空：Prefect 編排 + 品質 gate

對照：modules/m5-automation/sandbox/prefect/flow.py
怎麼跑：python workspace/flow.py
"""

from prefect import flow, task

SEED = 42
# TODO(M5-1): 設定 accuracy 門檻（例如 0.8）；低於門檻要讓 flow 失敗
ACCURACY_THRESHOLD = None


@task
def load_features():
    """TODO(M5-2): 載入特徵（可先讀 datasets/iris.csv，之後改接 Feast）。"""
    raise NotImplementedError("TODO(M5-2): load_features")


@task
def validate_schema(data):
    """TODO(M5-3): 檢查欄位 / 值域；不通過就 raise ValueError。"""
    raise NotImplementedError("TODO(M5-3): validate_schema")


@task
def train_and_evaluate(data) -> float:
    """TODO(M5-4): 訓練 + 回傳 test accuracy（可呼叫既有 train 邏輯）。"""
    raise NotImplementedError("TODO(M5-4): train_and_evaluate")


@task
def quality_gate(accuracy: float) -> None:
    """TODO(M5-5): accuracy < ACCURACY_THRESHOLD 時 raise，擋住爛模型。"""
    raise NotImplementedError("TODO(M5-5): quality_gate")


@flow(name="workspace-train-flow")
def train_flow():
    # TODO(M5-6): load → validate → train → quality_gate 串起來
    raise NotImplementedError("TODO(M5-6): 串 task")


if __name__ == "__main__":
    train_flow()
