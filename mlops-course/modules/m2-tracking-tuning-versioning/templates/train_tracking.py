"""
workspace/train_tracking.py — M2 填空模板（MLflow + Optuna）

前置：已完成 M1 的 workspace/train.py。
對照沙盒：
    modules/m2-tracking-tuning-versioning/sandbox/mlflow/
    modules/m2-tracking-tuning-versioning/sandbox/optuna/

怎麼用：
    make workspace-m2
    搜尋 TODO(M2- 依序填空
    python workspace/train_tracking.py
    make mlflow-ui   # 驗收：看得到 experiment / metrics / model

DVC（CLI，不在本檔）：
    對 datasets/iris.csv 做 dvc init / add，把 .dvc 檔 commit 進 Git。
    詳見本模組 README 整合任務。
"""

import os
from pathlib import Path

# 本地檔案後端教學用；沙盒同樣有這行
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
COURSE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = COURSE_ROOT / "datasets" / "iris.csv"
# mlruns 寫在 mlops-course/ 下，跟 make mlflow-ui 預設一致
TRACKING_URI = (COURSE_ROOT / "mlruns").as_uri()


def load_xy():
    """沿用 M1：讀 iris，回傳 X, y（丟掉 target_name 文字欄）。"""
    df = pd.read_csv(DATA_PATH)
    # iris.csv 有 target + target_name；訓練只用數值 target
    y = df["target"]
    X = df.drop(columns=["target", "target_name"])
    return X, y


def train_one(C: float, X_train, y_train, X_test, y_test):
    """訓一顆 LogisticRegression，回傳 (model, accuracy, f1)。"""
    model = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    return model, acc, f1, preds


def run_mlflow_baseline(X_train, y_train, X_test, y_test) -> None:
    """TODO(M2-1)：單次 run — log_param / log_metric / log_model + signature。

    對照：sandbox/mlflow/03_log_model.py
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    # TODO(M2-1a): set_experiment（自己取一個 experiment 名）
    raise NotImplementedError("TODO(M2-1a): mlflow.set_experiment(...)")

    # TODO(M2-1b): with mlflow.start_run(...): 訓模型、log_param("C", ...)、
    #   log_metric("accuracy" / "f1")、infer_signature、mlflow.sklearn.log_model
    raise NotImplementedError("TODO(M2-1b): start_run + log_*")


def objective(trial, X_train, y_train, X_test, y_test) -> float:
    """TODO(M2-2)：Optuna objective —— 建議 C，nested MLflow run，回傳 accuracy。

    對照：sandbox/optuna/02_mlflow_callback.py
    """
    # TODO(M2-2a): C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    raise NotImplementedError("TODO(M2-2a): trial.suggest_float")

    # TODO(M2-2b): with mlflow.start_run(nested=True): 訓練、log、return acc
    raise NotImplementedError("TODO(M2-2b): nested run + return accuracy")


def run_optuna(X_train, y_train, X_test, y_test) -> None:
    """TODO(M2-3)：create_study → optimize → 把 best_params 記到 parent run。"""
    import optuna

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("workspace-iris-optuna")

    # TODO(M2-3): parent run 包住 study.optimize；結束後 log_param best_* 
    raise NotImplementedError("TODO(M2-3): Optuna study + parent MLflow run")


def main() -> None:
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 先做 M2-1，再做 M2-2/3（可註解其中一段分開驗證）
    run_mlflow_baseline(X_train, y_train, X_test, y_test)
    # run_optuna(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
