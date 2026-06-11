"""
02_mlflow_callback.py —— 每個 Optuna trial = 一個 MLflow run（技能階梯 階 1 × 階 2）

這個檔示範什麼：
    這是 M2 的核心概念：「HPO 產生大量實驗，MLflow 負責記錄與比較」。
    我們在 objective 裡，用 mlflow.start_run(nested=True) 把「每一個 trial」
    包成「一個 MLflow run」——於是 20 個 trial 就變成 20 個可在 UI 並排比較的 run。

    為什麼用 nested run（巢狀）：
      最外層開一個 parent run 代表「這整場 HPO 搜尋」，
      每個 trial 是它底下的 child run。UI 上會看到一棵樹，收斂過程一目了然。

    （補充）Optuna 官方也有現成的 MLflowCallback（在 optuna-integration 套件），
     一行就能把 trial 寫進 MLflow；這裡刻意手寫 nested run，讓你看清楚背後發生什麼。

怎麼跑：
    python 02_mlflow_callback.py
    mlflow ui     # 進實驗 iris-optuna-hpo，展開 parent run 看 20 個 child run 比較收斂
"""

import os
from pathlib import Path

# 新版 MLflow 對「本地檔案紀錄（./mlruns）」預設會擋；教學要的就是這種零設定的本地紀錄，
# 因此開一個環境變數明確允許它（等於告訴 MLflow：我知道，就是要用檔案後端）。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[4] / "datasets" / "iris.csv"
TRACKING_URI = (Path(__file__).resolve().parent / "mlruns").as_uri()

# iris 欄位：四個數值特徵 + target（數值標籤）+ target_name（花種名，僅供人讀）
_df = pd.read_csv(DATA_PATH)
_X = _df.drop(columns=["target", "target_name"])
_y = _df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=SEED, stratify=_y
)


def objective(trial: optuna.Trial) -> float:
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 500, step=100)

    # 關鍵：nested=True —— 這個 run 掛在外層 parent run 底下，成為一個 child
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=SEED)
        score = float(
            cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
        )

        # 這個 trial 的超參與分數，原封不動記進它自己的 run
        mlflow.log_params({"C": C, "max_iter": max_iter})
        mlflow.log_metric("cv_accuracy", score)

    return score


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("iris-optuna-hpo")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # 最外層 parent run 代表「這整場 HPO」；optimize 期間開的 child run 都掛在它下面
    with mlflow.start_run(run_name="optuna-search"):
        study.optimize(objective, n_trials=20)

        # 搜尋結束，把全場最佳結果也記在 parent run 上，方便一眼看到結論
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_accuracy", study.best_value)

    print("best_value =", round(study.best_value, 4))
    print("best_params =", study.best_params)
    print("用 `mlflow ui` 進 iris-optuna-hpo，展開 optuna-search 看 20 個 child run")


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
