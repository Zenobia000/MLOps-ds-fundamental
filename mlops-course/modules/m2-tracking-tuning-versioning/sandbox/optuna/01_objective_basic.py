"""
01_objective_basic.py —— Optuna 最小可用單元：自動找超參（技能階梯 階 2）

這個檔示範什麼：
    02_params_metrics.py 是「手動」掃三組超參。Optuna 把這件事「自動化」：
    你只寫一個 objective(trial)，它自己決定下一組要試什麼、試很多次、回報最佳。
    只用四個動詞：
      1. trial.suggest_*  —— 在 objective 裡宣告「這個超參的搜尋範圍」。
      2. create_study      —— 建一個研究（direction 指定要最大化還是最小化）。
      3. study.optimize    —— 跑 n_trials 次，自動探索。
      4. study.best_params / best_value —— 拿到最好的那組。

    這個檔「還沒接 MLflow」，先把 Optuna 本身玩熟；下一個檔 02 才把每個 trial 寫進 MLflow。

怎麼跑：
    python 01_objective_basic.py
"""

from pathlib import Path

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[4] / "datasets" / "iris.csv"

# 在模組層級讀一次資料，objective 會重複用到
# iris 欄位：四個數值特徵 + target（數值標籤）+ target_name（花種名，僅供人讀）
_df = pd.read_csv(DATA_PATH)
_X = _df.drop(columns=["target", "target_name"])
_y = _df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=SEED, stratify=_y
)


def objective(trial: optuna.Trial) -> float:
    """一個 trial = 試一組超參，回傳要被最佳化的分數（這裡是交叉驗證 accuracy）。"""
    # 動詞：suggest_* —— 宣告搜尋空間，Optuna 會在範圍內幫你挑值
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)       # 對數尺度掃正則化強度
    max_iter = trial.suggest_int("max_iter", 100, 500, step=100)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=SEED)

    # 用 5-fold 交叉驗證的平均 accuracy 當分數（比單一切分更穩）
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return float(scores.mean())


def main() -> None:
    # 動詞：create_study —— direction="maximize" 因為 accuracy 越大越好
    #   sampler 設 seed，讓整個搜尋過程可重現
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # 動詞：optimize —— 自動跑 20 個 trial
    study.optimize(objective, n_trials=20)

    # 動詞：best_params / best_value —— 拿最佳結果
    print(f"試了 {len(study.trials)} 組超參")
    print("best_value (cv accuracy) =", round(study.best_value, 4))
    print("best_params =", study.best_params)


if __name__ == "__main__":
    # 安靜一點，只印我們自己的輸出（拿掉這行可看到每個 trial 的 log）
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
