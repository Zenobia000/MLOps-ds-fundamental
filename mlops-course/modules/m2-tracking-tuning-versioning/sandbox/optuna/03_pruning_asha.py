"""
03_pruning_asha.py —— 提早砍掉沒前途的 trial（pruning，技能階梯 階 2 進階）

這個檔示範什麼：
    有些 trial 一看就沒救（前幾步分數就很差），把剩下的算完是浪費。
    pruner 讓 Optuna「在訓練中途」判斷並提早終止這些 trial，省下大量時間。
    做法：
      1. 在 objective 裡分階段回報中間分數：trial.report(value, step)。
      2. 每一步問：trial.should_prune()？是的話 raise optuna.TrialPruned()。
      3. 建 study 時掛上 pruner（這裡用 MedianPruner：比同階段的中位數差就砍）。

    我們用 LogisticRegression 逐步加大 max_iter 來「模擬」分階段訓練：
    iter 太少時分數差的 trial，會在中途就被 pruner 砍掉。

怎麼跑：
    python 03_pruning_asha.py     # 觀察輸出有幾個 trial 被 PRUNED（提早終止）

延後/選配（先知道有這些，需要時再回來）：
    - 換 SuccessiveHalvingPruner（即 ASHA 的精神）：把資源（如 epoch）由少到多
      分配，先用少資源淘汰大部分 trial，只讓贏家拿到更多資源。建 study 時改成：
          optuna.create_study(direction="maximize",
                               pruner=optuna.pruners.SuccessiveHalvingPruner())
    - 多目標（例：同時要 accuracy 高、推論時間短）：改用
          study = optuna.create_study(directions=["maximize", "minimize"])
      objective 改成 `return accuracy, latency_seconds`，
      結束後看 study.best_trials（Pareto 前緣，會有多組互不支配的解，而非單一最佳）。
"""

import time
from pathlib import Path

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[4] / "datasets" / "iris.csv"

# iris 欄位：四個數值特徵 + target（數值標籤）+ target_name（花種名，僅供人讀）
_df = pd.read_csv(DATA_PATH)
_X = _df.drop(columns=["target", "target_name"])
_y = _df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=SEED, stratify=_y
)

# 用一連串遞增的 max_iter 當「訓練階段」，每階段回報一次中間分數
ITER_STEPS = [50, 100, 200, 400]


def objective(trial: optuna.Trial) -> float:
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)

    score = 0.0
    for step, max_iter in enumerate(ITER_STEPS):
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=SEED)
        score = float(
            cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
        )

        # 回報這一階段的中間分數，讓 pruner 有依據判斷
        trial.report(score, step)

        # 問 pruner：這個 trial 跟同階段別人比，是不是太差該砍了？
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score


def main() -> None:
    # 掛上 pruner：MedianPruner —— 中間分數低於「同階段所有 trial 中位數」就剪掉
    #   n_warmup_steps=1：第 0 步先不剪，給每個 trial 至少一次機會
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=30)
    elapsed = time.perf_counter() - t0

    n_pruned = len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,)))
    n_complete = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))

    print(f"總共 {len(study.trials)} 個 trial，耗時 {elapsed:.2f}s")
    print(f"  跑完 (COMPLETE)：{n_complete}")
    print(f"  提早砍掉 (PRUNED)：{n_pruned}  <- pruner 省下的算力")
    print("best_value =", round(study.best_value, 4))
    print("best_params =", study.best_params)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
