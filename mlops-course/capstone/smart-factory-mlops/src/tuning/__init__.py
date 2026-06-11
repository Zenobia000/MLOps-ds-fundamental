"""調參層（tuning）。

以 Optuna 進行超參搜尋，搜尋空間 / trial 數 / pruner 全部來自 ``conf/hpo/``：

- :mod:`src.tuning.optuna_tuning`：讀 ``conf/hpo/<model>.yaml`` 的 ``search_space``，
  建立 study 與 pruner，每個 trial 包成一個 MLflow nested run，回最佳超參。

從 repo 根執行::

    python -m src.tuning.optuna_tuning             # 對 active_model 調參
    python -m src.tuning.optuna_tuning --model xgboost
"""

from src.tuning.optuna_tuning import run_study, suggest_from_space

__all__ = ["run_study", "suggest_from_space"]
