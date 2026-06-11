"""Smart Factory MLOps —— 套件根（canonical import 起點）。

整個專案的 import 一律以 ``src`` 為套件根，例如::

    from src.utils.config import load_config
    from src.utils.seed import set_seed
    from src.models.tabular import XGBoostMaintenanceModel

從 repo 根（含 conf/、data/）以模組方式執行各進入點::

    python -m src.training.train          # 通用訓練入口
    python -m src.tuning.optuna_tuning    # Optuna 自動調參

pytest 以 ``pythonpath=["."]`` 設定，確保測試能解析 ``src`` 套件。

子套件分工（跨 agent 契約，詳見 docs/project-structure.md）：
    - :mod:`src.utils`      共用工具（config / logging / seed）
    - :mod:`src.data`       資料載入與驗證
    - :mod:`src.features`   特徵工程與 Feast 介接
    - :mod:`src.models`     三型態模型（tabular / timeseries / vision）
    - :mod:`src.training`   通用訓練與評估
    - :mod:`src.tuning`     Optuna 超參搜尋
    - :mod:`src.serving`    推論前後處理
    - :mod:`src.monitoring` 漂移與指標監控
"""

__version__ = "0.1.0"
