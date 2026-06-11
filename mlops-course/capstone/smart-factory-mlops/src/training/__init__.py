"""訓練層（training）。

通用訓練與評估入口，與模型型態無關（差異收斂在 :mod:`src.models`）：

- :mod:`src.training.train`：讀 config 選 ``active_model``，訓練、MLflow 記錄
  params / metrics / signature / model，並可註冊 registry 打 alias。
- :mod:`src.training.evaluate`：計算指標並套用品質門檻（quality gate），
  供 CI/CD 的 CT 流程決定「是否註冊 / 部署」。

從 repo 根執行::

    python -m src.training.train          # 用 conf/config.yaml 的 active_model
    python -m src.training.train --model lstm
"""

from src.training.evaluate import GateResult, evaluate_classification, quality_gate

__all__ = ["GateResult", "evaluate_classification", "quality_gate"]
