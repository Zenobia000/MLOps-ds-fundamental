"""編排層（Prefect）：把 src 的步驟串成可重現的 DAG。

三條主管線
----------
- ``feature_pipeline``：載入 → 清洗 → 驗證 → 特徵工程 →（選配）materialize 進 Feast。
- ``training_pipeline``：取特徵 → 訓練 → 評估 → 品質門檻 →（選配）註冊。
- ``deployment_pipeline``：載入註冊模型 → 打包 → canary 部署佔位。

設計原則
--------
- 每個 ``@task`` 只做一件事；flow 負責資料流與順序。
- 對 ``src.*`` 一律「軟相依」：sibling 模組尚未建好時退回小樣本後援，
  讓本地 ``python -m pipelines.training_pipeline`` 永遠跑得起來。
- Prefect 未安裝時提供 no-op 裝飾器，確保 import 不破。
"""

from __future__ import annotations

try:  # pragma: no cover - 環境相依
    from prefect import flow, task  # type: ignore
except Exception:  # noqa: BLE001 - Prefect 未安裝時的優雅退化

    def task(fn=None, **_kwargs):  # type: ignore
        """no-op task 裝飾器（Prefect 未安裝時）。"""

        def _wrap(f):
            return f

        return _wrap(fn) if callable(fn) else _wrap

    def flow(fn=None, **_kwargs):  # type: ignore
        """no-op flow 裝飾器（Prefect 未安裝時）。"""

        def _wrap(f):
            return f

        return _wrap(fn) if callable(fn) else _wrap


__all__ = ["flow", "task"]
