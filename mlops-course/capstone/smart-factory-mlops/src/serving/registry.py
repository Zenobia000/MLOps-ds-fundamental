"""Registry 查詢：服務端與部署端「該用哪一版模型」的單一入口。

與 ``src/training/registry.py`` 的分工
--------------------------------------
訓練側**寫入** registry（註冊版本、移動 alias）；本模組**只讀**。
方向分開是刻意的——服務端不該有能力改動 registry，避免推論路徑意外寫壞生產狀態。

解析順序（由明確到寬鬆）
------------------------
1. ``models:/<name>@<alias>``  — alias（champion）指到的版本，**建議的正式來源**
2. ``models:/<name>/<stage>``  — 舊式 stage（Production/Staging），相容既有設定
3. 最新版本號              — 前兩者都沒有時，取版本號最大的那個

第 3 步是**最後手段**：「最新」不等於「最好」，它可能是一個剛註冊、還沒被驗證的版本。
所以取到它時會記 warning，讓維運看得到自己正在用一個沒有 alias 背書的模型。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "champion"


def _client(config: Mapping[str, Any]):
    """建立 MlflowClient；未裝 mlflow 或設定不全時回傳 None（由呼叫端決定後援）。"""
    try:
        import mlflow

        tracking_uri = config.get("mlflow", {}).get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        return mlflow.tracking.MlflowClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("無法建立 MLflow client（%s）", exc)
        return None


def resolve_latest(
    config: Mapping[str, Any],
    *,
    model_name: str | None = None,
    alias: str = DEFAULT_ALIAS,
) -> str | None:
    """解析出「現在該部署／載入哪一版」的 model uri。

    參數形態對齊 ``pipelines/deployment_pipeline.py`` 既有的 ``resolver(config)`` 呼叫。

    Args:
        config: 專案設定；讀 ``project``、``active_model``、``mlflow.tracking_uri``。
        model_name: registry 名稱的模型部分；省略時取 ``active_model``。
        alias: 優先採用的別名，預設 ``champion``。

    Returns:
        ``models:/...`` 形式的 uri；registry 不可用或查無此模型時回傳 ``None``，
        **不拋例外**——呼叫端（部署管線 / 服務載入）自己決定要不要回退。
    """
    cfg = dict(config)
    project = str(cfg.get("project", "smart-factory"))
    name = f"{project}-{model_name or cfg.get('active_model', 'model')}"

    client = _client(cfg)
    if client is None:
        return None

    # 1) alias — 正式來源。有人明確指定「這一版是 champion」。
    try:
        version = client.get_model_version_by_alias(name, alias)
        logger.info("解析到 alias：%s@%s → v%s", name, alias, version.version)
        return f"models:/{name}@{alias}"
    except Exception:  # noqa: BLE001 — 沒有這個 alias 是正常情況，往下試
        logger.debug("模型 %s 沒有 alias=%s，改試 stage", name, alias)

    # 2) stage — 相容既有設定（conf/config.yaml 的 serving.*_model_stage）
    serving_cfg = cfg.get("serving", {})
    stage = serving_cfg.get("tabular_model_stage") or serving_cfg.get("vision_model_stage")
    if stage:
        try:
            versions = client.get_latest_versions(name, stages=[stage])
            if versions:
                logger.info("解析到 stage：%s/%s → v%s", name, stage, versions[0].version)
                return f"models:/{name}/{stage}"
        except Exception:  # noqa: BLE001 — 新版 MLflow 可能已移除 stage API
            logger.debug("stage 查詢不可用或查無結果：%s/%s", name, stage)

    # 3) 最新版本號 — 最後手段，沒有任何背書
    try:
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            newest = max(versions, key=lambda v: int(v.version))
            logger.warning(
                "模型 %s 沒有 alias=%s 也沒有 stage，退而使用最新版本 v%s。"
                "「最新」不等於「已驗證」——請確認是否該打上 alias。",
                name,
                alias,
                newest.version,
            )
            return f"models:/{name}/{newest.version}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("查詢模型版本失敗（%s）", exc)

    logger.warning("registry 中找不到模型 %s", name)
    return None
