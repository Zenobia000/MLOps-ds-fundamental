"""模型註冊：把訓練產物送進 MLflow Model Registry 並打 alias。

為什麼獨立成一個模組
--------------------
註冊是「訓練完成之後」的動作，跟訓練迴圈是不同的關注點：

- ``train.py``（走 ``make train-*`` / DVC stage）訓練完直接呼叫它
- ``pipelines/training_pipeline.py``（走 Prefect flow）也呼叫同一支

兩條路徑共用同一份註冊邏輯，才不會出現「用 make 訓練會註冊、用 flow 訓練不會」
這種不一致。之前這段邏輯藏在 ``train.py`` 的私有函式裡，flow 那條路只好跳過註冊。

註冊的前提（重要）
------------------
**只有通過品質門檻的模型才該被註冊。** 本模組不自己判斷門檻——判斷在
``src/training/evaluate.py`` 的 ``quality_gate()``，呼叫端負責先問過它再來註冊。
把「判斷」與「執行」分開，是為了讓門檻只有一個真相來源。

失敗處理
--------
註冊失敗（離線、registry 沒起、權限不足）**不拋例外**，回傳 ``None`` 並記警告。
理由：模型本身已經訓練好也存在本地了，registry 連不上不該讓整條訓練管線白跑。
呼叫端要能分辨「沒註冊」與「註冊成功」——所以回傳值是 ``str | None`` 而非 bool。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "champion"


def registered_name(config: Mapping[str, Any], model_name: str) -> str:
    """組出 registry 上的模型名稱：``<project>-<model>``。

    集中在這裡是為了讓「註冊時寫入的名字」與「服務端查詢的名字」用同一個規則，
    避免兩邊各自字串拼接而慢慢對不上。
    """
    project = str(config.get("project", "smart-factory"))
    return f"{project}-{model_name}"


def _extract_uri(model: Any) -> str | None:
    """從呼叫端傳進來的東西取出 model_uri。

    容許三種形態，因為兩條呼叫路徑手上的物件不同：
      1. 字串本身就是 uri（``runs:/<id>/model`` 或 ``models:/<name>/<v>``）
      2. MLflow 的 ``ModelInfo``，有 ``.model_uri``
      3. 其他物件 → 取不到就回 None，由呼叫端決定怎麼辦
    """
    if isinstance(model, str):
        return model
    uri = getattr(model, "model_uri", None)
    return str(uri) if uri else None


def register_model(
    model: Any,
    metrics: Mapping[str, float] | None = None,
    config: Mapping[str, Any] | None = None,
    *,
    model_name: str | None = None,
    alias: str = DEFAULT_ALIAS,
) -> str | None:
    """把模型註冊進 MLflow Registry，並把 ``alias`` 指向這個新版本。

    參數順序刻意對齊 ``pipelines/training_pipeline.py`` 既有的呼叫方式
    ``register_model(model, metrics, config)``，讓那邊不必改。

    Args:
        model: model uri 字串，或帶有 ``model_uri`` 屬性的物件（如 MLflow ModelInfo）。
        metrics: 這次訓練的指標；僅作為 tag 記錄，**不影響是否註冊**。
        config: 專案設定（讀 ``project`` 與 ``mlflow.tracking_uri``）。
        model_name: registry 名稱的模型部分；省略時從 config 的 ``active_model`` 取。
        alias: 要指向新版本的別名，預設 ``champion``。

    Returns:
        成功時回傳 ``"<registry_name>/<version>"``；失敗或取不到 uri 時回傳 ``None``。
    """
    cfg = dict(config or {})
    uri = _extract_uri(model)
    if not uri:
        logger.warning(
            "register_model 收到無法解析 model_uri 的物件（%s），略過註冊。", type(model).__name__
        )
        return None

    name = registered_name(cfg, model_name or str(cfg.get("active_model", "model")))

    try:
        import mlflow  # 延遲匯入：沒裝 mlflow 的環境仍可 import 本模組

        tracking_uri = cfg.get("mlflow", {}).get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        result = mlflow.register_model(uri, name)
        client = mlflow.tracking.MlflowClient()
        # 新版 MLflow 用 alias 取代 stage；alias 指向「當前該用哪一版」。
        client.set_registered_model_alias(name, alias, result.version)

        if metrics:
            # 指標寫成 tag，讓 registry 頁面上看得到「這一版當時考幾分」，
            # 不必回頭翻 run 才知道。
            for key, value in metrics.items():
                try:
                    client.set_model_version_tag(
                        name, result.version, f"metric.{key}", f"{value:.6g}"
                    )
                except Exception:  # noqa: BLE001 — tag 失敗不該影響註冊本身
                    logger.debug("寫入 metric tag 失敗：%s", key)

        logger.info("已註冊 %s v%s 並將 alias=%s 指向它", name, result.version, alias)
        return f"{name}/{result.version}"

    except Exception as exc:  # noqa: BLE001 — 離線 / registry 未啟動時要能繼續
        logger.warning("模型註冊失敗（%s）；本地產物仍在，管線不中斷。", exc)
        return None
