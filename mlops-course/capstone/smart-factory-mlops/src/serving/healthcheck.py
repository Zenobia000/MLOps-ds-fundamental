"""服務健康探測：canary 部署的決策依據。

這支存在的理由
--------------
部署管線要在「推廣新版」與「回滾」之間做決定，而那個決定必須基於**真實訊號**。
在此之前 ``canary_probe`` 回傳寫死的 ``1.0``，於是回滾分支永遠走不到——
一個永遠說 OK 的守門員，比沒有守門員更危險，因為它讓管線看起來是綠的。

判定原則（重要的教學點）
------------------------
``degraded`` **算失敗**。

服務的 ``/healthz`` 有三種結果：``ok``（兩個模型都就緒）、``degraded``
（活著但有模型沒載到）、連不上。canary 的問題不是「服務活著嗎」，而是
**「這個新版本可以承接生產流量嗎」**——一個少了影像模型的版本顯然不行。
liveness 探測可以容忍 degraded（不該重啟它），canary 不行。

只用標準函式庫
--------------
刻意不引入 ``requests``：這支會被烤進推論映像的依賴考量範圍，
而 ``urllib`` 已經夠用。少一個依賴就少一個要跟著鎖版本的東西。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_ATTEMPTS = 5
DEFAULT_TIMEOUT = 2.0

#: 探測要打的端點。BentoML 依 @bentoml.api 的方法名產生路由，且為 POST。
HEALTH_PATH = "/healthz"


def _base_url(config: Mapping[str, Any] | None = None) -> str:
    """決定要探測哪個位址。

    優先序：環境變數 > 設定檔 > 預設 localhost。
    canary 通常跑在另一個位址／連接埠，所以環境變數要能覆蓋。
    """
    env = os.environ.get("CANARY_BASE_URL")
    if env:
        return env.rstrip("/")
    if config:
        url = config.get("deploy", {}).get("canary_base_url")
        if url:
            return str(url).rstrip("/")
    return DEFAULT_BASE_URL


def _probe_once(url: str, timeout: float) -> bool:
    """打一次健康檢查；只有 status == "ok" 才算成功。

    回傳 False 的三種情況都視為同一件事——這個版本現在不能接流量：
      - 連不上 / 逾時
      - HTTP 非 2xx
      - 回應是 degraded（有模型沒載到）
    """
    request = urllib.request.Request(
        url, data=b"{}", method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 300:
                logger.debug("健康檢查回應 HTTP %s", response.status)
                return False
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug("健康檢查失敗：%s", exc)
        return False

    status = str(payload.get("status", "")).lower()
    if status != "ok":
        logger.debug(
            "健康檢查回報 %s（tabular=%s, vision=%s）—— canary 視為失敗",
            status or "(無 status)",
            payload.get("tabular_model_loaded"),
            payload.get("vision_model_loaded"),
        )
        return False
    return True


def probe(
    target: str | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT,
) -> float:
    """對候選版本連續探測，回傳成功率（0.0–1.0）。

    參數形態對齊 ``pipelines/deployment_pipeline.py`` 既有的 ``prober(image_tag)`` 呼叫。

    Args:
        target: 部署管線傳進來的 image tag，僅作為 log 標示；**實際位址**由
            ``CANARY_BASE_URL`` 環境變數或 ``config['deploy']['canary_base_url']`` 決定。
            image tag 本身不足以推導出服務位址，這一點要在課堂上講清楚。
        config: 專案設定，用來取 canary 位址。
        attempts: 探測次數。單次成功不足以判斷穩定性，預設連打 5 次。
        timeout: 每次的逾時秒數。

    Returns:
        成功次數 / 總次數。服務完全連不上時為 ``0.0``。
    """
    if attempts <= 0:
        raise ValueError("attempts 必須為正整數")

    url = f"{_base_url(config)}{HEALTH_PATH}"
    logger.info("canary 探測 %s（image=%s，%d 次）", url, target or "-", attempts)

    successes = sum(1 for _ in range(attempts) if _probe_once(url, timeout))
    rate = successes / attempts

    log = logger.info if rate == 1.0 else logger.warning
    log("canary 探測結果：%d/%d 成功（成功率 %.2f）", successes, attempts, rate)
    return rate
