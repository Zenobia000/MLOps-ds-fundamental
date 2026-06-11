"""線上服務指標匯出器（Prometheus）。

提供與模型型態無關的指標封裝，供 serving 層在每次推論時呼叫，
再由 Prometheus 透過 ``/metrics`` 端點抓取，最終於 Grafana 視覺化。

匯出指標：
- ``inference_requests_total``：請求計數（QPS 由 Prometheus 對其求 rate）。
- ``inference_latency_seconds``：推論延遲直方圖（可算 p50/p95/p99）。
- ``model_prediction``：預測值分布直方圖（監控分布漂移的線上代理指標）。
- ``inference_errors_total``：錯誤計數。

設計：所有指標掛在傳入的 ``CollectorRegistry`` 上，避免測試間全域註冊衝突；
``model`` 與 ``status`` 等以 label 區分，方便同一服務承載多模型。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

try:  # pragma: no cover - 取決於安裝版本
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    CollectorRegistry = object  # type: ignore
    Counter = Histogram = object  # type: ignore
    generate_latest = None  # type: ignore
    _PROM_AVAILABLE = False
    _IMPORT_ERROR = exc

# 延遲直方圖預設 bucket（秒），涵蓋毫秒級到秒級推論。
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def build_registry() -> CollectorRegistry:
    """建立一個全新的 Prometheus registry。

    每個服務實例（或測試案例）使用獨立 registry，避免全域 ``REGISTRY``
    重複註冊同名指標而拋錯。
    """
    _ensure_available()
    return CollectorRegistry()


def _ensure_available() -> None:
    if not _PROM_AVAILABLE:
        raise ImportError(
            "未偵測到 prometheus_client，請先 `pip install prometheus-client`。"
            f"原始 import 錯誤：{_IMPORT_ERROR!r}"
        )


class PredictionMetrics:
    """預測指標收集器，封裝四組 Prometheus 指標。

    Example:
        >>> reg = build_registry()
        >>> m = PredictionMetrics(registry=reg)
        >>> with m.track_latency(model="tabular"):
        ...     pred = 0.83
        >>> m.observe_prediction(pred, model="tabular")
        >>> m.export()  # bytes，供 /metrics 端點回傳
    """

    def __init__(
        self,
        registry: CollectorRegistry,
        namespace: str = "smartfactory",
        prediction_buckets: tuple[float, ...] | None = None,
    ) -> None:
        """初始化指標。

        Args:
            registry: Prometheus collector registry。
            namespace: 指標名稱前綴，便於多服務區隔。
            prediction_buckets: 預測值分布的 bucket 邊界；None 時用 0~1（適配機率輸出）。
        """
        _ensure_available()
        self._registry = registry
        pred_buckets = prediction_buckets or (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

        self.requests_total = Counter(
            f"{namespace}_inference_requests_total",
            "推論請求總數",
            labelnames=("model", "status"),
            registry=registry,
        )
        self.errors_total = Counter(
            f"{namespace}_inference_errors_total",
            "推論錯誤總數",
            labelnames=("model", "error_type"),
            registry=registry,
        )
        self.latency_seconds = Histogram(
            f"{namespace}_inference_latency_seconds",
            "推論延遲（秒）",
            labelnames=("model",),
            buckets=_LATENCY_BUCKETS,
            registry=registry,
        )
        self.prediction = Histogram(
            f"{namespace}_model_prediction",
            "模型預測值分布",
            labelnames=("model",),
            buckets=pred_buckets,
            registry=registry,
        )

    @contextmanager
    def track_latency(self, model: str) -> Iterator[None]:
        """量測一段推論程式區塊的延遲並計入請求計數。

        正常結束記為 status="ok"；若拋例外則記 status="error" 與錯誤型別後再 re-raise。
        """
        start = perf_counter()
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - 記錄後原樣往上拋
            self.requests_total.labels(model=model, status="error").inc()
            self.errors_total.labels(model=model, error_type=type(exc).__name__).inc()
            raise
        else:
            self.requests_total.labels(model=model, status="ok").inc()
        finally:
            self.latency_seconds.labels(model=model).observe(perf_counter() - start)

    def observe_prediction(self, value: float, model: str) -> None:
        """記錄單筆預測值，用於線上監控預測分布漂移。"""
        self.prediction.labels(model=model).observe(float(value))

    def export(self) -> bytes:
        """輸出 Prometheus 文字格式，供 ``/metrics`` 端點回傳。"""
        return generate_latest(self._registry)
