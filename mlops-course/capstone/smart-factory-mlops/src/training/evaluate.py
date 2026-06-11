"""評估指標與品質門檻（quality gate）。

職責：
    1. 依任務型態計算標準指標（分類 / 迴歸）。
    2. 對主指標套用門檻判定，回傳「是否通過」，供 CT pipeline 決定
       是否註冊 / 部署模型（避免劣化模型進 registry）。

門檻與主指標來自 ``conf/train/default.yaml`` 的 ``evaluation`` 區段，不寫死。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GateResult:
    """品質門檻判定結果（不可變）。"""

    passed: bool
    metric: str
    value: float
    threshold: float
    direction: str  # "maximize" | "minimize"


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """計算二元分類指標。

    Args:
        y_true:  真實標籤（0/1）。
        y_pred:  預測標籤（0/1）。
        y_proba: 正類機率（可選，提供時計算 ROC-AUC）。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # ROC-AUC 需要兩個類別都出現，否則 sklearn 會報錯。
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba).ravel()))
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """計算迴歸指標（時序需求預測用）。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": float(np.sqrt(mse)), "mae": mae, "mse": mse}


def quality_gate(
    metrics: Mapping[str, float],
    train_cfg: Mapping[str, Any],
) -> GateResult:
    """對主指標套用門檻判定。

    從 ``train_cfg['evaluation']`` 讀 ``primary_metric`` 與 ``min_threshold``。
    對「越小越好」的指標（rmse / mae / mse）採 ``<= threshold`` 判定，
    其餘採 ``>= threshold``。指標缺失時直接判定不通過（fail-safe）。
    """
    eval_cfg = dict(train_cfg.get("evaluation", {}))
    metric = str(eval_cfg.get("primary_metric", "f1"))
    threshold = float(eval_cfg.get("min_threshold", 0.0))

    minimize = {"rmse", "mae", "mse"}
    direction = "minimize" if metric in minimize else "maximize"

    if metric not in metrics:
        logger.warning("品質門檻找不到主指標 '%s'，判定不通過。", metric)
        return GateResult(False, metric, float("nan"), threshold, direction)

    value = float(metrics[metric])
    passed = value <= threshold if direction == "minimize" else value >= threshold
    logger.info(
        "品質門檻：%s=%.4f vs 門檻 %.4f（%s）→ %s",
        metric,
        value,
        threshold,
        direction,
        "通過" if passed else "未通過",
    )
    return GateResult(passed, metric, value, threshold, direction)
