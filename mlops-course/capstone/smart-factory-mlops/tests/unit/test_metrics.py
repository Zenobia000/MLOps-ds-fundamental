"""評估指標單元測試。

預測性維護是不平衡二分類（故障為少數類），因此主要守門指標為
PR-AUC / F1 / recall，而非單看 accuracy。本檔對指標的**數學性質**做
真實斷言，並在 ``src`` 指標模組就緒時驗其契約。
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# 1. 指標數學性質（不依賴 src）
# --------------------------------------------------------------------------- #
def test_perfect_classifier_scores_are_one() -> None:
    """完美預測時 ROC-AUC / PR-AUC / F1 皆為 1。"""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.01, 0.1, 0.99, 0.95, 0.05, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)
    assert roc_auc_score(y_true, y_prob) == pytest.approx(1.0)
    assert average_precision_score(y_true, y_prob) == pytest.approx(1.0)
    assert f1_score(y_true, y_pred) == pytest.approx(1.0)


def test_random_classifier_rocauc_near_half() -> None:
    """大量樣本下隨機分數的 ROC-AUC 應接近 0.5。"""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=5000)
    y_prob = rng.random(size=5000)
    assert roc_auc_score(y_true, y_prob) == pytest.approx(0.5, abs=0.05)


def test_pr_auc_rewards_minority_ranking() -> None:
    """把少數類（故障）排前面的模型，PR-AUC 應高於亂排。"""
    y_true = np.array([0] * 90 + [1] * 10)
    good = np.r_[np.linspace(0.0, 0.4, 90), np.linspace(0.6, 1.0, 10)]
    rng = np.random.default_rng(1)
    bad = rng.random(100)
    assert average_precision_score(y_true, good) > average_precision_score(y_true, bad)


def test_f1_is_harmonic_mean_of_precision_recall() -> None:
    """F1 = 2PR/(P+R)，以已知 precision/recall 反推驗證。"""
    # 4 TP, 1 FP, 1 FN -> precision=4/5, recall=4/5, f1=0.8
    y_true = np.array([1, 1, 1, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 1])
    p, r = 4 / 5, 4 / 5
    expected = 2 * p * r / (p + r)
    assert f1_score(y_true, y_pred) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 2. 契約測試：src.training.evaluate 就緒時驗其輸出
# --------------------------------------------------------------------------- #
def test_src_evaluate_classification_contract() -> None:
    """``evaluate_classification(y_true, y_pred, y_proba)`` 須回傳合法區間指標。"""
    mod = pytest.importorskip("src.training.evaluate", reason="src.training.evaluate 尚未建立")
    fn = getattr(mod, "evaluate_classification", None)
    if fn is None:
        pytest.skip("src.training.evaluate 未提供 evaluate_classification")

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    y_pred = (y_prob >= 0.5).astype(int)
    result = fn(y_true, y_pred, y_prob)
    assert isinstance(result, dict)
    # 至少含常見鍵
    assert {"accuracy", "precision", "recall", "f1"} <= set(result)
    for key, value in result.items():
        # 所有常見分類指標都落在 [0, 1]
        assert 0.0 - 1e-9 <= float(value) <= 1.0 + 1e-9, f"{key}={value} 超界"


def test_src_quality_gate_blocks_low_metric() -> None:
    """``quality_gate`` 對低於門檻的指標須判定不通過（CT 守門）。"""
    mod = pytest.importorskip("src.training.evaluate", reason="src.training.evaluate 尚未建立")
    gate = getattr(mod, "quality_gate", None)
    if gate is None:
        pytest.skip("src.training.evaluate 未提供 quality_gate")

    train_cfg = {"evaluation": {"primary_metric": "f1", "min_threshold": 0.9}}
    low = gate({"f1": 0.2}, train_cfg)
    high = gate({"f1": 0.95}, train_cfg)
    assert bool(getattr(low, "passed", low)) is False
    assert bool(getattr(high, "passed", high)) is True
