"""
sandbox/tests/test_sample.py — 給 GitHub Actions 跑的極簡 pytest（階 10）

這個檔示範什麼：
    一個「自我包含」的最小測試集，讓 sandbox/github-actions/ci.yml 有東西可以跑。
    我們不 import 任何外部模組（保持沙盒孤立、可獨立執行），而是在檔內
    自帶兩支被測函式：
      - accuracy()  ：一個 metric 函式（ML 專案最常見的被測對象）
      - load_toy_data()：一個資料載入 + 基本資料驗證函式

    這兩支正對應 README 裡講的「ML 專屬測試」：
      1. metric 函式要算對（單元測試）
      2. 資料要符合預期 schema / 範圍（資料驗證測試）

怎麼跑：
    # 安裝 pytest（課程 pyproject.toml 已含）
    #   pip install pytest
    # 在 m5-automation/sandbox 目錄下執行：
    python -m pytest tests/ -v
    # 或在 tests/ 目錄下直接：
    python -m pytest -v
"""

import random

# 設定隨機種子：任何用到隨機性的測試都要固定 seed，避免 CI 上偶發失敗（flaky）。
SEED = 42


# ----------------------------------------------------------------------
# 被測函式（在真實專案裡，這些會 import 自 src/；沙盒為了孤立而內嵌）
# ----------------------------------------------------------------------
def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """計算準確率：預測對的比例。這是最典型的「metric 函式」。"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 與 y_pred 長度必須相同")
    if not y_true:
        raise ValueError("不能對空資料計算 accuracy")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def load_toy_data(n: int = 50) -> list[dict]:
    """產生一份玩具資料（特徵 x in [0,1]、標籤 y in {0,1}）。

    對應「資料載入」步驟；下面的測試會驗證它的 schema 與數值範圍。
    """
    random.seed(SEED)
    return [{"x": random.random(), "y": int(random.random() > 0.5)} for _ in range(n)]


# ----------------------------------------------------------------------
# 測試一：metric 函式正確性（單元測試）
# ----------------------------------------------------------------------
def test_accuracy_all_correct():
    """全部預測正確時，accuracy 應為 1.0。"""
    assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_accuracy_half_correct():
    """一半正確時，accuracy 應為 0.5。"""
    assert accuracy([0, 1, 0, 1], [0, 1, 1, 0]) == 0.5


def test_accuracy_rejects_empty():
    """對空資料計算 accuracy 應拋出明確錯誤（快速失敗）。"""
    import pytest

    with pytest.raises(ValueError):
        accuracy([], [])


# ----------------------------------------------------------------------
# 測試二：資料驗證（ML 專屬測試 —— 確保進 pipeline 的資料符合預期）
# ----------------------------------------------------------------------
def test_toy_data_schema():
    """每筆資料都要有 x 與 y 兩個欄位（schema 驗證）。"""
    data = load_toy_data(n=20)
    assert len(data) == 20
    for row in data:
        assert set(row.keys()) == {"x", "y"}


def test_toy_data_value_range():
    """特徵 x 必須落在 [0,1]、標籤 y 必須是 0 或 1（範圍/值域驗證）。"""
    data = load_toy_data(n=20)
    for row in data:
        assert 0.0 <= row["x"] <= 1.0
        assert row["y"] in (0, 1)
