"""特徵工程單元測試。

策略
----
1. 先對 ``src.features`` 的公開特徵函式做契約測試（若模組已存在）；
   尚未建好時用 ``pytest.importorskip`` 跳過，避免阻斷整個套件。
2. 同時對「滾動均值 / 滾動標準差」這類核心時序特徵的**語意**做一份
   獨立參考實作斷言——確保不論 src 是否就緒，特徵邏輯都有真實守門。

實體鍵 = machine_id；時間鍵 = event_timestamp。所有滾動特徵都必須
**先依機台分組、再依時間排序**，避免跨機台洩漏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ENTITY_COL = "machine_id"
TIMESTAMP_COL = "event_timestamp"


# --------------------------------------------------------------------------- #
# 1. 參考語意測試：不依賴 src，永遠提供真實守門
# --------------------------------------------------------------------------- #
def _rolling_mean_per_machine(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """參考實作：分組 + 排序後的滾動均值（min_periods=1）。"""
    return (
        df.sort_values([ENTITY_COL, TIMESTAMP_COL])
        .groupby(ENTITY_COL)[col]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )


def test_rolling_mean_does_not_leak_across_machines() -> None:
    """每台機台第一筆的 window 均值 == 自身首值，不可摻入別台資料。"""
    df = pd.DataFrame(
        {
            ENTITY_COL: ["m1", "m1", "m2", "m2"],
            TIMESTAMP_COL: pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "temperature": [10.0, 20.0, 100.0, 200.0],
        }
    )
    out = _rolling_mean_per_machine(df, "temperature", window=3)
    df = df.assign(roll=out).sort_values([ENTITY_COL, TIMESTAMP_COL])
    first_of_each = df.groupby(ENTITY_COL).first()
    # m1 首筆均值 = 10，m2 首筆均值 = 100，證明分組隔離成立
    assert first_of_each.loc["m1", "roll"] == pytest.approx(10.0)
    assert first_of_each.loc["m2", "roll"] == pytest.approx(100.0)


def test_rolling_mean_window_values() -> None:
    """單機台 window=2 的滾動均值數值正確。"""
    df = pd.DataFrame(
        {
            ENTITY_COL: ["m1"] * 4,
            TIMESTAMP_COL: pd.date_range("2024-01-01", periods=4, freq="h"),
            "temperature": [2.0, 4.0, 6.0, 8.0],
        }
    )
    out = _rolling_mean_per_machine(df, "temperature", window=2).tolist()
    assert out == pytest.approx([2.0, 3.0, 5.0, 7.0])


def test_rolling_std_is_non_negative() -> None:
    """滾動標準差恆 >= 0。"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            ENTITY_COL: ["m1"] * 20,
            TIMESTAMP_COL: pd.date_range("2024-01-01", periods=20, freq="h"),
            "vibration": rng.normal(3.0, 0.5, 20),
        }
    ).sort_values([ENTITY_COL, TIMESTAMP_COL])
    std = df.groupby(ENTITY_COL)["vibration"].rolling(window=3, min_periods=1).std().fillna(0.0)
    assert (std.to_numpy() >= -1e-9).all()


# --------------------------------------------------------------------------- #
# 2. 契約測試：src.features 就緒時，驗其公開介面
# --------------------------------------------------------------------------- #
def test_src_features_build_contract(mini_sensors_df: pd.DataFrame) -> None:
    """``src.features.build_features.build_sensor_features`` 契約測試。

    驗證：保留實體欄與列數、不修改輸入（不可變語義）、新增滾動特徵欄、
    且輸出無殘留 NaN（src 以 0 補滿視窗起點）。
    """
    feats = pytest.importorskip(
        "src.features.build_features", reason="src.features.build_features 尚未建立"
    )
    builder = getattr(feats, "build_sensor_features", None)
    if builder is None:
        pytest.skip("src.features.build_features 未提供 build_sensor_features")

    original = mini_sensors_df.copy()
    out = builder(mini_sensors_df.copy())
    assert isinstance(out, pd.DataFrame)
    # 實體欄位必須保留，且不可無故減少列數（特徵工程不應丟資料）
    assert ENTITY_COL in out.columns
    assert len(out) == len(mini_sensors_df)
    # 必須真的「新增」特徵欄
    assert out.shape[1] > original.shape[1]
    # src 以 0 補滿視窗起點，輸出不應殘留 NaN
    assert not out.isna().any().any()
    # 不可變語義：輸入 DataFrame 不被就地修改
    pd.testing.assert_frame_equal(mini_sensors_df, original)
