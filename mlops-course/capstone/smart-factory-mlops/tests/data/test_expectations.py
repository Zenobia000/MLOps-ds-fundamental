"""資料 schema / 分布契約測試。

把「資料應該長什麼樣」寫成可執行的期望，CI 在訓練前先驗資料，
避免髒資料悄悄汙染模型。這裡用純 pandas 表達期望（等價於 Great
Expectations 的 expect_* 系列），不額外引入重依賴。

期望來源：datasets/README 對 toy_sensors 的定義。
"""
from __future__ import annotations

import pandas as pd
import pytest

ENTITY_COL = "machine_id"
TIMESTAMP_COL = "event_timestamp"
SENSOR_COLS = ["temperature", "vibration", "current"]
LABEL_COL = "failure"

# 合理工程範圍（寬鬆邊界，抓離譜值而非正常波動）
RANGES = {
    "temperature": (0.0, 200.0),   # °C
    "vibration": (0.0, 50.0),      # mm/s
    "current": (0.0, 100.0),       # A
}


def test_required_columns_present(sensors_df: pd.DataFrame) -> None:
    """必要欄位齊全（schema 契約）。"""
    required = {ENTITY_COL, TIMESTAMP_COL, LABEL_COL, *SENSOR_COLS}
    missing = required - set(sensors_df.columns)
    assert not missing, f"缺少必要欄位: {missing}"


def test_entity_and_timestamp_naming(sensors_df: pd.DataFrame) -> None:
    """實體鍵與時間鍵命名符合全專案契約。"""
    assert ENTITY_COL in sensors_df.columns
    assert TIMESTAMP_COL in sensors_df.columns


def test_no_nulls_in_core_columns(sensors_df: pd.DataFrame) -> None:
    """核心欄位不得有缺值。"""
    core = [ENTITY_COL, TIMESTAMP_COL, LABEL_COL, *SENSOR_COLS]
    nulls = sensors_df[core].isna().sum()
    assert nulls.sum() == 0, f"核心欄位存在缺值:\n{nulls[nulls > 0]}"


def test_label_is_binary(sensors_df: pd.DataFrame) -> None:
    """故障標籤僅含 0 / 1。"""
    uniq = set(pd.unique(sensors_df[LABEL_COL]))
    assert uniq.issubset({0, 1}), f"failure 含非法值: {uniq}"


def test_label_not_degenerate(sensors_df: pd.DataFrame) -> None:
    """正負類都存在，否則無法訓練分類器（分布契約）。"""
    rate = float(sensors_df[LABEL_COL].mean())
    assert 0.0 < rate < 1.0, f"故障率退化: {rate}"


@pytest.mark.parametrize("col", SENSOR_COLS)
def test_sensor_values_in_range(sensors_df: pd.DataFrame, col: str) -> None:
    """感測值落在合理工程範圍內（分布契約）。"""
    lo, hi = RANGES[col]
    s = sensors_df[col]
    assert s.min() >= lo, f"{col} 最小值 {s.min()} < {lo}"
    assert s.max() <= hi, f"{col} 最大值 {s.max()} > {hi}"


def test_timestamp_parseable_and_monotonic_per_machine(
    sensors_df: pd.DataFrame,
) -> None:
    """時間戳可解析，且每台機台內依時間嚴格遞增（時序契約）。"""
    df = sensors_df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    for mid, grp in df.groupby(ENTITY_COL):
        ts = grp.sort_index()[TIMESTAMP_COL]
        assert ts.is_monotonic_increasing or grp[TIMESTAMP_COL].is_unique, (
            f"機台 {mid} 時間戳非遞增或有重複"
        )


def test_no_duplicate_entity_timestamp(sensors_df: pd.DataFrame) -> None:
    """(machine_id, event_timestamp) 唯一，避免 Feast point-in-time 取錯。"""
    dup = sensors_df.duplicated(subset=[ENTITY_COL, TIMESTAMP_COL]).sum()
    assert dup == 0, f"存在 {dup} 筆重複的 (實體, 時間) 組合"


def test_src_validation_passes_clean_data(sensors_csv: str) -> None:
    """``src.data.validation.validate_sensors`` 對乾淨資料須通過（回傳同列數）。"""
    mod = pytest.importorskip(
        "src.data.validation", reason="src.data.validation 尚未建立"
    )
    fn = getattr(mod, "validate_sensors", None)
    if fn is None:
        pytest.skip("src.data.validation 未提供 validate_sensors")

    df = pd.read_csv(sensors_csv)
    # 契約：通過則回傳同一 DataFrame，失敗則 raise。
    out = fn(df)
    assert len(out) == len(df)


def test_src_validation_rejects_bad_label(sensors_csv: str) -> None:
    """注入非法標籤後，validate_sensors 須拋出錯誤（快速失敗）。"""
    mod = pytest.importorskip(
        "src.data.validation", reason="src.data.validation 尚未建立"
    )
    fn = getattr(mod, "validate_sensors", None)
    err = getattr(mod, "DataValidationError", Exception)
    if fn is None:
        pytest.skip("src.data.validation 未提供 validate_sensors")

    df = pd.read_csv(sensors_csv)
    df.loc[0, LABEL_COL] = 7  # 非 0/1 的非法標籤
    with pytest.raises(err):
        fn(df)
