"""感測器時序特徵工程（build_features）。

由每台機台（``machine_id``）各自的時間序列，造出滾動統計特徵：
溫度滾動均值、振動滾動標準差、電流滾動均值、以及相鄰時點的變化量。

**避免時間穿越（point-in-time correctness）的關鍵**：
所有滾動視窗都先 ``shift(1)`` 再聚合，確保 *t* 時刻的特徵只用到
*t-1* 及更早的觀測；標籤 ``failure`` 屬於 *t* 時刻本身，不被特徵看見。
這條規則是 notebook ``03_pointintime_leakage_demo`` 的教學主軸。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ENTITY_COLUMN = "machine_id"
TIMESTAMP_COLUMN = "event_timestamp"

# 造出的特徵欄位（供 Feast FeatureView 與訓練層共用、保持單一定義）。
FEATURE_VIEW_COLUMNS: list[str] = [
    "temp_roll_mean",
    "vib_roll_std",
    "current_roll_mean",
    "temp_delta",
]


def build_sensor_features(
    df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    *,
    window: int = 3,
) -> pd.DataFrame:
    """為感測器資料造「無時間穿越」的滾動時序特徵。

    Args:
        df: 已驗證的感測器資料，須含 ``machine_id`` / ``event_timestamp`` /
            ``temperature`` / ``vibration`` / ``current``。
        config: ``data`` 子設定（可覆蓋欄位名與滾動視窗）。
        window: 滾動視窗大小（時點數）；亦可由 config ``features.window`` 指定。

    Returns:
        在原欄位上新增 :data:`FEATURE_VIEW_COLUMNS` 的新 DataFrame
        （回傳副本，不修改輸入，維持不可變語義）。

    Raises:
        KeyError: 缺少造特徵所需的來源欄位。
    """
    cfg = config or {}
    data_cfg = cfg.get("data", cfg) if isinstance(cfg, Mapping) else {}
    feat_cfg = cfg.get("features", {}) if isinstance(cfg, Mapping) else {}
    window = int(feat_cfg.get("window", window))

    entity = data_cfg.get("entity_column", ENTITY_COLUMN)
    ts = data_cfg.get("timestamp_column", TIMESTAMP_COLUMN)

    required = {entity, ts, "temperature", "vibration", "current"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"造特徵缺少來源欄位：{sorted(missing)}")

    out = df.sort_values([entity, ts]).copy()

    def _per_machine(g: pd.DataFrame) -> pd.DataFrame:
        """對「單一機台」的子序列造特徵，確保視窗不跨實體邊界。

        關鍵：``shift(1)`` 與 ``rolling`` 都在同一台機台的序列內完成，
        既避免時間穿越（看未來），也避免跨機台洩漏（machine_02 的視窗
        吃到 machine_01 的尾端資料）。
        """
        g = g.copy()
        temp_prev = g["temperature"].shift(1)
        vib_prev = g["vibration"].shift(1)
        cur_prev = g["current"].shift(1)
        roll = dict(window=window, min_periods=1)
        g["temp_roll_mean"] = temp_prev.rolling(**roll).mean()
        g["vib_roll_std"] = vib_prev.rolling(**roll).std()
        g["current_roll_mean"] = cur_prev.rolling(**roll).mean()
        # 與前一時點的溫度變化量（同樣只用過去資訊）。
        g["temp_delta"] = g["temperature"].diff()
        return g

    out = out.groupby(entity, group_keys=False)[list(out.columns)].apply(
        _per_machine
    )

    # 視窗起點與單點 std 會產生 NaN，補 0 視為「尚無歷史」的中性值。
    out[FEATURE_VIEW_COLUMNS] = out[FEATURE_VIEW_COLUMNS].fillna(0.0)

    logger.info(
        "造出 %d 個時序特徵（window=%d，%d 列）。",
        len(FEATURE_VIEW_COLUMNS),
        window,
        len(out),
    )
    return out
