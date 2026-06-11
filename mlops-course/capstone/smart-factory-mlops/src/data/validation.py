"""資料驗證（validation）。

以「Great Expectations 風格」自寫的輕量契約檢查：schema（欄位齊全、
型別）、值域（物理合理範圍）、缺值、標籤域。任一違反即丟出
:class:`DataValidationError` 並附清晰訊息，符合「快速失敗」原則。

刻意零重量級依賴（不引 great_expectations），讓 CI smoke test 極快。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# 感測器數值欄的物理合理範圍（min, max）；可被 config 覆蓋。
_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-50.0, 250.0),
    "vibration": (0.0, 100.0),
    "current": (0.0, 200.0),
}

ENTITY_COLUMN = "machine_id"
TIMESTAMP_COLUMN = "event_timestamp"


class DataValidationError(ValueError):
    """資料未通過契約檢查時拋出；訊息描述哪條期望被違反。"""


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """斷言所有必要欄位存在，否則快速失敗。"""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"缺少必要欄位：{missing}；實際欄位：{list(df.columns)}"
        )


def _check_ranges(df: pd.DataFrame, ranges: Mapping[str, tuple[float, float]]) -> None:
    """檢查數值欄是否落在合理範圍內。"""
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = series[(series < lo) | (series > hi)]
        if not out_of_range.empty:
            raise DataValidationError(
                f"欄位 '{col}' 有 {len(out_of_range)} 筆超出合理範圍 "
                f"[{lo}, {hi}]，例如：{out_of_range.iloc[0]}"
            )


def validate_sensors(
    df: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    *,
    allow_nulls: bool = False,
) -> pd.DataFrame:
    """驗證感測器資料契約並回傳原 DataFrame（驗證通過時）。

    檢查項目：
      1. schema：entity / timestamp / 特徵 / 標籤欄位齊全。
      2. 空集合：至少一列。
      3. 缺值：預設不允許關鍵欄位有 NaN。
      4. 值域：感測器數值落在物理合理範圍。
      5. 標籤域：``failure`` 僅含 {0, 1}。

    Args:
        df: 待驗證資料（通常來自 :func:`src.data.loaders.load_sensors`）。
        config: ``data`` 子設定，用於取得欄位名與自訂範圍。
        allow_nulls: 是否容忍關鍵欄位缺值（預設 False，快速失敗）。

    Returns:
        通過驗證的同一 DataFrame（不修改，維持不可變語義）。

    Raises:
        DataValidationError: 任一契約被違反。
    """
    cfg = config or {}
    data_cfg = cfg.get("data", cfg) if isinstance(cfg, Mapping) else {}

    entity = data_cfg.get("entity_column", ENTITY_COLUMN)
    ts = data_cfg.get("timestamp_column", TIMESTAMP_COLUMN)
    features = list(data_cfg.get("feature_columns", list(_DEFAULT_RANGES)))
    target = data_cfg.get("target_column", "failure")

    if df.empty:
        raise DataValidationError("感測器資料為空（0 列），無法用於訓練。")

    _require_columns(df, [entity, ts, *features])

    key_cols = [entity, ts, *features]
    if not allow_nulls:
        null_counts = df[key_cols].isna().sum()
        bad = null_counts[null_counts > 0]
        if not bad.empty:
            raise DataValidationError(f"關鍵欄位含缺值：{bad.to_dict()}")

    ranges = {**_DEFAULT_RANGES, **dict(data_cfg.get("expectations", {}))}
    _check_ranges(df, {k: v for k, v in ranges.items() if k in features})

    if target in df.columns:
        invalid = set(df[target].dropna().unique()) - {0, 1}
        if invalid:
            raise DataValidationError(
                f"標籤欄 '{target}' 含非 0/1 值：{sorted(invalid)}"
            )

    logger.info("感測器資料驗證通過（%d 列）。", len(df))
    return df


def validate_image_manifest(
    manifest: pd.DataFrame,
    *,
    allow_empty: bool = True,
) -> pd.DataFrame:
    """驗證影像 manifest 的 schema 與檔案路徑欄位。

    Args:
        manifest: 來自 :func:`src.data.loaders.load_images` 的 DataFrame。
        allow_empty: 是否容許空 manifest（真實影像尚未取得時為 True）。

    Returns:
        通過驗證的同一 manifest。

    Raises:
        DataValidationError: schema 不符或不允許空集合時。
    """
    _require_columns(manifest, ["image_path", "label", "split"])
    if manifest.empty and not allow_empty:
        raise DataValidationError("影像 manifest 為空，但呼叫端要求非空。")
    logger.info("影像 manifest 驗證通過（%d 筆）。", len(manifest))
    return manifest
