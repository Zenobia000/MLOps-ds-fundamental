"""特徵管線（Prefect flow）。

階段：load → clean → validate → build_features →（選配）materialize 進 Feast。
對 ``src.*`` 全為軟相依，本地可直接：``python -m pipelines.feature_pipeline``。
"""

from __future__ import annotations

import logging

import pandas as pd

from pipelines import flow, task
from pipelines._helpers import (
    ENTITY_COL,
    TIMESTAMP_COL,
    load_config,
    load_sensors,
    soft_import,
)

logger = logging.getLogger("pipelines.feature")


@task
def extract(config: dict) -> pd.DataFrame:
    """載入原始感測資料。"""
    df = load_sensors(config)
    logger.info("載入 %d 筆感測資料、%d 台機台。", len(df), df[ENTITY_COL].nunique())
    return df


@task
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """清洗：去重 (machine_id, event_timestamp)、依時間排序。"""
    out = (
        df.drop_duplicates(subset=[ENTITY_COL, TIMESTAMP_COL])
        .sort_values([ENTITY_COL, TIMESTAMP_COL])
        .reset_index(drop=True)
    )
    return out


@task
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """資料契約檢查；通過才往下走，否則丟出明確錯誤。

    ``src.data.validation.validate_sensors`` 的契約是「通過回傳同一 df，
    失敗則 raise DataValidationError」，故此處直接交給它把關。
    """
    validator = soft_import("src.data.validation", "validate_sensors")
    if validator is not None:
        return validator(df)
    # 後援：最小 schema 檢查
    required = {ENTITY_COL, TIMESTAMP_COL, "temperature", "vibration", "current"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"資料缺少必要欄位: {missing}")
    return df


@task
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徵工程：一律委派給 src.features.build_features。

    ★ 這裡刻意「沒有後援」。

    先前這支在 src.features 載不到時，會在管線內**自己重算一份**滾動均值／標準差。
    那份副本跟 src/features/build_features.py 是兩份程式碼，一旦其中一份改了視窗大小、
    改了 fillna 策略、或多加一個特徵，管線算出來的特徵就與訓練期不同——
    而且**不會有任何錯誤訊息**，只會得到一個安靜地變差的模型。
    那正是本專案存在要防的 training/serving skew。

    所以現在的選擇是：載不到就大聲失敗。特徵定義只能有一個真相來源。
    """
    builder = soft_import("src.features.build_features", "build_sensor_features")
    if builder is None:
        raise RuntimeError(
            "找不到 src.features.build_features.build_sensor_features。"
            "特徵計算不提供後援實作——兩份特徵邏輯會造成訓練/服務不一致。"
            "請確認在專案根目錄執行，且已 make setup。"
        )
    return builder(df)


@task
def materialize(df: pd.DataFrame, config: dict) -> int:
    """（選配）寫入 processed 並嘗試 Feast materialize；回傳寫出列數。"""
    out_dir = config.get("paths", {}).get("data_processed", "data/processed")
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "features.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:  # noqa: BLE001 - 無 pyarrow 時退回 csv
        out_path = Path(out_dir) / "features.csv"
        df.to_csv(out_path, index=False)
    logger.info("特徵集已寫出至 %s（%d 列）。", out_path, len(df))
    return len(df)


@flow(name="feature-pipeline")
def feature_flow(config_path: str | None = None) -> pd.DataFrame:
    """特徵管線主流程，回傳建好的特徵 DataFrame。"""
    config = load_config(config_path)
    raw = extract(config)
    cleaned = clean(raw)
    validated = validate(cleaned)
    features = build_features(validated)
    materialize(features, config)
    return features


def main() -> None:
    """CLI 入口：``python -m pipelines.feature_pipeline``。"""
    logging.basicConfig(level=logging.INFO)
    feats = feature_flow()
    print(f"[feature-pipeline] 完成，特徵欄位 {feats.shape[1]} 個、列數 {len(feats)}。")


if __name__ == "__main__":
    main()
