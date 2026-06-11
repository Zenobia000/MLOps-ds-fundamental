"""DVC pipeline stage 1：資料清洗與切分（prepare）。

讀 ``params.yaml`` 的 ``prepare`` 區段，載入感測器資料，依時間做 train/val/test
時序切分，寫出 ``data/interim/{train,val,test}.parquet``。供 ``dvc.yaml`` 的
``prepare`` 階段呼叫，與 MLflow 編排的 ``src.training.train`` 互補（這條是
「可重現 artifact DAG」的示範路線）。

執行（從 repo 根）::

    python -m src.data.prepare
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.data.loaders import TIMESTAMP_COLUMN, load_sensors
from src.utils.logging import get_logger

logger = get_logger(__name__)
_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    """讀 params.yaml 的 prepare 區段（管線層旋鈕）。"""
    with (_ROOT / "params.yaml").open(encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("prepare", {})


def main() -> None:
    params = _load_params()
    df = load_sensors(source=params.get("source"))

    # 時序切分：依時間排序後，前段 train、中段 val、尾段 test，避免未來資訊洩漏。
    if TIMESTAMP_COLUMN in df.columns:
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    n = len(df)
    n_test = max(1, int(n * float(params.get("test_size", 0.2))))
    n_val = max(1, int(n * float(params.get("val_size", 0.1))))
    n_train = max(1, n - n_test - n_val)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    out_dir = _ROOT / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)

    logger.info(
        "prepare 完成：train=%d val=%d test=%d → %s",
        len(train),
        len(val),
        len(test),
        out_dir,
    )


if __name__ == "__main__":
    main()
