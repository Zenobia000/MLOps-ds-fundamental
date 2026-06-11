"""資料載入器（loaders）。

三型態各一個進入點，皆 config-driven 並具「玩具資料後援」：

- :func:`load_sensors`：結構化 + 時序感測器（預測性維護）。
- :func:`load_images`：產線視覺瑕疵影像的 manifest（路徑 + 標籤）。
- :func:`load_demand`：產能需求時序（forecasting）。

共同契約（跨 agent）：實體鍵 ``machine_id``、時間鍵 ``event_timestamp``。
真實資料不存在時，後援到課程共用玩具資料（``datasets/toy_sensors.csv``），
讓 smoke test / CI 永遠跑得起來。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# repo 根 = 本檔上溯三層（src/data/loaders.py → repo 根）。
_REPO_ROOT = Path(__file__).resolve().parents[2]

# 課程共用玩具資料（Layer 1 沙盒），作為缺真實資料時的最終後援。
_TOY_SENSORS = _REPO_ROOT.parents[1] / "datasets" / "toy_sensors.csv"

ENTITY_COLUMN = "machine_id"
TIMESTAMP_COLUMN = "event_timestamp"


def _resolve_path(raw_path: str | Path) -> Path:
    """把設定中的相對路徑解析為絕對路徑（相對 repo 根）。"""
    path = Path(raw_path)
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _data_cfg(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """從 root config 取出 ``data`` 子設定；容忍直接傳入子設定的情況。"""
    if config is None:
        return {}
    if "data" in config and isinstance(config["data"], Mapping):
        return config["data"]
    return config


def load_sensors(
    config: Optional[Mapping[str, Any]] = None,
    *,
    source: Optional[str | Path] = None,
) -> pd.DataFrame:
    """載入感測器時序資料，並標準化時間欄為 ``datetime``。

    解析順序：明確 ``source`` 參數 → config 的 ``data.source`` → 玩具後援。

    Args:
        config: root config 或 ``data`` 子設定（含 ``source`` 鍵）。
        source: 直接指定的資料路徑，優先級最高。

    Returns:
        依 (machine_id, event_timestamp) 排序的 :class:`pandas.DataFrame`。

    Raises:
        FileNotFoundError: 指定來源與玩具後援皆不存在。
    """
    cfg = _data_cfg(config)
    candidate = source or cfg.get("source")
    path = _resolve_path(candidate) if candidate else None

    if path is None or not path.exists():
        if path is not None:
            logger.warning("感測器來源不存在：%s，後援到玩具資料。", path)
        if not _TOY_SENSORS.exists():
            raise FileNotFoundError(
                f"找不到感測器資料：{path} 與玩具後援 {_TOY_SENSORS} 皆不存在。"
            )
        path = _TOY_SENSORS

    logger.info("載入感測器資料：%s", path)
    df = pd.read_csv(path)
    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
        sort_cols = [c for c in (ENTITY_COLUMN, TIMESTAMP_COLUMN) if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def load_images(
    config: Optional[Mapping[str, Any]] = None,
    *,
    image_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """掃描影像目錄，建立 ``(image_path, label, split)`` manifest。

    採 MVTec AD 風格的目錄佈局：``<root>/<split>/<label>/*.png``，
    例如 ``data/external/mvtec/<class>/test/good/000.png``。
    目錄不存在時回傳空 manifest（欄位齊全），讓下游可優雅跳過。

    Args:
        config: root config 或 ``data`` 子設定（含 ``image_root`` 鍵）。
        image_root: 直接指定的影像根目錄，優先級最高。

    Returns:
        欄位為 ``image_path / label / split`` 的 :class:`pandas.DataFrame`。
    """
    cfg = _data_cfg(config)
    candidate = image_root or cfg.get("image_root") or "data/external/mvtec"
    root = _resolve_path(candidate)

    columns = ["image_path", "label", "split"]
    if not root.exists():
        logger.warning("影像根目錄不存在：%s，回傳空 manifest（待取得資料）。", root)
        return pd.DataFrame(columns=columns)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    records: list[dict[str, str]] = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        rel = img.relative_to(root)
        # 以相對路徑層級推斷 split / label：<split>/<label>/<file>。
        parts = rel.parts
        split = parts[0] if len(parts) >= 3 else "unknown"
        label = parts[-2] if len(parts) >= 2 else "unknown"
        records.append({"image_path": str(img), "label": label, "split": split})

    logger.info("影像 manifest 共 %d 筆（root=%s）。", len(records), root)
    return pd.DataFrame(records, columns=columns)


def load_demand(
    config: Optional[Mapping[str, Any]] = None,
    *,
    source: Optional[str | Path] = None,
) -> pd.DataFrame:
    """載入產能需求時序（forecasting 用）。

    優先讀取設定 / 參數指定的需求檔；缺檔時由感測器資料聚合出
    每日筆數作為「需求 proxy」，確保時序模型 smoke test 可跑。

    Returns:
        欄位為 ``event_timestamp / demand``（單一序列）的 DataFrame。
    """
    cfg = _data_cfg(config)
    candidate = source or cfg.get("demand_source")
    path = _resolve_path(candidate) if candidate else None

    if path is not None and path.exists():
        logger.info("載入需求資料：%s", path)
        df = pd.read_csv(path)
        if TIMESTAMP_COLUMN in df.columns:
            df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
        return df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    logger.warning("無需求資料，改由感測器聚合出 demand proxy（toy 模式）。")
    sensors = load_sensors(config)
    proxy = (
        sensors.set_index(TIMESTAMP_COLUMN)
        .resample("1D")
        .size()
        .rename("demand")
        .reset_index()
    )
    return proxy
