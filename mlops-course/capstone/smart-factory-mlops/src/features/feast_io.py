"""Feast 取數薄封裝（feast_io）。

把 Feast 的 ``get_historical_features`` / ``get_online_features`` 包成
專案內統一介面，集中 feature_repo 路徑與 FeatureService 名稱，
讓訓練 / 服務層不必直接依賴 Feast SDK 的細節。

Feast 為選用依賴：未安裝時拋出清晰的 :class:`ImportError`，
而非在 import 階段就讓整包壞掉（保持骨架可 import）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # 僅型別檢查時引入，執行期不強制安裝 Feast。
    from feast import FeatureStore

logger = logging.getLogger(__name__)

# feature_repo 位於 repo 根；本檔上溯三層（src/features/feast_io.py）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO_PATH = _REPO_ROOT / "feature_repo"

# 與 feature_repo/features.py 中定義的 FeatureService 名稱保持一致。
DEFAULT_FEATURE_SERVICE = "predictive_maintenance_v1"


def _get_store(repo_path: str | Path | None = None) -> FeatureStore:
    """建立並回傳 Feast :class:`FeatureStore`；缺套件時清晰報錯。"""
    try:
        from feast import FeatureStore
    except ImportError as exc:  # pragma: no cover - 依賴缺失路徑
        raise ImportError("需要 feast 才能取數；請先 `pip install feast`。") from exc

    path = Path(repo_path) if repo_path else _DEFAULT_REPO_PATH
    logger.info("初始化 Feast FeatureStore：%s", path)
    return FeatureStore(repo_path=str(path))


def get_historical_features(
    entity_df: pd.DataFrame,
    features: Sequence[str] | None = None,
    *,
    feature_service: str | None = DEFAULT_FEATURE_SERVICE,
    repo_path: str | Path | None = None,
) -> pd.DataFrame:
    """取 point-in-time 正確的歷史特徵（離線訓練用）。

    ``entity_df`` 須含 ``machine_id`` 與 ``event_timestamp``，Feast 會據此
    做 as-of join，保證每列只取到該時間點「之前」的特徵值。

    Args:
        entity_df: 實體 + 時間的 spine（訓練樣本的時間戳）。
        features: 明確指定的特徵清單；省略時用 ``feature_service``。
        feature_service: FeatureService 名稱（與 features.py 一致）。
        repo_path: feature_repo 路徑，省略時用專案預設。

    Returns:
        合併特徵後的訓練用 DataFrame。
    """
    store = _get_store(repo_path)
    if features is not None:
        ref = list(features)
    else:
        ref = store.get_feature_service(feature_service)
    job = store.get_historical_features(entity_df=entity_df, features=ref)
    return job.to_df()


def get_online_features(
    entity_rows: Sequence[dict],
    features: Sequence[str] | None = None,
    *,
    feature_service: str | None = DEFAULT_FEATURE_SERVICE,
    repo_path: str | Path | None = None,
) -> pd.DataFrame:
    """取最新線上特徵（低延遲推論用）。

    Args:
        entity_rows: 實體鍵列表，如 ``[{"machine_id": "machine_01"}]``。
        features: 明確特徵清單；省略時用 ``feature_service``。
        feature_service: FeatureService 名稱。
        repo_path: feature_repo 路徑。

    Returns:
        每個實體一列的線上特徵 DataFrame。
    """
    store = _get_store(repo_path)
    if features is not None:
        ref = list(features)
    else:
        ref = store.get_feature_service(feature_service)
    response = store.get_online_features(features=ref, entity_rows=list(entity_rows))
    return pd.DataFrame(response.to_dict())
