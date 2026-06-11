"""XGBoost 預測性維護模型。

場景：以設備感測器（temperature / vibration / current）預測「是否即將故障」
（failure 0/1 二元分類）。提供清楚的 fit / predict / predict_proba / save / load
介面，並與服務端（src.serving.predict）的本地產物契約對齊：
    模型權重以 XGBoost 原生格式存成 ``model.xgb``，
    讓 ``xgboost.Booster().load_model()`` 能直接讀回（registry 不可用時的回退）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.utils.logging import get_logger

logger = get_logger(__name__)

# 與 datasets/toy_sensors.csv 及 src.serving.predict.TABULAR_FEATURES 對齊。
DEFAULT_FEATURES: tuple[str, ...] = ("temperature", "vibration", "current")
WEIGHTS_FILENAME = "model.xgb"


class XGBoostMaintenanceModel:
    """XGBoost 二元分類器封裝（預測性維護）。

    Attributes:
        features: 特徵欄位順序（務必與訓練 / 推論端一致）。
        params:   傳入 :class:`xgboost.XGBClassifier` 的超參數。
    """

    def __init__(
        self,
        features: Sequence[str] = DEFAULT_FEATURES,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.features: list[str] = list(features)
        self.params: dict[str, Any] = dict(params or {})
        self._model: XGBClassifier = XGBClassifier(**self.params)
        self._fitted: bool = False

    # ── 訓練 / 推論 ───────────────────────────────────────────────────────
    def _as_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """把輸入轉成固定特徵順序的 float32 矩陣（不可變：操作副本）。"""
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.features if c not in X.columns]
            if missing:
                raise ValueError(f"輸入缺少特徵欄位：{missing}")
            return X.loc[:, self.features].to_numpy(dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **fit_kwargs: Any,
    ) -> "XGBoostMaintenanceModel":
        """訓練模型。回傳 self 以支援鏈式呼叫。"""
        matrix = self._as_matrix(X)
        target = np.asarray(y).ravel()
        logger.info("訓練 XGBoost：n_samples=%d, n_features=%d", *matrix.shape)
        self._model.fit(matrix, target, **fit_kwargs)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """回傳正類（故障）機率，一維陣列。"""
        self._ensure_fitted()
        proba = self._model.predict_proba(self._as_matrix(X))
        return np.asarray(proba)[:, 1]

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """依門檻回傳 0/1 預測。"""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── 序列化 ───────────────────────────────────────────────────────────
    def save(self, dir_path: str | Path) -> Path:
        """把模型存到 ``<dir_path>/model.xgb``（XGBoost 原生格式）。

        Returns:
            實際寫出的權重檔路徑。
        """
        self._ensure_fitted()
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = out_dir / WEIGHTS_FILENAME
        # 存底層 booster：與 src.serving 的 Booster().load_model() 回退契約一致。
        self._model.get_booster().save_model(str(weights))
        logger.info("已儲存 XGBoost 模型：%s", weights)
        return weights

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        features: Sequence[str] = DEFAULT_FEATURES,
    ) -> "XGBoostMaintenanceModel":
        """從 ``<dir_path>/model.xgb`` 載入模型。"""
        weights = Path(dir_path) / WEIGHTS_FILENAME
        if not weights.exists():
            raise FileNotFoundError(f"找不到模型權重：{weights}")
        instance = cls(features=features)
        instance._model.load_model(str(weights))
        instance._fitted = True
        logger.info("已載入 XGBoost 模型：%s", weights)
        return instance

    @property
    def sklearn_model(self) -> XGBClassifier:
        """暴露底層 estimator，供 MLflow log_model / signature 推導使用。"""
        return self._model

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("模型尚未訓練，請先呼叫 fit()。")
