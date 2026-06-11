"""各模型型態的訓練實作（被 src.training.train 編排呼叫）。

每個 trainer 回傳統一的 :class:`TrainArtifacts`，封裝：
    - params:    要 log 的超參數
    - metrics:   評估指標（quality gate 會用）
    - log_model: 把模型寫進當前 MLflow run 的 callable（封裝各 flavor 差異）

如此 train.py 不需知道模型是 XGBoost / LSTM / ResNet，達成「型態無關的編排」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.data.loaders import load_demand, load_sensors
from src.models.tabular import XGBoostMaintenanceModel
from src.models.timeseries import LSTMForecaster, make_windows
from src.training.evaluate import evaluate_classification, evaluate_regression
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainArtifacts:
    """訓練產物的統一契約。"""

    model_name: str
    params: Mapping[str, Any]
    metrics: Mapping[str, float]
    log_model: Callable[..., Any]  # (mlflow, *, artifact_path) -> model_info
    local_path: Path | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def _models_dir(cfg: Mapping[str, Any], sub: str) -> Path:
    """組出 ``<paths.models>/<sub>`` 的本地產物目錄。"""
    base = Path(cfg.get("paths", {}).get("models", "models"))
    return base / sub


def _artifact_logger(artifact: Path) -> Callable[..., Any]:
    """回傳「把本地產物（lstm 權重 / resnet ONNX）log 進當前 run」的 callable。

    回傳具 ``model_uri`` 屬性的輕量物件，與 train.py 的註冊流程介面一致。
    """

    def _log(mlflow, *, artifact_path: str):
        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id
        return type("Info", (), {"model_uri": f"runs:/{run_id}/{artifact_path}"})()

    return _log


# ── XGBoost：預測性維護 ─────────────────────────────────────────────────────
def _train_xgboost(cfg: Mapping[str, Any]) -> TrainArtifacts:
    from mlflow.models.signature import infer_signature

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    features = list(data_cfg.get("feature_columns", ["temperature", "vibration", "current"]))
    target = str(data_cfg.get("target_column", "failure"))

    df = load_sensors(cfg)
    X, y = df[features], df[target].to_numpy()
    n_test = max(1, int(len(df) * 0.2))
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]  # 時序切分：尾段當測試
    y_train, y_test = y[:-n_test], y[-n_test:]

    model = XGBoostMaintenanceModel(features, model_cfg.get("params")).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    metrics = evaluate_classification(y_test, (proba >= 0.5).astype(int), proba)
    local = model.save(_models_dir(cfg, "tabular"))
    signature = infer_signature(X_test, proba)

    def _log(mlflow, *, artifact_path: str):
        return mlflow.sklearn.log_model(
            sk_model=model.sklearn_model,
            name=artifact_path,
            signature=signature,
            input_example=X_test.iloc[:3],
        )

    return TrainArtifacts("xgboost", dict(model_cfg.get("params", {})), metrics, _log, local)


# ── LSTM：產能需求預測 ──────────────────────────────────────────────────────
def _train_lstm(cfg: Mapping[str, Any]) -> TrainArtifacts:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    window = data_cfg.get("window", {})
    lookback = int(window.get("lookback", 12))
    horizon = int(window.get("horizon", model_cfg.get("params", {}).get("output_size", 6)))

    demand = load_demand(cfg)
    target_col = data_cfg.get("target_column", "demand")
    series = demand[target_col].to_numpy(dtype=np.float32)

    # toy 後援：日聚合需求序列過短，無法切出視窗時，改用每小時感測序列
    # （以 current 為單變量需求 proxy，~750 點），確保 smoke test 永遠可跑。
    if len(series) < lookback + horizon:
        logger.warning(
            "需求序列長度 %d 不足以切視窗（需 ≥ %d），改用每小時感測序列做 smoke。",
            len(series),
            lookback + horizon,
        )
        series = load_sensors(cfg)["current"].to_numpy(dtype=np.float32)

    X, y = make_windows(series, lookback, horizon)
    n_test = max(1, int(len(X) * 0.2))
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

    params = {**model_cfg.get("params", {}), "horizon": horizon}
    model = LSTMForecaster(params).fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    local = model.save(_models_dir(cfg, "timeseries"))
    return TrainArtifacts("lstm", params, metrics, _artifact_logger(local), local)


# ── ResNet：瑕疵檢測（小樣本 smoke）─────────────────────────────────────────
def _train_resnet(cfg: Mapping[str, Any]) -> TrainArtifacts:
    """ResNet 微調 smoke：用隨機張量跑一個 forward/backward 並匯出 ONNX。

    TODO（生產化）：接 src.data.load_images 的真實 ImageFolder，
    加入完整訓練迴圈、augmentation、val 評估；此處僅驗證骨架與 ONNX 匯出可跑。
    """
    import torch

    from src.models.vision import ResNetDefectClassifier

    model_cfg = cfg.get("model", {})
    params = dict(model_cfg.get("params", {}))
    # smoke：未必有 ImageNet 權重 / 網路，預設不下載以保證可跑。
    params.setdefault("pretrained", False)
    model = ResNetDefectClassifier(params)

    img = int(params.get("image_size", 224))
    dummy_x = torch.randn(4, 3, img, img)
    dummy_y = torch.randint(0, int(params.get("num_classes", 2)), (4,))
    optim = torch.optim.Adam(model.trainable_parameters(), lr=float(params.get("lr", 1e-3)))
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    optim.zero_grad()
    loss = loss_fn(model(dummy_x.to(model.device)), dummy_y.to(model.device))
    loss.backward()
    optim.step()

    out_dir = _models_dir(cfg, "vision")
    model.save(out_dir)
    onnx_path = model.export_onnx(out_dir)
    metrics = {"train_loss": float(loss.item()), "f1": 0.0}  # f1 待真實 val 計算
    return TrainArtifacts("resnet", params, metrics, _artifact_logger(onnx_path), onnx_path)


_DISPATCH: dict[str, Callable[[Mapping[str, Any]], TrainArtifacts]] = {
    "xgboost": _train_xgboost,
    "lstm": _train_lstm,
    "resnet": _train_resnet,
}


def train_one(model_name: str, cfg: Mapping[str, Any]) -> TrainArtifacts:
    """依模型名稱派發到對應 trainer。"""
    if model_name not in _DISPATCH:
        raise ValueError(
            f"未知的 active_model：{model_name}（可選 {list(_DISPATCH)}）"
        )
    return _DISPATCH[model_name](cfg)
