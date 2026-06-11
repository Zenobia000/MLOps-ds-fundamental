"""通用訓練入口（config-driven dispatcher）。

流程：
    讀 config → set_seed → 依 ``active_model`` 載資料、建模、訓練、評估 →
    MLflow 記錄 params / metrics / signature / model → 套品質門檻 →
    通過則註冊 registry 並打 alias（champion）。

執行（從 repo 根）::

    python -m src.training.train                 # 用 conf/config.yaml 的 active_model
    python -m src.training.train --model lstm    # 臨時覆蓋 active_model

模型專屬訓練細節委派給 :mod:`src.training._trainers`，本檔只負責編排與 MLflow。
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from src.training._trainers import TrainArtifacts, train_one
from src.training.evaluate import quality_gate
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _setup_mlflow(cfg: Mapping[str, Any]):
    """設定 MLflow tracking URI 與 experiment；回傳 mlflow 模組（延遲匯入）。"""
    import mlflow

    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg.get("experiment", "smart-factory"))
    return mlflow


def _register_and_alias(
    mlflow, model_info, art: TrainArtifacts, cfg: Mapping[str, Any]
) -> None:
    """把模型註冊進 registry 並打 alias（champion）。失敗不致命，只記錄警告。"""
    registry_name = f"{cfg.get('project', 'smart-factory')}-{art.model_name}"
    try:
        result = mlflow.register_model(model_info.model_uri, registry_name)
        client = mlflow.tracking.MlflowClient()
        # 新版 MLflow 用 alias 取代 stage；champion 指向當前最佳版本。
        client.set_registered_model_alias(registry_name, "champion", result.version)
        logger.info("已註冊模型 %s v%s 並打 alias=champion", registry_name, result.version)
    except Exception as exc:  # noqa: BLE001 — 離線 / 教學環境註冊可失敗
        logger.warning("模型註冊失敗（%s），略過（不影響本地產物）。", exc)


def run(cfg: Mapping[str, Any]) -> TrainArtifacts:
    """執行一次完整訓練並記錄到 MLflow，回傳訓練產物。"""
    set_seed(int(cfg.get("seed", 42)))
    model_name = str(cfg.get("active_model"))
    logger.info("啟動訓練：active_model=%s", model_name)

    mlflow = _setup_mlflow(cfg)
    train_cfg = cfg.get("train", {})

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        # 委派模型專屬訓練；回傳指標、signature、可 log 的模型物件。
        art = train_one(model_name, cfg)

        mlflow.log_params({"active_model": model_name, **art.params})
        mlflow.log_metrics(art.metrics)

        gate = quality_gate(art.metrics, train_cfg)
        mlflow.log_metric("gate_passed", float(gate.passed))
        mlflow.set_tag("gate_metric", gate.metric)

        # 記錄模型（含 signature）；flavor 由各 trainer 決定。
        model_info = art.log_model(mlflow, artifact_path="model")
        logger.info("run_id=%s metrics=%s", run.info.run_id, art.metrics)

        # 僅在通過門檻且設定要求時才註冊（CT 守門）。
        if gate.passed and bool(train_cfg.get("logging", {}).get("register_model")):
            _register_and_alias(mlflow, model_info, art, cfg)
        elif not gate.passed:
            logger.warning("未通過品質門檻，跳過 registry 註冊。")

    return art


def main() -> None:
    """CLI 進入點：解析 --model 覆蓋後執行訓練。"""
    parser = argparse.ArgumentParser(description="Smart Factory 通用訓練入口")
    parser.add_argument(
        "--model",
        default=None,
        help="覆蓋 conf/config.yaml 的 active_model（xgboost / lstm / resnet）",
    )
    args = parser.parse_args()

    overrides = {"active_model": args.model} if args.model else None
    cfg = load_config(overrides=overrides)
    run(cfg)


if __name__ == "__main__":
    main()
