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
from collections.abc import Mapping
from typing import Any

from src.training._trainers import TrainArtifacts, train_one
from src.training.evaluate import quality_gate
from src.training.registry import register_model
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


def _register_and_alias(mlflow, model_info, art: TrainArtifacts, cfg: Mapping[str, Any]) -> None:
    """把模型註冊進 registry 並打 alias（champion）。

    實作已抽到 ``src.training.registry``，讓走 Prefect flow 的那條路徑
    （``pipelines/training_pipeline.py``）能共用同一份邏輯——
    否則會出現「用 make 訓練會註冊、用 flow 訓練不會」的不一致。
    """
    register_model(
        model_info,
        art.metrics,
        cfg,
        model_name=art.model_name,
    )


def run(cfg: Mapping[str, Any]) -> TrainArtifacts:
    """執行一次完整訓練並記錄到 MLflow，回傳訓練產物。"""
    set_seed(int(cfg.get("seed", 42)))
    model_name = str(cfg.get("active_model"))
    logger.info("啟動訓練：active_model=%s", model_name)

    mlflow = _setup_mlflow(cfg)
    train_cfg = cfg.get("train", {})
    # 模型型態可覆蓋評估指標/門檻：分類用 f1、迴歸（ts）用 rmse。
    # conf/model/<name>.yaml 的 evaluation 區段覆蓋 conf/train 的全域預設，
    # 避免對迴歸模型套用 f1 門檻而誤判「找不到主指標」。
    model_eval = cfg.get("model", {}).get("evaluation")
    if model_eval:
        train_cfg = {
            **train_cfg,
            "evaluation": {**train_cfg.get("evaluation", {}), **model_eval},
        }

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
    """CLI 進入點：支援兩種覆蓋語法後執行訓練。

    - ``--model xgboost``：旗標式，覆蓋 active_model。
    - ``model=xgboost data=sensors``：Hydra 風格 key=value（Makefile 採此式），
      支援切換 model / data / train 群組。
    """
    parser = argparse.ArgumentParser(description="Smart Factory 通用訓練入口")
    parser.add_argument(
        "--model",
        default=None,
        help="覆蓋 conf/config.yaml 的 active_model（xgboost / lstm / resnet）",
    )
    args, extras = parser.parse_known_args()

    # 解析 Hydra 風格的 `key=value` 覆蓋（如 `model=xgboost data=sensors`）。
    overrides: dict[str, Any] = {}
    for token in extras:
        if "=" in token and not token.startswith("-"):
            key, value = token.split("=", 1)
            overrides[key.strip()] = value.strip()
    if args.model:
        overrides["model"] = args.model

    cfg = load_config(overrides=overrides or None)
    run(cfg)


if __name__ == "__main__":
    main()
