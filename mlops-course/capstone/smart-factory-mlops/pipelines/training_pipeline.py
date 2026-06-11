"""訓練管線（Prefect flow）。

階段：build features → split → train → evaluate → quality gate →（選配）register。
品質門檻是 CT（Continuous Training）的核心：指標未達標就**不註冊**，
避免劣化模型流入下游。對 ``src.training`` 軟相依，本地可直接跑。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipelines import flow, task
from pipelines._helpers import load_config, soft_import
from pipelines.feature_pipeline import feature_flow

logger = logging.getLogger("pipelines.training")

LABEL_COL = "failure"
# 品質門檻：低於此 ROC-AUC 不予註冊（教學用保守值，可由 conf 覆寫）
DEFAULT_GATE_ROC_AUC = 0.6


@task
def split(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """切分訓練 / 測試集（保留標籤分層）。"""
    from sklearn.model_selection import train_test_split

    feature_cols = [
        c
        for c in features.columns
        if c not in (LABEL_COL, "machine_id", "event_timestamp")
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    X = features[feature_cols].to_numpy()
    y = features[LABEL_COL].to_numpy()
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


@task
def train(X_tr: np.ndarray, y_tr: np.ndarray, config: dict):
    """訓練模型：sklearn GradientBoosting 作為與型態無關的編排 smoke 模型。

    註：本編排層刻意用輕量、零重依賴的模型把「DAG 串得起來」示範清楚；
    真正的 XGBoost / ResNet / LSTM 訓練由 ``src.training.train.run`` 在
    MLflow run 內完成（見 train.yml 與 src/training）。
    """
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(
        n_estimators=40, max_depth=2, random_state=config.get("seed", 42)
    )
    model.fit(X_tr, y_tr)
    return model


@task
def evaluate(model, X_te: np.ndarray, y_te: np.ndarray) -> dict:
    """評估：優先用 src.training.evaluate.evaluate_classification，後援自算。"""
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    evaluator = soft_import("src.training.evaluate", "evaluate_classification")
    if evaluator is not None:
        metrics = dict(evaluator(y_te, pred, prob))
    else:
        from sklearn.metrics import f1_score, roc_auc_score

        metrics = {
            "roc_auc": float(roc_auc_score(y_te, prob)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
        }
    logger.info("評估指標: %s", metrics)
    return metrics


@task
def quality_gate(metrics: dict, config: dict) -> bool:
    """品質門檻：優先委派 src.training.evaluate.quality_gate，後援 ROC-AUC 門檻。"""
    gate_fn = soft_import("src.training.evaluate", "quality_gate")
    if gate_fn is not None:
        train_cfg = config.get("train", {})
        try:
            result = gate_fn(metrics, train_cfg)
            passed = bool(getattr(result, "passed", result))
            logger.info("品質門檻（src）-> %s", "通過" if passed else "未通過")
            return passed
        except Exception as exc:  # noqa: BLE001
            logger.warning("src 品質門檻失敗（%s），改用後援。", exc)

    gate = config.get("train", {}).get("gate_roc_auc", DEFAULT_GATE_ROC_AUC)
    passed = metrics.get("roc_auc", 0.0) >= gate
    logger.info("品質門檻 ROC-AUC>=%.3f -> %s", gate, "通過" if passed else "未通過")
    return passed


@task
def register(model, metrics: dict, config: dict) -> str | None:
    """（選配）註冊模型到 MLflow Registry；軟相依，失敗不阻斷流程。"""
    registrar = soft_import("src.training.registry", "register_model")
    if registrar is None:
        logger.info("未提供 src.training.registry，跳過註冊（教學佔位）。")
        return None
    try:
        return registrar(model, metrics, config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("註冊失敗（%s），不阻斷管線。", exc)
        return None


@flow(name="training-pipeline")
def training_flow(config_path: str | None = None) -> dict:
    """訓練管線主流程，回傳含 metrics / gate / model_uri 的結果 dict。"""
    config = load_config(config_path)
    features = feature_flow(config_path)
    X_tr, X_te, y_tr, y_te = split(features)
    model = train(X_tr, y_tr, config)
    metrics = evaluate(model, X_te, y_te)
    passed = quality_gate(metrics, config)
    model_uri = register(model, metrics, config) if passed else None
    return {"metrics": metrics, "gate_passed": passed, "model_uri": model_uri}


def main() -> None:
    """CLI 入口：``python -m pipelines.training_pipeline``。"""
    logging.basicConfig(level=logging.INFO)
    result = training_flow()
    status = "通過並註冊" if result["gate_passed"] else "未達品質門檻"
    print(f"[training-pipeline] {status}；指標={result['metrics']}")


if __name__ == "__main__":
    main()
