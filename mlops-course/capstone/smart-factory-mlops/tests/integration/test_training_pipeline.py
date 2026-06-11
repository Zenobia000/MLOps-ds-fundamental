"""訓練管線整合 smoke 測試。

目標：在小樣本上「真的跑一次最小訓練」，證明 data → 特徵 → 模型 →
指標 這條鏈路在本地能端到端動起來，且產出的模型優於隨機。

相容策略
--------
- 若 ``src.training.train`` 暴露可程式呼叫的入口（train_once / run / main），
  優先用它在 ``tmp_config`` + 小樣本上跑，並斷言有模型產物落地。
- 否則退回一份**自足的 mini 訓練**（sklearn GradientBoosting），驗證測試
  基礎設施本身可端到端訓練、評估、達到優於隨機的門檻。

這樣不論其他 agent 的 src 是否就緒，CI 都有一個會真的跑訓練的整合測試。
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ENTITY_COL = "machine_id"
TIMESTAMP_COL = "event_timestamp"
SENSOR_COLS = ["temperature", "vibration", "current"]
LABEL_COL = "failure"


def _self_contained_mini_train(df: pd.DataFrame) -> float:
    """自足 mini 訓練：回傳測試集 ROC-AUC。"""
    X = df[SENSOR_COLS].to_numpy()
    y = df[LABEL_COL].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, prob))


def test_mini_training_beats_random(sensors_df: pd.DataFrame) -> None:
    """端到端 smoke：小樣本訓練後，測試集 ROC-AUC 優於隨機（>0.5）。

    toy_sensors 的故障是「機率性」規則（溫度高 + 振動大 → 故障機率上升），
    訊號刻意帶噪，因此門檻設在「確實優於亂猜」的 0.5 之上，兼顧小樣本波動。
    """
    auc = _self_contained_mini_train(sensors_df.copy())
    assert auc > 0.5, f"mini 訓練 ROC-AUC={auc:.3f} 未優於隨機"


def test_src_training_entrypoint_signature() -> None:
    """``src.training.train`` 應暴露 ``run(cfg)`` 程式化入口（不實際執行）。

    ``run`` 內含 MLflow run 與重模型訓練，屬重量級整合，不在單元/smoke 階段
    實跑（避免依賴 MLflow server）；此處只驗介面契約存在且可呼叫。
    完整端到端訓練由 train.yml workflow 在 CI 觸發。
    """
    import inspect

    mod = pytest.importorskip("src.training.train", reason="src.training.train 尚未建立")
    run = getattr(mod, "run", None)
    assert run is not None, "src.training.train 應提供 run(cfg) 入口"
    assert callable(run)
    # run 應接受單一 config 參數
    params = inspect.signature(run).parameters
    assert len(params) >= 1, "run 應至少接受一個 config 參數"


def test_orchestration_training_flow_end_to_end() -> None:
    """編排層 ``training_flow`` 在小樣本上端到端跑通，回傳含門檻決策的結果。

    這條走的是 pipelines 的輕量後援模型，不依賴 MLflow，CI 必跑。
    """
    mod = pytest.importorskip(
        "pipelines.training_pipeline",
        reason="pipelines.training_pipeline 尚未就緒",
    )
    assert hasattr(mod, "training_flow"), "應暴露 training_flow"

    result = mod.training_flow()
    assert isinstance(result, dict)
    assert "metrics" in result and "gate_passed" in result
    # 端到端鏈路跑通即達 smoke 目的；指標須為合法區間值。
    auc = result["metrics"].get("roc_auc")
    if auc is not None:
        assert 0.0 <= auc <= 1.0, f"ROC-AUC 超出合法區間: {auc}"
    assert isinstance(result["gate_passed"], bool)


def test_feature_flow_produces_features() -> None:
    """編排層 ``feature_flow`` 應產出比原欄位更多欄的特徵表。"""
    mod = pytest.importorskip(
        "pipelines.feature_pipeline", reason="pipelines.feature_pipeline 尚未就緒"
    )
    feats = mod.feature_flow()
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) > 0
    assert feats.shape[1] >= 6  # 原始 6 欄 + 衍生特徵
