"""Registry 讀寫測試 —— 訓練側註冊、服務側解析。

這兩支的共同特性是**失敗不該中斷流程**：registry 連不上時，
訓練仍要留下本地模型、服務仍要能靠本地產物降級啟動。
所以測試的重點在「失敗時回傳什麼」，而不只是成功路徑。

不連真的 MLflow：用假的 client 驗證呼叫契約與回退行為。
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from src.serving.registry import resolve_latest
from src.training.registry import register_model, registered_name

CFG = {
    "project": "smart-factory",
    "active_model": "xgboost",
    "mlflow": {"tracking_uri": "sqlite:///test.db"},
    "serving": {"tabular_model_stage": "Production"},
}


# ---------------------------------------------------------------------------
# 命名規則：註冊時寫入的名字與服務端查詢的名字必須用同一條規則
# ---------------------------------------------------------------------------


def test_registered_name_combines_project_and_model() -> None:
    assert registered_name(CFG, "xgboost") == "smart-factory-xgboost"


def test_registered_name_falls_back_to_default_project() -> None:
    assert registered_name({}, "resnet") == "smart-factory-resnet"


# ---------------------------------------------------------------------------
# 訓練側：register_model
# ---------------------------------------------------------------------------


def _fake_mlflow(version: str = "3"):
    """組一個假的 mlflow 模組，記錄呼叫並回傳固定版本。"""
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = MagicMock()
    fake.register_model = MagicMock(return_value=types.SimpleNamespace(version=version))
    client = MagicMock()
    fake.tracking = types.SimpleNamespace(MlflowClient=MagicMock(return_value=client))
    return fake, client


def test_register_model_sets_alias_to_new_version() -> None:
    fake, client = _fake_mlflow(version="7")
    with patch.dict("sys.modules", {"mlflow": fake}):
        out = register_model("runs:/abc/model", {"f1": 0.91}, CFG, model_name="xgboost")

    assert out == "smart-factory-xgboost/7"
    fake.register_model.assert_called_once_with("runs:/abc/model", "smart-factory-xgboost")
    client.set_registered_model_alias.assert_called_once_with(
        "smart-factory-xgboost", "champion", "7"
    )


def test_register_model_accepts_object_with_model_uri() -> None:
    """MLflow 的 ModelInfo 帶 .model_uri；train.py 直接把它傳進來。"""
    fake, _ = _fake_mlflow()
    info = types.SimpleNamespace(model_uri="runs:/xyz/model")
    with patch.dict("sys.modules", {"mlflow": fake}):
        register_model(info, {}, CFG, model_name="xgboost")
    fake.register_model.assert_called_once_with("runs:/xyz/model", "smart-factory-xgboost")


def test_register_model_returns_none_when_uri_unresolvable() -> None:
    """傳進來的東西取不到 uri 就放棄，而不是拿一個亂猜的字串去註冊。"""
    assert register_model(object(), {}, CFG) is None


def test_register_model_survives_registry_failure() -> None:
    """★ registry 掛掉不該讓整條訓練白跑——回 None，不拋例外。"""
    fake, _ = _fake_mlflow()
    fake.register_model = MagicMock(side_effect=RuntimeError("registry 連不上"))
    with patch.dict("sys.modules", {"mlflow": fake}):
        assert register_model("runs:/abc/model", {}, CFG, model_name="xgboost") is None


def test_metric_tag_failure_does_not_break_registration() -> None:
    """寫 metric tag 是加分項；它失敗不該讓已完成的註冊被判定為失敗。"""
    fake, client = _fake_mlflow(version="2")
    client.set_model_version_tag.side_effect = RuntimeError("tag 不支援")
    with patch.dict("sys.modules", {"mlflow": fake}):
        assert (
            register_model("runs:/a/model", {"f1": 0.8}, CFG, model_name="xgboost")
            == "smart-factory-xgboost/2"
        )


# ---------------------------------------------------------------------------
# 服務側：resolve_latest 的三層解析順序
# ---------------------------------------------------------------------------


def _patch_client(client):
    return patch("src.serving.registry._client", return_value=client)


def test_resolve_prefers_alias() -> None:
    """有 alias 就用 alias——那是「有人明確背書過」的版本。"""
    client = MagicMock()
    client.get_model_version_by_alias.return_value = types.SimpleNamespace(version="5")
    with _patch_client(client):
        assert resolve_latest(CFG) == "models:/smart-factory-xgboost@champion"


def test_resolve_falls_back_to_stage() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("無此 alias")
    client.get_latest_versions.return_value = [types.SimpleNamespace(version="4")]
    with _patch_client(client):
        assert resolve_latest(CFG) == "models:/smart-factory-xgboost/Production"


def test_resolve_falls_back_to_highest_version_number() -> None:
    """最後手段：版本號最大的那個。注意它沒有任何背書。"""
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("無此 alias")
    client.get_latest_versions.return_value = []
    client.search_model_versions.return_value = [
        types.SimpleNamespace(version="2"),
        types.SimpleNamespace(version="10"),  # 字串排序會選錯，必須轉 int 比較
        types.SimpleNamespace(version="9"),
    ]
    with _patch_client(client):
        assert resolve_latest(CFG) == "models:/smart-factory-xgboost/10"


def test_resolve_returns_none_when_registry_unavailable() -> None:
    """registry 不可用時回 None，讓呼叫端自己決定回退，而不是拋例外。"""
    with _patch_client(None):
        assert resolve_latest(CFG) is None


def test_resolve_returns_none_when_model_absent() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RuntimeError("無此 alias")
    client.get_latest_versions.return_value = []
    client.search_model_versions.return_value = []
    with _patch_client(client):
        assert resolve_latest(CFG) is None
