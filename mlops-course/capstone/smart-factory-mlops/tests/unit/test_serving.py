"""服務層測試 —— 補上 test_plan §6 列為最優先的覆蓋缺口。

為什麼服務層特別需要測試
------------------------
訓練錯了，指標會掉，你看得出來。**服務層錯了，通常沒有任何訊號**：
schema 少驗一個範圍、門檻比較寫成 ``>`` 而不是 ``>=``、前處理的正規化係數不對——
這些都不會拋例外，只會讓線上預測安靜地變差。

所以這裡測的重點不是「函式跑得動」，而是**邊界與契約**：
範圍擋不擋得住、門檻在等號上怎麼判、降級行為對不對。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.serving.healthcheck import probe
from src.serving.predict import predict_defect, predict_maintenance, preprocess_image
from src.serving.schemas import (
    HealthResponse,
    MaintenanceRequest,
    SensorReading,
)

# ---------------------------------------------------------------------------
# Schema：邊界驗證。這些數字是擋掉感測器故障值的第一道防線。
# ---------------------------------------------------------------------------


def _reading(**overrides) -> dict:
    base = {"machine_id": "machine_01", "temperature": 70.0, "vibration": 3.0, "current": 10.0}
    base.update(overrides)
    return base


def test_valid_reading_is_accepted() -> None:
    r = SensorReading(**_reading())
    assert r.machine_id == "machine_01"
    assert r.event_timestamp is None  # 可選欄位，純表格推論不需要


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -50.1),  # 低於下界
        ("temperature", 500.1),  # 高於上界
        ("vibration", -0.1),  # 振動不可為負
        ("current", -0.1),  # 電流不可為負
    ],
)
def test_out_of_range_sensor_values_are_rejected(field: str, value: float) -> None:
    """超出物理合理範圍的值必須被擋下，而不是餵進模型。"""
    with pytest.raises(ValidationError):
        SensorReading(**_reading(**{field: value}))


@pytest.mark.parametrize("bound", [-50.0, 500.0])
def test_range_bounds_are_inclusive(bound: float) -> None:
    """邊界值本身是合法的（ge/le 而非 gt/lt）——差一個等號就會誤擋正常讀數。"""
    assert SensorReading(**_reading(temperature=bound)).temperature == bound


def test_missing_required_field_is_rejected() -> None:
    payload = _reading()
    del payload["temperature"]
    with pytest.raises(ValidationError):
        SensorReading(**payload)


def test_batch_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        MaintenanceRequest(readings=[])


def test_batch_cap_is_enforced() -> None:
    """1000 筆上限是防止單一請求耗盡記憶體的保護，不是建議值。"""
    ok = MaintenanceRequest(readings=[SensorReading(**_reading())] * 1000)
    assert len(ok.readings) == 1000
    with pytest.raises(ValidationError):
        MaintenanceRequest(readings=[SensorReading(**_reading())] * 1001)


# ---------------------------------------------------------------------------
# 推論：門檻判定。等號落在哪一邊會改變線上的告警量。
# ---------------------------------------------------------------------------


class _StubModel:
    """回傳固定機率的假模型；測門檻邏輯不需要真的訓練。"""

    def __init__(self, proba: list[float]) -> None:
        self._proba = np.asarray(proba, dtype=np.float32)

    def predict(self, features):  # noqa: ANN001, ARG002
        return self._proba


def test_threshold_is_inclusive_on_the_boundary() -> None:
    """機率**等於**門檻時判定為 will_fail —— 契約是 p >= threshold。"""
    readings = [SensorReading(**_reading())]
    out = predict_maintenance(_StubModel([0.5]), readings, threshold=0.5)
    assert out[0].will_fail is True


def test_threshold_separates_above_and_below() -> None:
    readings = [SensorReading(**_reading(machine_id=f"m{i}")) for i in range(3)]
    out = predict_maintenance(_StubModel([0.9, 0.5, 0.1]), readings, threshold=0.5)
    assert [p.will_fail for p in out] == [True, True, False]


def test_prediction_preserves_machine_id_order() -> None:
    """回應必須逐筆對齊請求順序，否則預測會被貼到錯的設備上。"""
    ids = ["m_a", "m_b", "m_c"]
    readings = [SensorReading(**_reading(machine_id=i)) for i in ids]
    out = predict_maintenance(_StubModel([0.1, 0.2, 0.3]), readings, threshold=0.5)
    assert [p.machine_id for p in out] == ids


def test_defect_softmax_two_logits() -> None:
    """兩個 logits 走 softmax，取 defect 那一類的機率。"""
    result = predict_defect(np.array([[0.0, 5.0]]), threshold=0.5)
    assert result.label == "defect"
    assert result.defect_probability > 0.99


def test_defect_below_threshold_is_good() -> None:
    result = predict_defect(np.array([[5.0, 0.0]]), threshold=0.5)
    assert result.label == "good"
    assert result.defect_probability < 0.01


def test_defect_probability_within_unit_interval() -> None:
    """機率必須落在 [0,1]——schema 也擋，但這裡確保後處理本身不會產出離譜值。"""
    for logits in ([[-40.0, 40.0]], [[40.0, -40.0]], [[0.0, 0.0]]):
        p = predict_defect(np.array(logits), threshold=0.5).defect_probability
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# 前處理：ADR-004 標為最大風險（訓練用 torchvision、服務用 numpy，兩邊要一致）
# ---------------------------------------------------------------------------


def test_preprocess_outputs_nchw_batch() -> None:
    out = preprocess_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32


def test_preprocess_applies_imagenet_normalisation() -> None:
    """全黑影像經 0-1 縮放後為 0，再標準化應得 -mean/std（各通道不同）。

    這一條在守 ADR-004 指出的風險：服務端的正規化係數若與訓練期不同，
    模型不會報錯，只會安靜地降準。
    """
    out = preprocess_image(np.zeros((8, 8, 3), dtype=np.uint8))
    per_channel = out[0].reshape(3, -1).mean(axis=1)
    expected = -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(per_channel, expected, rtol=1e-4)


def test_preprocess_is_deterministic() -> None:
    img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    np.testing.assert_array_equal(preprocess_image(img), preprocess_image(img))


# ---------------------------------------------------------------------------
# 健康檢查：canary 的判定依據
# ---------------------------------------------------------------------------


def test_probe_returns_zero_when_service_unreachable() -> None:
    """連不上時必須是 0.0。預設值站錯邊會把壞版本推上線。"""
    assert probe("img:test", attempts=2, timeout=0.05) == 0.0


def test_probe_counts_degraded_as_failure() -> None:
    """★ 關鍵契約：degraded 不算成功。

    服務活著但少了一個模型，liveness 可以容忍，canary 不行——
    它要回答的是「這個版本能承接生產流量嗎」。
    """
    with patch("src.serving.healthcheck._probe_once", return_value=False):
        assert probe("img", attempts=4) == 0.0


def test_probe_computes_success_rate() -> None:
    outcomes = iter([True, True, False, True])
    with patch("src.serving.healthcheck._probe_once", side_effect=lambda *a: next(outcomes)):
        assert probe("img", attempts=4) == 0.75


def test_probe_rejects_non_positive_attempts() -> None:
    with pytest.raises(ValueError):
        probe("img", attempts=0)


def test_health_response_defaults_to_not_loaded() -> None:
    """預設值是「沒載到」——樂觀的預設會讓未就緒的服務看起來是健康的。"""
    h = HealthResponse()
    assert h.status == "ok"
    assert h.tabular_model_loaded is False
    assert h.vision_model_loaded is False
