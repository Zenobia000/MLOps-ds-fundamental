"""
workspace/tests/test_pipeline.py — M5 CI 用最小測試

對照：modules/m5-automation/sandbox/tests/test_sample.py
把 sandbox 的 ci.yml 複製到 repo 根 .github/workflows/ 後，
把 pytest 路徑指到 workspace/tests/。
"""


def test_seed_is_fixed():
    """TODO(M5-7): 斷言訓練腳本使用的 SEED 是固定整數（可 import train）。"""
    raise NotImplementedError("TODO(M5-7): test_seed_is_fixed")


def test_accuracy_threshold_configured():
    """TODO(M5-8): 斷言 flow.ACCURACY_THRESHOLD 不為 None 且在 (0, 1]。"""
    raise NotImplementedError("TODO(M5-8): test_accuracy_threshold_configured")
