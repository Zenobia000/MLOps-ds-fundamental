"""
workspace/monitoring/drift_report.py — M6 填空：Evidently 資料漂移

對照：modules/m6-monitoring-governance/sandbox/evidently/drift_report.py
怎麼跑：python workspace/monitoring/drift_report.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

HERE = Path(__file__).resolve().parent
COURSE_ROOT = HERE.parents[1]
DATASET = COURSE_ROOT / "datasets" / "toy_sensors.csv"
OUT_HTML = HERE / "drift_report.html"


def load_frame() -> pd.DataFrame:
    """TODO(M6-1): 讀 toy_sensors.csv；不存在可 fallback 生玩具資料（見 sandbox）。"""
    raise NotImplementedError("TODO(M6-1): load_frame")


def split_reference_current(df: pd.DataFrame):
    """TODO(M6-2): 切 reference / current；對 current 某欄做人工平移注入漂移。"""
    raise NotImplementedError("TODO(M6-2): split + inject drift")


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """TODO(M6-3): DataDriftPreset → 存 HTML → print dataset_drift。

    對照 sandbox 的 Evidently API（注意 0.4 vs 0.6+ 差異）。
    """
    raise NotImplementedError("TODO(M6-3): Evidently report")


def main() -> None:
    df = load_frame()
    reference, current = split_reference_current(df)
    build_report(reference, current)
    print(f"報告：{OUT_HTML}")
    # TODO(M6-4 進階): dataset_drift=True 時接回 M5 Prefect（重訓或告警 task）


if __name__ == "__main__":
    main()
