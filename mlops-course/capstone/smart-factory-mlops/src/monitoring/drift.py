"""資料漂移偵測（Evidently 封裝）。

以參考資料集（reference，通常為訓練分布快照）對比當前資料集（current，
線上推論輸入），產出 Data Drift 報告並回傳結構化結果。

Evidently 在 0.4.x 後大改 API（``Report`` 移至 ``evidently.report``，
preset 移至 ``evidently.metric_preset``）；更舊版本則位於不同路徑。
本模組以 try/except 相容新舊版 API，import 失敗時拋出明確錯誤而非靜默吞掉。

典型用法::

    from src.monitoring.drift import run_data_drift
    result = run_data_drift(ref_df, cur_df, columns=["temperature", "vibration", "current"])
    if result.dataset_drift:
        ...  # 觸發告警 / 重訓
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# --- 相容新舊版 Evidently API ------------------------------------------------
# 新版（>= 0.4）：evidently.report.Report + evidently.metric_preset.DataDriftPreset
# 舊版（< 0.4）：evidently.report.Report 仍在，但 preset 路徑可能不同。
try:  # pragma: no cover - 取決於安裝版本
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    _EVIDENTLY_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc_new:  # noqa: BLE001 - 嘗試更舊路徑
    try:  # pragma: no cover
        from evidently.report import Report  # type: ignore
        from evidently.metric_preset.data_drift import DataDriftPreset  # type: ignore

        _EVIDENTLY_AVAILABLE = True
        _IMPORT_ERROR = None
    except Exception as exc_old:  # noqa: BLE001
        Report = None  # type: ignore
        DataDriftPreset = None  # type: ignore
        _EVIDENTLY_AVAILABLE = False
        _IMPORT_ERROR = exc_old or exc_new


@dataclass
class DriftResult:
    """漂移偵測結果（結構化、可序列化）。

    Attributes:
        dataset_drift: 整體資料集是否判定為漂移。
        n_drifted_features: 偵測到漂移的特徵數。
        share_drifted_features: 漂移特徵佔比（0~1）。
        drift_by_column: 各欄位是否漂移的對照表。
        report_path: 若有匯出 HTML 報告，其路徑；否則為 None。
    """

    dataset_drift: bool
    n_drifted_features: int
    share_drifted_features: float
    drift_by_column: dict[str, bool] = field(default_factory=dict)
    report_path: Path | None = None


def _ensure_available() -> None:
    """確認 Evidently 可用，否則拋出明確的 ImportError。"""
    if not _EVIDENTLY_AVAILABLE:
        raise ImportError(
            "未偵測到可用的 Evidently，請先 `pip install evidently`。"
            f"原始 import 錯誤：{_IMPORT_ERROR!r}"
        )


def run_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str] | None = None,
    report_path: str | Path | None = None,
) -> DriftResult:
    """執行資料漂移偵測。

    Args:
        reference: 參考分布（訓練快照）。
        current: 當前線上資料。
        columns: 僅比對的欄位；None 表示比對兩者共同欄位。
        report_path: 若提供，輸出 Evidently HTML 報告至此路徑。

    Returns:
        DriftResult：結構化漂移結果。

    Raises:
        ImportError: Evidently 未安裝。
        ValueError: 兩個 DataFrame 無共同可比對欄位。
    """
    _ensure_available()

    if columns is None:
        columns = [c for c in reference.columns if c in current.columns]
    columns = list(columns)
    if not columns:
        raise ValueError("reference 與 current 無共同欄位可供比對。")

    ref = reference.loc[:, columns]
    cur = current.loc[:, columns]

    report = Report(metrics=[DataDriftPreset()])
    # 新舊版 run() 簽名一致：reference_data / current_data。
    report.run(reference_data=ref, current_data=cur)

    summary = _extract_summary(report)

    out_path: Path | None = None
    if report_path is not None:
        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))

    return DriftResult(
        dataset_drift=summary["dataset_drift"],
        n_drifted_features=summary["n_drifted"],
        share_drifted_features=summary["share_drifted"],
        drift_by_column=summary["by_column"],
        report_path=out_path,
    )


def _extract_summary(report: Any) -> dict[str, Any]:
    """從 Evidently report 物件萃取摘要，相容新舊版 dict 結構。"""
    payload = report.as_dict()
    metrics = payload.get("metrics", [])
    dataset_result: dict[str, Any] = {}
    by_column: dict[str, bool] = {}

    for metric in metrics:
        result = metric.get("result", {})
        if "dataset_drift" in result:
            dataset_result = result
        # 各欄位漂移細節（新版放在 drift_by_columns）。
        for col, info in result.get("drift_by_columns", {}).items():
            by_column[col] = bool(info.get("drift_detected", False))

    n_drifted = int(dataset_result.get("number_of_drifted_columns", sum(by_column.values())))
    n_total = int(dataset_result.get("number_of_columns", max(len(by_column), 1)))
    share = float(dataset_result.get("share_of_drifted_columns", n_drifted / n_total))

    return {
        "dataset_drift": bool(dataset_result.get("dataset_drift", n_drifted > 0)),
        "n_drifted": n_drifted,
        "share_drifted": share,
        "by_column": by_column,
    }
