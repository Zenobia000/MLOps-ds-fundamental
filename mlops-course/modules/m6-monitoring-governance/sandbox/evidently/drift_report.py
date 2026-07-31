"""
drift_report.py — 階 11：用 Evidently 算「資料漂移」報告（Layer 1 單工具沙盒）

這個檔示範什麼
---------------
把資料**按時間**切成兩份（切法很關鍵，見 split_reference_current）：
  - reference（參考分布，想成「模型上線當時、訓練資料的長相」）
  - current  （當前分布，想成「線上現在進來的新資料」）
然後「人工注入漂移」——刻意把某一欄的分布平移，模擬感測器老化 / 環境改變。
最後用 Evidently 的 DataDriftPreset 產生一份 HTML 報告，並在終端機印出
「是否偵測到漂移、哪幾欄漂了」。

這是監控四層裡的第三層（資料漂移層）的最小可用單元：
  系統 → 資料品質 → 【漂移】 → 業務。
你只要會「ref vs current → DataDriftPreset → 報告」這一個動詞就夠了，
test suite / 監控服務 / dashboard 等進階功能之後再回來學。

怎麼跑
------
先裝套件（建議course 統一裝好，這裡列出最小依賴）：
    pip install "evidently>=0.4" pandas numpy scikit-learn

從本檔所在資料夾直接跑：
    python drift_report.py

跑完會在同資料夾產生 drift_report.html，用瀏覽器打開即可看到視覺化報告。
終端機也會印出 dataset_drift（True/False）與漂移欄位數。

備註
----
- 有設 random seed（SEED=42），結果可重現。
- 若 datasets/toy_sensors.csv 還沒建好，本檔會自動產生一份等價的玩具時序資料，
  讓你「現在就能跑起來」，不必等資料就位。
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. 設定 seed —— 任何用到隨機性的腳本都要設，確保可重現
# ----------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# 本檔位置：mlops-course/modules/m6-monitoring-governance/sandbox/evidently/
# 玩具資料位置：mlops-course/datasets/toy_sensors.csv（往上四層回到 mlops-course）
HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[3] / "datasets" / "toy_sensors.csv"
OUTPUT_HTML = HERE / "drift_report.html"


def load_sensors() -> pd.DataFrame:
    """讀取玩具感測器資料；若檔案不存在則就地產生等價的玩具資料。

    toy_sensors.csv 是極簡時序樣本，欄位：
        machine_id, event_timestamp, temperature, vibration, current, failure
    有時間欄時一律**按時間排序**後回傳——reference/current 必須是「不同時間」，
    詳見 split_reference_current 的說明。
    """
    if DATASET.exists():
        print(f"[load] 讀取玩具資料：{DATASET}")
        df = pd.read_csv(DATASET)
        if "event_timestamp" in df.columns:
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
            df = df.sort_values("event_timestamp").reset_index(drop=True)
        return df

    # ---- fallback：資料還沒就位時，自己造一份，保證可獨立執行 ----
    print(f"[load] 找不到 {DATASET}，改用內建合成玩具資料（不影響教學概念）")
    n = 400
    df = pd.DataFrame(
        {
            "temperature": rng.normal(loc=25.0, scale=2.0, size=n),   # 溫度 ~25°C
            "vibration": rng.normal(loc=0.5, scale=0.1, size=n),      # 振動
            "pressure": rng.normal(loc=101.3, scale=1.0, size=n),     # 壓力
        }
    )
    return df


# 要監控的欄位：只取「上線後拿得到的感測器讀值」。
# 刻意排除 machine_id（識別碼，不是分布）、event_timestamp（時間本身）、
# failure（標籤——線上推論當下根本還沒有它，拿它算漂移是自欺欺人）。
MONITORED_COLS = ["temperature", "vibration", "current"]


def split_reference_current(df: pd.DataFrame):
    """把資料切成 reference（過去）/ current（現在）兩份 DataFrame。

    ★ 一定要按「時間」切，不能按列序切。
      toy_sensors.csv 是「一台機器接一台機器」排序的，照列序對半切等於
      **照機台切**——而不同機台的溫度基準天生就差好幾度。那樣切出來的
      reference/current，比較的是「不同機器」而不是「不同時間」，
      還沒注入任何漂移就會報 drift=True，監控從第一天就在說謊。
      load_sensors 已先按時間排序，這裡才能安全地對半切。

    只取 MONITORED_COLS（數值欄）比較分布——Evidently 對數值欄用統計檢定。
    """
    cols = [c for c in MONITORED_COLS if c in df.columns]
    if not cols:  # fallback 合成資料沒有這些欄名，退回「所有數值欄」
        cols = list(df.select_dtypes(include="number").columns)

    ordered = df[cols].reset_index(drop=True)
    half = len(ordered) // 2
    reference = ordered.iloc[:half].reset_index(drop=True)
    current = ordered.iloc[half:].reset_index(drop=True)
    return reference, current


def inject_drift(current: pd.DataFrame, column: str, shift: float) -> pd.DataFrame:
    """人工注入漂移：把 current 的某一欄整體平移 shift。

    不可變寫法 —— 回傳新的 DataFrame，不改動傳入的物件。
    平移分布的均值 = covariate drift（輸入特徵分布改變）的最直接模擬。
    """
    drifted = current.copy()
    drifted[column] = drifted[column] + shift
    print(f"[drift] 已對欄位 '{column}' 注入平移 +{shift}（模擬感測器老化）")
    return drifted


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    """用 DataDriftPreset 產生報告，並回傳 (html 是否已存檔, 漂移摘要 dict)。

    Evidently 在不同版本 API 不同，這裡同時相容：
      - 舊版 (>=0.4, <0.6)：evidently.report.Report + evidently.metric_preset
      - 新版 (>=0.6)       ：evidently.Report + evidently.presets
    """
    summary = {}

    # ---- 嘗試新版 API（0.6+）----
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        result.save_html(str(OUTPUT_HTML))
        # 新版用 dict() 取結果摘要
        as_dict = result.dict()
        summary = _extract_drift_summary_new(as_dict)
        print("[report] 使用 Evidently 新版 API（>=0.6）")
        return True, summary
    except Exception:
        pass

    # ---- 退回舊版 API（>=0.4, <0.6）----
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(OUTPUT_HTML))
    as_dict = report.as_dict()
    summary = _extract_drift_summary_old(as_dict)
    print("[report] 使用 Evidently 舊版 API（0.4.x / 0.5.x）")
    return True, summary


def _extract_drift_summary_old(as_dict: dict) -> dict:
    """從舊版 as_dict() 取出整體結論 + 逐欄明細（哪幾欄漂了）。"""
    summary = {
        "dataset_drift": None, "n_drifted_columns": None,
        "n_columns": None, "drift_share": None, "drifted_columns": [],
    }
    for m in as_dict.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            summary.update(
                dataset_drift=res.get("dataset_drift"),
                n_drifted_columns=res.get("number_of_drifted_columns"),
                n_columns=res.get("number_of_columns"),
                drift_share=res.get("drift_share"),
            )
        if m.get("metric") == "DataDriftTable":
            by_col = m.get("result", {}).get("drift_by_columns", {})
            summary["drifted_columns"] = [
                name for name, info in by_col.items() if info.get("drift_detected")
            ]
    return summary


def _extract_drift_summary_new(as_dict: dict) -> dict:
    """從新版 dict() 盡力取出整體漂移結論（欄位名稱依版本略有不同，盡量容錯）。"""
    text = str(as_dict).lower()
    return {
        "dataset_drift": "drift" in text,  # 新版結構多變，這裡只做粗略提示
        "n_drifted_columns": None,
        "n_columns": None,
        "drift_share": None,
        "drifted_columns": [],
    }


def main():
    print("=" * 64)
    print(" Evidently 資料漂移沙盒：reference vs current")
    print("=" * 64)

    df = load_sensors()
    reference, current = split_reference_current(df)
    print(f"[split] reference={len(reference)} 列, current={len(current)} 列, "
          f"欄位={list(reference.columns)}")

    # 對第一個數值欄注入明顯漂移；平移量取該欄標準差的 3 倍，確保偵測得到
    target_col = reference.columns[0]
    shift = 3.0 * float(reference[target_col].std())
    current = inject_drift(current, column=target_col, shift=shift)

    saved, summary = build_drift_report(reference, current)

    print("-" * 64)
    # ★ 逐欄結果才是這個 demo 的重點：我們只推了一欄，就該只有那一欄被標記。
    if summary.get("n_drifted_columns") is not None:
        drifted = summary.get("drifted_columns") or []
        print(f"[結果] 逐欄：{summary['n_drifted_columns']} / {summary['n_columns']} "
              f"欄偵測到漂移" + (f" → {', '.join(drifted)}" if drifted else ""))

    # ★ 整體旗標會是 False,這不是失敗——它有自己的門檻,務必看懂再往下走。
    drift = summary.get("dataset_drift")
    share = summary.get("drift_share")
    print(f"[結果] 整體 dataset_drift = {drift}"
          + (f"(門檻 drift_share={share}:要超過這個比例的欄位漂移才亮燈)" if share else ""))
    if drift is False and summary.get("n_drifted_columns"):
        print("        ↑ 注意:有欄位漂了但整體仍是 False,因為漂移欄位比例沒過門檻。")
        print("          只看整體旗標會漏掉單欄漂移——生產環境請以逐欄結果為準,")
        print("          或依風險承受度調低 drift_share。")

    if saved:
        print(f"[結果] HTML 報告已輸出：{OUTPUT_HTML}")
        print("        用瀏覽器打開它，對照看哪一欄的分布被推開了。")
    print("=" * 64)


if __name__ == "__main__":
    main()
