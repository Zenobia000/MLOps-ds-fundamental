# Evidently 漂移監控（M6）

本目錄存放資料 / 預測漂移偵測的設定與參考快照，搭配 `src/monitoring/drift.py` 使用。

## 目錄結構

```
monitoring/evidently/
├── README.md                 # 本說明
├── drift_config.yaml         # 漂移監控設定（欄位、閾值、輸出路徑）
├── reference/                # 參考分布快照（不進 Git，用 DVC 追蹤）
│   ├── tabular_reference.parquet
│   └── vision_reference.parquet
└── reports/                  # 產生的 HTML 報告（gitignore）
    ├── tabular_drift.html
    └── vision_drift.html
```

## 核心概念：reference vs current

漂移偵測比較兩個分布：

| 名稱 | 來源 | 角色 |
| :--- | :--- | :--- |
| **reference** | 訓練集統計快照 | 「模型認得的世界」基準 |
| **current** | 線上推論輸入（最近 N 小時） | 「模型現在看到的世界」 |

當 current 對比 reference 出現顯著分布偏移（如 `temperature` 平均上移、`vibration`
變異放大），代表線上資料已偏離訓練分布，模型表現可能退化，應觸發告警或重訓。

## 如何建立 reference 快照

reference 快照應在**模型訓練當下**從訓練集擷取，與該版本模型綁定：

```bash
# 概念示範（實際擷取邏輯由 training agent 在訓練流程中產生）
# 將訓練集的特徵欄位另存為 parquet 作為基準分布
python -m src.training.snapshot_reference \
    --input data/processed/sensors_train.parquet \
    --output monitoring/evidently/reference/tabular_reference.parquet
```

> 注意：reference 快照**隨模型版本更新**。每次重訓並升版模型時，應同步刷新對應快照，
> 否則漂移基準會與線上模型脫鉤，導致誤報或漏報。

## 如何執行漂移偵測

```python
import pandas as pd
import yaml
from src.monitoring.drift import run_data_drift

cfg = yaml.safe_load(open("monitoring/evidently/drift_config.yaml"))["tabular"]
reference = pd.read_parquet(cfg["reference_snapshot"])
current = pd.read_parquet("data/online/sensors_last_24h.parquet")  # 線上資料

result = run_data_drift(
    reference=reference,
    current=current,
    columns=cfg["columns"],
    report_path=cfg["report_output"],
)

if result.dataset_drift:
    print(f"偵測到資料集漂移！漂移特徵佔比={result.share_drifted_features:.0%}")
    # → 觸發 Prometheus alert / 重訓 pipeline
```

## 與監控閉環的關係

```
線上推論 → Prometheus 即時指標（延遲/QPS/預測分布，src/monitoring/metrics.py）
         → Evidently 批次漂移報告（本目錄，每日）
         → Grafana 視覺化 + Alertmanager 告警（monitoring/grafana/）
         → 觸發重訓（M5 CI/CD pipeline）
```

Prometheus 指標負責**即時、輕量**的線上監控；Evidently 負責**批次、深入**的分布分析，
兩者互補，共同構成 M6 監控閉環。
