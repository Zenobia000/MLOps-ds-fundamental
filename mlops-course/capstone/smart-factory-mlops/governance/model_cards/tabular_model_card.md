# Model Card：設備預測性維護模型（Tabular / XGBoost）

> 格式參考 Google Model Cards（Mitchell et al., 2019）。本卡為 Smart Factory MLOps
> capstone 之教學範例，內容以玩具資料（`datasets/toy_sensors.csv`）情境填寫。

## 1. 模型概述（Model Details）

| 項目 | 內容 |
| :--- | :--- |
| 模型名稱 | `smartfactory-predictive-maintenance` |
| 版本 | v1.0.0 |
| 類型 | 二元分類（故障 / 正常） |
| 演算法 | XGBoost（gradient boosted trees） |
| 擁有者 | MLOps 平台組（mlops@smartfactory.example） |
| 訓練框架 | xgboost, scikit-learn |
| 追蹤 | MLflow experiment `predictive-maintenance` |
| 授權 | 內部使用（Internal Use Only） |
| 最後更新 | 2026-06 |

## 2. 預期用途（Intended Use）

- **主要用途**：依設備感測器讀數（temperature / vibration / current）預測未來短期內的故障機率，
  供維護排程系統提前安排檢修，降低非計畫停機。
- **預期使用者**：產線維護工程師、設備可靠度（reliability）團隊。
- **不適用情境（Out-of-scope）**：
  - 不可作為**安全關鍵**的即時停機決策唯一依據（須人工複核）。
  - 不適用於訓練分布外的新機種 / 新感測器配置，需重新訓練與驗證。
  - 非用於人員績效評估或任何涉及個資的判斷。

## 3. 訓練資料（Training Data）

- 來源：產線設備時序感測器（玩具資料：`datasets/toy_sensors.csv`；真實情境對應 NASA C-MAPSS）。
- 實體鍵：`machine_id`；時間戳：`event_timestamp`。
- 特徵：`temperature`、`vibration`、`current`（及其滾動統計，見 Feast feature view）。
- 標籤：`failure`（0/1）。
- 詳細資料品質與偏誤分析見 `governance/datasheets/sensors_datasheet.md`。

## 4. 評估資料與指標（Evaluation）

- 切分策略：依 `event_timestamp` **時間序切分**，避免未來資訊洩漏（point-in-time）。
- 主要指標：PR-AUC（資料高度不平衡）、Recall@故障（漏抓故障成本高）、ROC-AUC。
- 次要指標：精準率、F1、校準（calibration）。

| 指標 | 驗證集（示例值，TODO 以實際訓練填入） |
| :--- | :--- |
| PR-AUC | 0.71 |
| ROC-AUC | 0.88 |
| Recall（故障類） | 0.83 |
| Precision（故障類） | 0.64 |

> TODO：上述為占位示例值，正式發布前需以 MLflow run 的實測結果替換。

## 5. 倫理考量與限制（Ethical Considerations & Limitations）

- **誤報（False Positive）**：造成不必要檢修與工時浪費，但風險可控。
- **漏報（False Negative）**：可能導致未預期停機，成本高 → 故以 Recall 為重點指標。
- **分布漂移**：感測器老化、製程變更會使線上分布偏離訓練分布，須由 Evidently 監控
  （見 `monitoring/evidently/`）並觸發重訓。
- **公平性**：本模型對象為設備而非人，無個人敏感屬性；但不同機台/班別資料量不均可能造成偏誤，
  需於 datasheet 中持續追蹤。

## 6. 監控與維運（Monitoring）

- 線上即時指標：延遲 / QPS / 預測分布（Prometheus，`src/monitoring/metrics.py`）。
- 批次漂移：每日 Evidently DataDrift 報告。
- 告警：預測分布偏移、錯誤率上升（`monitoring/grafana/prometheus_rules.yml`）。
- 重訓觸發：偵測到資料集漂移或指標退化時，由 M5 CI/CD pipeline 啟動重訓與品質門檻。

## 7. AI Act 對應

依 `governance/ai_act_risk_assessment.md` 分級，本模型屬**有限風險 / 低風險**用途
（工業設備維護，非高風險清單），主要義務為透明度與紀錄留存。
