# Model Card（M6 填空）

> 對照本模組 README「治理」段。用你 workspace 實際模型填，不要留空話。

## 1. 模型概要

- **名稱 / 版本**：TODO(M6-5)
- **問題類型**：TODO
- **框架**：TODO（sklearn / …）
- **訓練資料**：TODO（路徑 + DVC 版本若有）

## 2. 預期用途

- **要用在哪**：TODO
- **不要用在哪（適用邊界）**：TODO(M6-6) — 寫清漂移到什麼程度就重訓/下線

## 3. 指標

| 指標 | 數值 | 資料切分 |
| :--- | :--- | :--- |
| accuracy | TODO | test |
| … | | |

## 4. 倫理 / 風險 / AI Act 自評

- **風險等級（自評）**：TODO(M6-7) — minimal / limited / high / unacceptable
- **理由**：TODO
- **人為複核點**：TODO

## 5. 監控計畫

- **reference 資料位置**：TODO
- **漂移檢查頻率**：TODO
- **告警 / 重訓觸發條件**：TODO（對齊 drift_report + Prefect）
