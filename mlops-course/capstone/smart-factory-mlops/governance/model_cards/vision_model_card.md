# Model Card：產線視覺瑕疵檢測模型（Vision / ResNet）

> 格式參考 Google Model Cards（Mitchell et al., 2019）。本卡為 Smart Factory MLOps
> capstone 之教學範例；真實情境對應 MVTec AD 資料集。

## 1. 模型概述（Model Details）

| 項目 | 內容 |
| :--- | :--- |
| 模型名稱 | `smartfactory-defect-detection` |
| 版本 | v1.0.0 |
| 類型 | 影像二元分類（瑕疵 / 正常） |
| 演算法 | 預訓練 ResNet-18/50 微調（transfer learning） |
| 推論格式 | PyTorch → 匯出 ONNX 供 serving |
| 擁有者 | MLOps 平台組（mlops@smartfactory.example） |
| 訓練框架 | PyTorch, torchvision |
| 追蹤 | MLflow experiment `defect-detection` |
| 授權 | 內部使用（Internal Use Only） |
| 最後更新 | 2026-06 |

## 2. 預期用途（Intended Use）

- **主要用途**：對產線拍攝之零件影像進行瑕疵檢測，作為人工目檢的輔助與初篩，提升良率與一致性。
- **預期使用者**：品保（QA）人員、產線檢測站。
- **不適用情境（Out-of-scope）**：
  - 不可作為**唯一**的放行/報廢決策依據，須保留人工複核機制（human-in-the-loop）。
  - 不適用於訓練影像未涵蓋的新產品、新瑕疵型態、不同光源/鏡頭設定。
  - 非用於任何人臉或人員影像分析。

## 3. 訓練資料（Training Data）

- 來源：產線視覺檢測站影像（真實情境對應 MVTec AD；教學以小樣本玩具影像 smoke test）。
- 取得方式與授權見 `data/README`（MVTec AD 為學術授權，需自行下載至 `data/external/`）。
- 前處理：resize、normalize（ImageNet 統計）、資料增強（翻轉、亮度抖動）。
- 類別平衡：瑕疵樣本通常稀少，採過採樣 / 加權損失處理不平衡。

## 4. 評估資料與指標（Evaluation）

- 切分：依產品批次切分，避免同批影像同時落在 train/val 造成洩漏。
- 主要指標：Recall（漏檢瑕疵成本高）、Precision、PR-AUC。
- 影像級補充：混淆矩陣、各瑕疵類別 per-class recall。

| 指標 | 驗證集（示例值，TODO 以實際訓練填入） |
| :--- | :--- |
| Recall（瑕疵類） | 0.90 |
| Precision（瑕疵類） | 0.78 |
| PR-AUC | 0.85 |

> TODO：上述為占位示例值，正式發布前需以 MLflow run 的實測結果替換。

## 5. 倫理考量與限制（Ethical Considerations & Limitations）

- **漏檢（False Negative）**：瑕疵流入下游，品質風險高 → 以 Recall 為首要指標，並設保守閾值。
- **過檢（False Positive）**：良品被誤判報廢，造成成本浪費，需平衡閾值。
- **領域漂移**：光源變化、鏡頭更換、新產品上線會顯著影響影像分布，須監控線上預測信心分布並定期重訓。
- **可解釋性**：建議搭配 Grad-CAM 等視覺化，協助 QA 理解模型關注區域，提升可信度。

## 6. 監控與維運（Monitoring）

- 線上即時指標：延遲 / QPS / 預測信心分布（Prometheus，`src/monitoring/metrics.py`）。
- 批次漂移：以預測機率與類別分布作為線上代理指標，餵入 Evidently（見 `monitoring/evidently/drift_config.yaml` 的 `vision` 區段）。
- 告警與重訓：同 tabular，閉環於 M5/M6。

## 7. AI Act 對應

依 `governance/ai_act_risk_assessment.md`，本模型用於工業品質檢測（非高風險清單上的安全元件認證），
屬**有限風險**等級，主要義務為透明度、人工監督與紀錄留存；若用於安全關鍵零件之合格性判定，
則須重新評估是否落入高風險並補齊對應義務。
