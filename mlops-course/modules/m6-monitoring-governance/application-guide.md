# M6 應用指導手冊：監控、漂移偵測與治理

## 這個模組解決什麼工程問題

M6 解決的是模型上線後的維運與問責問題。模型不是部署完就結束；資料會變、服務會慢、使用情境會偏離原始假設，模型也可能開始傷害業務。

本模組的工程心法是：模型上線後必須被監控；監控結果必須能觸發行動；模型用途、限制、資料與風險必須被文件化。

## 哪邊應用

M6 適用在模型已進入或準備進入線上環境的階段。它處理的是部署後的長期維運：服務是否健康、資料是否變了、模型是否仍有效、出事時能不能追溯責任與依據。

常見應用：

- 線上模型監控：定期比較 reference 與 current data。
- 漂移告警：特徵分布偏移時通知或觸發重訓。
- 上線審查：用 Model Card 說清楚用途、限制、資料與風險。
- 合規與稽核：用風險分級檢查是否需要更多治理措施。

## 怎麼用

使用順序是先監控資料，再接行動，最後補治理文件：

1. 選定 reference data，記錄版本與期間。
2. 收集 current data，按時間窗口切分。
3. 用 Evidently 產生 drift report。
4. 檢查整體與逐欄 drift。
5. 將漂移結果接到 Prefect、CI 或告警系統。
6. 補系統層 metrics，例如 latency、error rate、QPS。
7. 填寫 Model Card，包含用途、限制、評估與維運條件。
8. 做 AI Act 風險分級自評。

## 本模組元件

| 元件 | 在 MLOps 裡的職責 | 本課使用方式 |
| :--- | :--- | :--- |
| Evidently | 資料漂移與資料品質報告 | reference vs current drift report |
| Reference / Current data | 監控比較基準 | 按時間切資料 |
| Drift | 分布或關係改變 | covariate、label、concept drift |
| Prometheus / Grafana | 系統層監控 | 延遲、錯誤率、QPS |
| Great Expectations | 資料品質驗證 | schema、值域、缺值 |
| Model Card | 模型治理文件 | 用途、限制、評估、維運 |
| EU AI Act | 風險分級與合規心智 | 風險自評模板 |

## 監控四層怎麼應用

模型監控不是只看 accuracy。實務上要分層看，因為每層回答的問題不同。

| 層 | 問題 | 指標例子 | 工具 |
| :--- | :--- | :--- | :--- |
| 系統 | 服務是否活著、是否變慢 | latency、error rate、QPS、CPU | Prometheus、Grafana |
| 資料品質 | 輸入資料是否壞掉 | 缺值、型別、欄位、值域 | Evidently、Great Expectations |
| 漂移 | 資料分布是否改變 | feature drift、prediction drift | Evidently |
| 業務 | 模型是否仍帶來價值 | 轉換率、誤判成本、真實 accuracy | 自訂指標、A/B test |

工程判斷：

- 系統層最快發現服務掛掉。
- 資料品質層最快發現資料管線壞掉。
- 漂移層發現模型可能即將失效。
- 業務層最接近真正價值，但通常最慢才知道。

## Evidently 怎麼應用

Evidently 用來比較 reference data 與 current data，產生資料漂移報告。

最小流程：

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)
report.save_html("drift_report.html")
```

工程應用場景：

- 每日檢查新進資料是否偏離訓練資料。
- 模型表現下降但尚未有 label 時，先用 feature drift 找線索。
- 將 drift 結果接到 Prefect，觸發告警或重訓。
- 在 PR 或資料更新流程中產生資料品質報告。

設計原則：

- reference 必須代表你認為正常的基準。
- current 必須代表最新上線流量或新資料。
- 時序資料應按時間切，不要隨機切。
- 不要只看整體 `dataset_drift`，也要看逐欄 drift。

## Reference Data 與 Current Data 怎麼選

Reference data 是比較基準；current data 是你要監控的新資料。

選擇建議：

| 資料 | 選擇方式 |
| :--- | :--- |
| reference | 訓練資料、驗證資料、或已確認穩定的歷史期間 |
| current | 最近一天、最近一週、或新版本服務收集的資料 |

常見錯誤：

- 用隨機切分做時序監控，導致 reference/current 都混在一起。
- 用太舊的 reference，讓正常季節性變化被誤判為漂移。
- 只用單一 reference，不考慮週期性或季節性。

工程建議：

- 時序資料優先按時間切。
- 有季節性時可建立多組 reference。
- reference 版本要和模型版本一起記錄。

## Drift 怎麼應用

Drift 表示資料或資料關係隨時間改變。

| 類型 | 什麼變了 | 能否立刻觀察 | 處理方式 |
| :--- | :--- | :--- | :--- |
| Covariate drift | `P(X)` 輸入特徵分布 | 通常可以 | 監控特徵分布 |
| Label drift | `P(y)` 標籤分布 | 需要 label 回流 | 監控 label 或 prediction |
| Concept drift | `P(y|X)` 關係 | 最難，需要真實結果 | 等 label 回流後評估 |

應用判斷：

- 特徵分布變了，不代表模型一定壞，但代表風險上升。
- 沒有 drift，不代表模型一定好，因為 concept 可能變了但 feature 分布沒變。
- 有 label 回流時，應同時監控真實 performance。

## Prometheus / Grafana 怎麼應用

Prometheus 與 Grafana 偏系統監控，回答服務是否穩定。

應用面：

- API latency 是否升高。
- error rate 是否增加。
- request volume 是否異常。
- CPU、memory、GPU 使用率是否接近瓶頸。

和模型監控分工：

| 問題 | 工具 |
| :--- | :--- |
| API 掛了或變慢 | Prometheus / Grafana |
| 輸入資料欄位壞掉 | Great Expectations / Evidently |
| 特徵分布變了 | Evidently |
| 模型業務效果變差 | 業務 metric / label feedback |

## Great Expectations 怎麼應用

Great Expectations 適合把資料品質規則寫成可執行規格。

應用面：

- 欄位必須存在。
- 欄位型別必須正確。
- 數值必須落在合理範圍。
- 缺值比例不能超過門檻。
- 類別值只能出現在允許清單中。

工程判斷：

- 如果你需要強 schema gate，Great Expectations 很適合。
- 如果你要快速做漂移報告，Evidently 更直接。
- 兩者可以一起用：Great Expectations 擋壞資料，Evidently 看分布變化。

## Model Card 怎麼應用

Model Card 是模型的工程交接與治理文件。

應包含內容：

| 區塊 | 要回答的問題 |
| :--- | :--- |
| 模型用途 | 這個模型被允許拿來做什麼 |
| 不適用範圍 | 哪些情境不能用 |
| 訓練資料 | 用了哪些資料、版本、期間 |
| 評估結果 | 指標、切分方式、限制 |
| 公平性與風險 | 哪些群體或場景可能出問題 |
| 維運條件 | 何時告警、重訓、下線 |
| 責任人 | 誰維護、誰審核 |

工程應用場景：

- 模型交接給服務團隊。
- 上線審查。
- 合規稽核。
- 事故回溯。

設計原則：

- 不要只寫好消息，也要寫限制。
- 資料版本、模型版本、評估日期要明確。
- 維運條件要可執行，例如 drift 超過門檻就告警。

## EU AI Act 風險分級怎麼應用

EU AI Act 提供一種風險分級心智。即使專案不直接受 EU 法規約束，工程團隊也可以用它檢查模型風險。

常見分級：

| 等級 | 意義 |
| :--- | :--- |
| 不可接受風險 | 原則上禁止 |
| 高風險 | 需要嚴格義務，例如資料治理、人為監督、記錄 |
| 有限風險 | 需要透明度與使用者告知 |
| 最小風險 | 一般低風險應用 |

工程應用：

- 在設計早期做風險自評。
- 對高風險用途建立更完整的資料、模型、監控紀錄。
- 將 Model Card 作為技術文件的一部分。
- 讓產品、法務、資料科學、工程團隊有共同語言。

## 最小落地流程

1. 選定 reference data，記錄其資料版本與期間。
2. 收集 current data，按時間窗口切分。
3. 用 Evidently 產生 drift report。
4. 檢查逐欄 drift，不只看整體旗標。
5. 將 drift 結果接到 Prefect 或 CI，作為告警或重訓條件。
6. 補上系統層 metrics，例如 latency、error rate。
7. 填 Model Card，記錄用途、限制、資料、評估與維運條件。
8. 做 AI Act 風險自評，判斷是否需要更嚴格治理。

## 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 只看離線 accuracy | 上線退化無法察覺 | 加資料與業務監控 |
| reference/current 切錯 | 監控一開始就說謊 | 按時間與業務週期切 |
| 只看整體 drift flag | 單一關鍵欄位漂移被忽略 | 檢查逐欄結果 |
| 沒有 label 回流設計 | concept drift 無法確認 | 設計真實結果回收流程 |
| Model Card 只寫形式 | 不能支援維運與稽核 | 寫清用途、限制、責任與行動條件 |
| 治理上線後才補 | 缺少可追溯紀錄 | 從資料與訓練階段開始記 |

## 工程驗收標準

- drift report 可重複產生。
- reference 與 current 切分邏輯明確。
- 逐欄 drift 結果能被讀取或人工檢查。
- 漂移結果有對應行動，例如告警、重訓、人工審查。
- Model Card 有用途、限制、資料、評估、維運條件。
- 風險分級有明確理由。

## 你應該帶走的能力

完成 M6 後，工程師應該能設計模型上線後的監控與治理流程：知道要看哪些訊號、如何判斷漂移、何時觸發行動，以及如何讓模型具備可交接與可稽核能力。
