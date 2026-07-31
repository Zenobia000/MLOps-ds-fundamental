# M6 — 監控與治理（Monitoring & Governance）

> 對應技能階梯 **階 11（Evidently）+ 治理 + 收尾**。
> 這是最後一個工具模組：學會「模型上線後怎麼盯著它」與「怎麼讓它合規可問責」，
> 然後**解鎖 capstone** 做完整端到端整合。

---

## 1. 本模組學什麼

學會在模型「上線之後」持續監控它：用 **Evidently** 比較參考分布與當前分布、偵測**資料漂移**，
並補上**治理文件**（Model Card、EU AI Act 風險分級），讓系統可問責、可稽核（技能階梯階 11）。
跑完本模組你會**解鎖 capstone**——因為到這裡，你對每一個工具都已有獨立的成功經驗。

### 1.1 監控四層（由下而上，問題越上面越貴）

| 層 | 看什麼 | 範例訊號 | 典型工具 |
| :--- | :--- | :--- | :--- |
| ① 系統 | 服務活著嗎、快不快 | 延遲、錯誤率、QPS、CPU/記憶體 | Prometheus / Grafana |
| ② 資料品質 | 進來的資料壞了嗎 | 缺值、型別錯、超出範圍、schema 變動 | Evidently / Great Expectations |
| ③ 漂移 | 資料/關係變了嗎 | 特徵分布平移、預測分布改變 | **Evidently（本模組）** |
| ④ 業務 | 模型還有用嗎 | 準確率、誤判成本、轉換率 | 自訂指標 + A/B |

> 心法：**越下層越即時、越好抓；越上層越貴、越延遲、越接近「真正重要」**。
> 系統層 5 分鐘就會吵你；業務層可能要兩週才看得出模型爛了。四層都要，但要知道各自的角色。

### 1.2 漂移的三種類型（別搞混）

| 類型 | 定義 | 工廠例子 | 本模組能抓到嗎 |
| :--- | :--- | :--- | :--- |
| **covariate drift** | 輸入特徵分布 P(X) 變了 | 感測器老化，溫度讀數整體偏移 | ✅ DataDriftPreset 直接抓 |
| **label drift** | 標籤分布 P(y) 變了 | 不良率本身上升 | ✅ 對預測/標籤欄做 drift |
| **concept drift** | X→y 的關係 P(y\|X) 變了 | 同樣溫度，以前正常現在會壞 | ⚠️ 需有真實標籤回流才驗證得出 |

> 沙盒裡我們「人工平移某一欄」= 最典型的 **covariate drift**。
> concept drift 最難抓——因為你需要等真實結果回來，才知道「關係」是否變了。

### 1.3 可觀測性（Observability） vs 監控（Monitoring）

- **監控**：盯著**你預先想到的**指標（漂移率、延遲）→ 回答「**有沒有出事**」。
- **可觀測性**：系統對外暴露足夠的訊號（logs / metrics / traces），讓你能**事後問沒預想過的問題**
  → 回答「**為什麼出事、出在哪**」。
- 一句話：**監控是已知問題的儀表板；可觀測性是未知問題的偵查能力。** 兩者互補，不是二選一。

### 1.4 治理（Governance）：讓模型可問責

- **EU AI Act 風險分級**：以風險為基礎的監管——不可接受 / 高 / 有限 / 最小，**風險越高義務越多**。
  自評模板見 `sandbox/governance/ai_act_risk_assessment_template.md`。
- **Model Card**：模型的「身分證 + 使用說明書」——用途、限制、訓練資料、公平性、適用邊界、維運聯絡。
  模板見 `sandbox/governance/model_card_template.md`。
- 治理不是上線後才補的文件，而是**從訓練資料就開始記錄**的可追溯鏈。

---

## 2. 沙盒步驟（Layer 1：照編號逐個跑，只學最小可用動詞）

> **指令起點**：本檔所有路徑都相對 **`mlops-course/`**（放 `Makefile` 的那一層）。
> `cd` 進沙盒跑完，記得 `cd` 回 `mlops-course/` 再接下一段。
> Git 是例外——repo 根在再上一層，詳見[課程 README「指令從哪裡下」](../../README.md#指令從哪裡下)。

> 一次一概念。每個檔都能獨立執行，彼此不 import。

### 2.1 Evidently 漂移報告

```bash
# 先確認套件（course 通常已統一裝好）
pip install "evidently>=0.4,<0.5" pandas numpy scikit-learn matplotlib

cd mlops-course/modules/m6-monitoring-governance/sandbox/evidently   # 從 repo 根出發

jupyter lab drift_explore.ipynb   # 人在看：調參數、決定監控哪幾欄、找出門檻
python drift_report.py            # 機器在跑：把結論固定成每天自動執行的檢查

cd ../../../..                    # ★ 回到 mlops-course/（evidently → sandbox → m6-… → modules）
```

**本模組是全課唯一兩種介質並存的地方**，因為監控天生同時需要「人的判斷」與「機器的重複執行」：

| | `drift_explore.ipynb` | `drift_report.py` |
| :--- | :--- | :--- |
| 誰在用 | **人**，互動式 | **機器**，無人值守 |
| 目的 | 決定「監控什麼、門檻多少」 | 執行既定檢查、輸出結論 |
| 產出 | 理解 | 報告檔 + 可當 CI 門檻的結果 |

> 一句話：**你在 notebook 想清楚要監控什麼，然後寫成腳本讓機器每天替你看。**
> 判準見 [`docs/notebook-vs-script.md`](../../../docs/notebook-vs-script.md)。

`drift_report.py` 做的事：
1. 讀 `datasets/toy_sensors.csv`，**按時間**切成 **reference / current** 兩份 DataFrame；
2. **人工注入漂移**——把某一欄整體平移（模擬感測器老化 = covariate drift）；
3. 用 **`DataDriftPreset`** 產生 `drift_report.html`；
4. 在終端機印出**逐欄**漂移結果與**整體** `dataset_drift`。

> 你只要掌握這一個動詞：**ref vs current → DataDriftPreset → 報告**。
> 用瀏覽器打開 `drift_report.html`，對照看「哪一欄的分布被推開了」。
> （`datasets/toy_sensors.csv` 若還沒就位，腳本會自動產生等價玩具資料，照樣能跑。）

#### 兩個一定要看懂的坑（notebook 裡有完整實驗）

**坑 1：reference/current 一定要按「時間」切，不能按列序切。**
`toy_sensors.csv` 是「一台機器接一台機器」排序的，照列序對半切等於**照機台切**——
而各機台的平均溫度差了好幾度（55.9 ~ 63.6）。那樣切出來，**還沒注入任何漂移就會報 drift**，
監控從第一天就在說謊。切對之後基線是乾淨的 0 欄漂移，注入才會亮。

**坑 2：整體 `dataset_drift` 是 `False` 不代表沒事。**
它預設要**超過半數欄位**都漂移才亮燈（`drift_share = 0.5`）。推一欄 → 1/3 沒過門檻 → 整體仍是 `False`。
**只看整體旗標會漏掉單欄漂移**——生產環境請以逐欄結果為準，或依風險承受度調低 `drift_share`。
一個關鍵欄位漂掉，往往就足以讓模型失效。

### 2.2 治理模板（填寫，不需執行）

| 檔案 | 做什麼 |
| :--- | :--- |
| `sandbox/governance/model_card_template.md` | 複製後逐欄填，產出你模型的 Model Card |
| `sandbox/governance/ai_act_risk_assessment_template.md` | 先做分級判定，再對照級別完成義務清單 |

> 練習：拿你在 M2–M4 訓練/服務化的模型，實際填一份 Model Card，並做一次 AI Act 自評。

---

## 3. 整合任務（Layer 2 → 解鎖 Layer 3）

先解鎖填空模板，再接到主線、**解鎖 capstone**：

```bash
make workspace-m6
# 新增 monitoring/drift_report.py、monitoring/model_card_template.md
```

搜尋 `TODO(M6-`（對照 `sandbox/evidently/drift_report.py`）：

- [ ] **TODO(M6-1～3)**：reference vs current → `DataDriftPreset` → HTML 報告
- [ ] **TODO(M6-4)**：漂移接進 M5 Prefect（重訓或告警）
- [ ] **TODO(M6-5～7)**：填完 `model_card_template.md`（含下線/重訓條件 + AI Act 自評）
- [ ] **🔓 解鎖 capstone**：打開 `capstone/smart-factory-mlops/`，自行組端到端 pipeline

> 此時重點是「**設計與權衡**」——門檻怎麼定、哪些要人為複核。
---

## 4. 卡住怎麼辦

- 想回到乾淨起點：用上一個模組的快照重置 `workspace/`
  ```bash
  cp -r checkpoints/after-m5/* workspace/
  ```
- 沙盒範例**本身就是可跑的正解**：`drift_report.py` 照打、改平移量、換欄位，觀察 `dataset_drift` 怎麼變。
- Evidently 版本差異：本沙盒已同時相容 0.4/0.5 舊版與 0.6+ 新版 API；若 import 報錯，先確認 `pip show evidently` 的版本，再用對應的 `pip install "evidently>=0.4"`。
- `toy_sensors.csv` 找不到：腳本會自動 fallback 產生玩具資料，先跑通流程不卡關。

---

## 5. 檢核題（自我確認）

1. 監控四層是哪四層？哪一層「最即時但最不重要」、哪一層「最延遲但最接近業務」？
2. covariate / label / concept drift 各是什麼變了？沙盒裡「人工平移某欄」屬於哪一種？
3. 為什麼 **concept drift 最難偵測**？（提示：你需要等什麼回來？）
4. 「監控」和「可觀測性」差在哪？各自回答什麼問題？
5. EU AI Act 把系統分成哪四級？你 `workspace/` 的工廠瑕疵模型大概落在哪一級、為什麼？
