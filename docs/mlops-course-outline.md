# MLOps 入門完整課程大綱（12 小時）

> **版本**：v1 · 2026 視角
> **定位**：以傳統 ML 三大資料型態（結構化 / 時序 / 影像）打穩 MLOps 骨幹，並點出 LLMOps 是同一套原則的延伸。
> **與本 repo 的關係**：以現有的 `MLflow/`（實驗追蹤、registry、evaluation）與 `MLOps/feast/`（特徵商店）為「紅線」往外擴，學員可直接接續現有程式碼。

---

## 0. 課程設定（Assumptions）

| 項目 | 設定 |
| :--- | :--- |
| **對象** | 會 Python、懂基本 ML（能 train/eval 一個模型），但沒做過 production 部署的資料科學家／後端工程師。 |
| **目標** | 從「能 train 模型」進化到「能把模型穩定送上線並持續維運」。 |
| **形式** | 6 模組 × 2 小時，理論 40% / 動手 60%。 |
| **總時數** | 12 小時。 |

> 分級調整見 [§6 評量與分級](#6-評量與分級建議)。

---

## 1. 2026 年的 MLOps：教什麼才不過時

設計大綱前先定錨「2026 視角」與 2022 年 MLOps 課的差異，這決定了內容取捨。

| 維度 | 2022 主流 | **2026 現況（本課採用）** |
| :--- | :--- | :--- |
| 範疇 | 傳統 ML 模型生命週期 | 傳統 ML **+ LLMOps/GenAIOps 融合**（RAG、eval、prompt、vector DB） |
| 核心痛點 | 「模型上不了線」 | 「模型上線後**治理不了、成本爆掉、漂移沒人管**」——重心從 deploy → **operate** |
| 實驗追蹤 | MLflow 1.x / W&B | **MLflow 3.x**（內建 GenAI tracing/eval）、W&B |
| 服務化 | Flask/FastAPI 包一包 | **BentoML / KServe / Triton / Ray Serve**；LLM 用 vLLM；ONNX + 量化 |
| 監控 | 看 accuracy | **資料漂移 + 概念漂移 + 線上品質**（Evidently / Arize / WhyLabs） |
| 治理 | 加分項 | **EU AI Act 已上路（2025–2026 強制）**→ model card、lineage、bias 變必選 |
| 成本 | 少談 | **GPU 利用率、推論優化、batching、量化**是 KPI |
| 資料層 | CSV / 資料庫 | **Lakehouse（Iceberg/Delta）+ Feature Platform** |

---

## 2. 貫穿全課的實務案例：「智慧工廠 Smart Factory」

選一條紅線比零散範例更能教會學員「整條 pipeline 怎麼接」。選**智慧製造**，因為它**天然同時涵蓋三種資料型態**：

| 子場景 | 資料型態 | 模型 | 對應本 repo |
| :--- | :--- | :--- | :--- |
| 🔧 **設備預測性維護**（預測機台何時故障） | **結構化 + 時序**（溫度/振動/電流感測器） | XGBoost / LSTM | 延伸 Feast 範式 |
| 👁️ **產線視覺瑕疵檢測**（NG/OK 判定） | **影像** | **PyTorch 預訓練 ResNet50 / ViT 微調** | 新增 |
| 📦 **產能需求預測** | **時序** | Prophet / Temporal Fusion Transformer | 新增 |

**公開資料集**：NASA C-MAPSS（預測維護）、MVTec AD（瑕疵檢測）、Kaggle Predictive Maintenance。

> 替代產業（同樣涵蓋三型態）：**零售**（需求預測＋商品圖像標記＋客戶流失）、**金融**（信用風險＋詐欺時序＋單據影像辨識）。

---

## 3. 12 小時課程地圖

```
M1 ── MLOps 全景 + 成熟度模型 (2026)            [2h] 建立心智模型
M2 ── 實驗管理 + 可重現性 (MLflow + DVC)        [2h] 結構化案例｜接現有 MLflow
M3 ── 特徵工程 + 特徵商店 (Feast) + 訓練管線     [2h] 時序案例｜接現有 Feast
M4 ── 模型打包 + 服務化 (Docker/BentoML/Triton)  [2h] 影像案例｜PyTorch CV
M5 ── CI/CD/CT 自動化 + 編排 (Actions/ZenML)     [2h] 串成一條自動管線
M6 ── 監控 + 漂移偵測 + 治理 + Capstone          [2h] operate 思維 + AI Act + 收尾
```

**設計原則**：每個模組都把上一模組的產出當輸入——M2 訓練的模型在 M4 被打包、M5 被自動化、M6 被監控。學員最後手上是**一條真的能跑的端到端 pipeline**，不是六個孤立 demo。

---

## 4. 模組逐節大綱

### Module 1 — MLOps 全景與成熟度模型（2026 視角）｜2h

**學習目標**：能說清楚「為什麼 ML 系統不能只用傳統 SWE 那套」，並定位團隊現在的成熟度。

**核心觀念（~70 min）**
- ML 系統 vs 傳統軟體：**程式碼 + 資料 + 模型三者皆會變**，且資料會自己腐壞。
- 經典痛點：*Hidden Technical Debt in ML Systems*（Google）——膠水程式、pipeline 叢林、**訓練/服務偏差（training-serving skew）**。
- **MLOps 成熟度模型**：Google Level 0/1/2、Microsoft 五級——讓學員自評。
- 名詞地圖：DataOps / MLOps / **LLMOps** / ModelOps 的分工。
- **2026 現代技術棧地圖**（一張圖把全課工具定位）。
- 角色分工：Data Scientist、ML Engineer、MLOps/Platform Engineer。

**動手 Lab（~50 min）**
1. 環境建置：`uv`/`conda` + Docker + repo 結構（cookiecutter-data-science 風格）。
2. 導覽智慧工廠案例與三個資料集。
3. 對本 repo 做「成熟度健檢」：目前在 Level 幾？缺哪幾塊？（產出 gap 表當作全課 roadmap）。

**對應趨勢**：建立「operate > deploy」與「治理內建」的世界觀。

---

### Module 2 — 實驗管理、自動調參與可重現性｜2h ｜🔧 結構化案例

**學習目標**：任何一次實驗都能被別人一鍵重現，並用**自動調參**在大量 run 中系統性找到最佳模型。

**核心觀念（~45 min）**
- 可重現性要素：**程式碼版本（Git）+ 資料版本（DVC）+ 環境版本（Docker/lockfile）+ 隨機種子**。
- 實驗追蹤該記什麼：params / metrics / artifacts / **model signature** / tags / lineage。
- **超參搜尋演進**：手動 → grid/random → **貝氏最佳化（Optuna TPE）** → **pruning（ASHA/Hyperband）提早砍掉爛 trial** → **多目標（accuracy vs latency vs cost）**。
- **HPO × 追蹤的關係**：每個 trial = 一個 MLflow run；Optuna 負責「探索」、MLflow 負責「記錄與比較」。AutoML（FLAML/AutoGluon）= HPO + 模型選擇的自動化封裝（對應你的 `autoML_template`）。
- Model Registry 與生命週期：Staging → Production → Archived，alias 與版本治理。
- 訓練/服務偏差怎麼來、怎麼防。

**動手 Lab（~75 min）**——擴充現有 `MLflow/basic` 與 `autoML_template`
1. 用預測性維護結構化資料訓練 XGBoost，**MLflow 記錄 params/metrics/signature**（接 `09`、`10`、`13` 號 notebook）。
2. **用 Optuna 自動調參**：寫 `objective(trial)` + `trial.suggest_*`，`study.optimize` 跑數十 trial，**每個 trial 自動寫成一個 MLflow run**；用 MLflow UI 比較收斂（接 `07`、`08`）。
3. 開啟 **pruning**（ASHA）觀察提早終止對時間的節省；（選配）示範**多目標**：同時優化 AUC 與推論延遲。
4. **加上 DVC**：把資料集與 `.pkl` 納入版本控制（同一 commit 一定拉到同一份資料）。
5. 把最佳 trial 的模型註冊進 **Model Registry** 並打 `@champion` alias。

**工具**：MLflow 3.x、**Optuna**、DVC、XGBoost、Git（進階：Ray Tune 分散式、FLAML/AutoGluon AutoML）
**對應趨勢**：自動調參成標配、多目標 HPO 呼應成本/延遲 KPI、資料版本化為治理鋪路。

---

### Module 3 — 特徵工程與特徵商店 + 訓練管線｜2h ｜📦 時序案例

**學習目標**：理解 feature store 解決的真實問題（**point-in-time 正確性**與訓練/服務一致），並把特徵化抽離成可重用管線。

**核心觀念（~45 min）**
- 為什麼需要 Feature Store：離線/線上一致、**避免時間穿越（data leakage）**、特徵重用與共享。
- **Point-in-time join**：時序資料最致命的雷，用感測器資料現場演示「未來資訊洩漏」。
- 離線 store（訓練）vs 線上 store（低延遲推論）。
- 把「資料前處理」升級成「**特徵管線**」：可參數化、可排程、可測試。

**動手 Lab（~75 min）**——擴充現有 `MLOps/feat_test`
1. 用 Feast 定義設備感測器的 **time-series feature view**（滾動均值、振動 std 等）。
2. `get_historical_features` 做 point-in-time 正確的訓練集（對照「錯誤做法」看指標虛高）。
3. `materialize` 到線上 store，`get_online_features` 模擬即時推論。
4. 把「特徵→訓練→註冊」包成可重跑的 Python pipeline script（為 M5 自動化鋪路）。

**工具**：Feast、Pandas/Polars、（選配 Parquet + DuckDB）
**對應趨勢**：Feature Platform 化、data-centric AI。

---

### Module 4 — 模型打包與服務化｜2h ｜👁️ PyTorch 影像案例

**學習目標**：把模型從 notebook 變成「別人能用 HTTP 呼叫的服務」，並理解 CPU 表格模型 vs GPU 深度模型的服務差異。

**核心觀念（~45 min）**
- 打包格式：pickle / **ONNX** / TorchScript / **safetensors**——為何 production 不直接送 `.pt`。
- 服務模式：線上即時 vs 批次 vs 串流；同步 vs 非同步。
- **服務框架選型（2026）**：FastAPI（輕量）/ **BentoML**（ML 原生）/ **NVIDIA Triton**（多框架高吞吐）/ KServe（K8s 原生）。
- GPU 服務關鍵字：**dynamic batching、ONNX/量化（INT8）、模型暖機、並發**。

**動手 Lab（~75 min）**——新增 PyTorch CV
1. 載入 **torchvision 預訓練 ResNet50**，在 MVTec 瑕疵資料上微調最後幾層（transfer learning）。
2. 用 MLflow 記錄這個 PyTorch 模型（與 M2 表格模型同一套追蹤，凸顯「同一套 MLOps 管不同模型」）。
3. **匯出 ONNX + 量化**，量測延遲/體積差異。
4. 用 **BentoML** 把「瑕疵檢測 + 預測維護」兩模型包成一個 service、build Docker image、`curl` 打 API。
5. （Demo）同一份模型丟上 **Triton**，看 dynamic batching 吞吐差異。

**工具**：PyTorch / torchvision、ONNX Runtime、BentoML、Docker、（Demo: Triton）
**對應趨勢**：推論優化與成本意識、多框架統一服務。

---

### Module 5 — CI/CD/CT 自動化與管線編排｜2h

**學習目標**：把前四模組的手動步驟串成「**push 程式碼 → 自動測試 → 自動訓練 → 自動部署**」，並理解 ML 特有的 **CT（持續訓練）**。

**核心觀念（~50 min）**
- CI/CD **vs CT**：ML 多了「資料變了要重訓」這條觸發路徑。
- ML 專屬測試：資料驗證（schema/分布）、**模型行為測試**（不變性測試）、模型品質門檻（接 `15_models_validation_threshold`）。
- **編排器選型（2026）**：Airflow（成熟）/ **ZenML**、**Metaflow**、Prefect（輕量 pythonic）/ Kubeflow Pipelines（K8s 重裝）。
- 部署策略：Shadow / Canary / Blue-Green / A-B test。
- IaC 概念（Terraform）讓環境可複製。

**動手 Lab（~70 min）**
1. 寫 **GitHub Actions**：lint → 單元測試 → 資料驗證 → 訓練 → **品質門檻 gate（沒過不准註冊）**。
2. 用 **ZenML 或 Prefect** 把 M3 特徵管線 + M2 訓練 + 註冊 編成一個 DAG。
3. 設一條 **CT 觸發**：偵測到漂移（M6 訊號）→ 自動觸發重訓 pipeline。
4. 模擬 **Canary 部署**：5% 流量導到新模型。

**工具**：GitHub Actions、ZenML/Prefect、Great Expectations、（概念: Terraform）
**對應趨勢**：CT 與 pipeline-as-product、平台工程思維。

---

### Module 6 — 監控、漂移偵測、治理與 Capstone｜2h

**學習目標**：建立「上線才是開始」的維運心智，並滿足 2026 的合規要求。

**核心觀念（~50 min）**
- 監控四層：**系統（延遲/QPS）→ 資料品質 → 資料/概念漂移 → 業務指標**。
- **漂移類型**：covariate shift / label shift / concept drift——感測器隨季節漂移現場演示。
- 標籤延遲問題：故障要等真的壞掉才有 ground truth，怎麼辦（proxy metric）。
- **可觀測性 vs 監控**：能回答「為什麼這筆預測錯了」（lineage + feature 回溯）。
- **治理與合規（2026 重點）**：**EU AI Act** 風險分級、**Model Card**、Data Sheet、bias/fairness 稽核、模型 lineage 與審計軌跡。
- 一頁帶過 **LLMOps 延伸**：同一套原則如何套到 RAG（eval、guardrails、prompt 版本、token 成本監控）。

**動手 Lab（~70 min）**
1. 用 **Evidently AI** 對時序感測器做漂移報告（注入人工漂移看它抓不抓得到）。
2. 接上 **Prometheus + Grafana**（或 Evidently dashboard）看線上指標。
3. 串回 M5：**漂移告警 → 觸發 CT 重訓**，閉環完成。
4. 產出 **Model Card**（用途、限制、訓練資料、公平性、適用邊界）。

**Capstone（整合收尾）**
- 交付一條**端到端 pipeline**：資料版本化 → Feast 特徵 → MLflow 訓練註冊 → BentoML 服務 → GitHub Actions 自動化 → Evidently 監控 → Model Card。
- 智慧工廠三案例**任選一條**完整貫通，其餘兩條口頭說明差異。

**工具**：Evidently AI、Prometheus/Grafana、（治理: MLflow lineage + Model Card 模板）

---

## 5. 技術棧總表（學員的「2026 工具帶」）

| 階段 | 主工具 | 備選 |
| :--- | :--- | :--- |
| 版本控制 | Git + **DVC** | LakeFS、Delta Lake |
| 實驗追蹤 | **MLflow 3.x** | W&B、Neptune |
| 自動調參 / AutoML | **Optuna** | Ray Tune、W&B Sweeps、FLAML、AutoGluon、H2O |
| 特徵商店 | **Feast** | Tecton、Featureform |
| 編排 | **ZenML / Prefect** | Airflow、Kubeflow、Metaflow |
| 服務化 | **BentoML** | Triton、KServe、Ray Serve、FastAPI |
| CI/CD | **GitHub Actions** | GitLab CI |
| 監控 | **Evidently AI** | Arize、WhyLabs、Prometheus/Grafana |
| 容器/編排 | **Docker** | Kubernetes（進階） |
| 治理 | Model Card + MLflow lineage | 自建 / Collibra |

---

## 6. 評量與分級建議

**評量**
- **形成性**：每模組一個可驗收的 Lab 產出（commit / artifact / API 截圖）。
- **總結性**：Capstone（佔比最高）+ 一份「團隊 MLOps 成熟度提升 roadmap」。

**分級調整**

| 班型 | 調整方向 |
| :--- | :--- |
| 純資料科學家班 | 加重 M2/M3，M5 編排簡化為概念。 |
| 平台/後端工程師班 | 加重 M4/M5，補 Kubernetes。 |
| 12h 太緊 | Triton、Terraform、Grafana 降為 Demo。 |
| 擴成 16–18h | 「LLMOps 延伸」獨立成一個完整模組。 |

---

## 附錄 A：環境需求

- Python 3.11+、Docker、Git
- （CV 模組）建議一張 GPU；無 GPU 可改用較小模型或 CPU + 量化推論
- 套件：`mlflow`, `optuna`, `dvc`, `feast`, `xgboost`, `torch`, `torchvision`, `onnxruntime`, `bentoml`, `evidently`, `prefect`/`zenml`, `great-expectations`（進階：`ray[tune]`, `flaml`）

## 附錄 B：延伸閱讀

- *Hidden Technical Debt in Machine Learning Systems* — Sculley et al., Google
- *Machine Learning Design Patterns* — Lakshmanan et al.
- *Designing Machine Learning Systems* — Chip Huyen
- Google Cloud：*MLOps: Continuous delivery and automation pipelines in ML*
- EU AI Act 官方文本與風險分級指引
