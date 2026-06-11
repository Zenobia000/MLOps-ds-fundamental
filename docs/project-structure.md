# 課程專案資料夾架構（Smart Factory MLOps Capstone）

> **用途**：本文件規劃「智慧工廠 MLOps」capstone 專案的標準資料夾結構，作為學員 clone 即可開工的骨架，也是六大模組的程式落點。
> **設計原則**：關注點分離（探索 / 原始碼 / 設定 / 資料 / 管線分開）、config-driven、可重現、production-like。
> **狀態**：規劃中（尚未建立實體資料夾）。

---

## 1. 設計原則

| 原則 | 說明 |
| :--- | :--- |
| **notebooks ≠ src** | `notebooks/` 只做探索與教學示範；正式邏輯一律落 `src/`，可被測試、被 import、被 pipeline 呼叫。 |
| **config-driven** | 參數不寫死在程式裡，集中於 `conf/`（YAML / Hydra），切資料集與超參只改設定。 |
| **資料不進 Git** | `data/` 與 `models/` 用 **DVC** 追蹤，Git 只存指標檔；大檔走遠端 storage。 |
| **三資料型態並存** | 結構化 / 時序 / 影像 各自有 data source、feature、model、service，但共用同一套 MLOps 骨架。 |
| **每層對應模組** | 資料夾結構直接映射 M1–M6，學員照模組順序逐層填滿。 |

---

## 2. 完整資料夾樹

```
smart-factory-mlops/
├── README.md                      # 專案說明、快速開始
├── pyproject.toml                 # 依賴與專案 metadata（或 requirements.txt）
├── Makefile                       # 常用指令封裝：make train / serve / test / monitor
├── .env.example                   # 環境變數範本（不含密鑰）
├── .gitignore
├── .pre-commit-config.yaml        # lint / format / secret 掃描
│
├── .dvc/                          # DVC 設定 ............................ [M2]
├── dvc.yaml                       # 資料/訓練 pipeline 階段定義
├── params.yaml                    # DVC 追蹤的超參數
│
├── docker/ .................................................. [M4]
│   ├── Dockerfile.train           # 訓練映像
│   ├── Dockerfile.serve           # 服務映像
│   └── docker-compose.yml         # 本地起 MLflow + Feast + 服務 + 監控
│
├── .github/
│   └── workflows/ .......................................... [M5]
│       ├── ci.yml                 # lint → test → 資料驗證
│       ├── train.yml              # 訓練 → 品質門檻 → 註冊
│       └── deploy.yml             # build image → canary 部署
│
├── conf/                          # 集中設定 ........................... [全模組]
│   ├── config.yaml                # 根設定
│   ├── data/                      # 各資料集設定
│   ├── model/                     # xgboost.yaml / resnet.yaml / lstm.yaml
│   ├── train/                     # 訓練超參、切分策略
│   └── hpo/                       # Optuna 搜尋空間、trial 數、pruner、目標
│
├── data/                          # DVC 追蹤，不進 Git ................. [M2/M3]
│   ├── raw/                       # 原始：感測器 csv、瑕疵影像、需求歷史
│   ├── interim/                   # 清洗中繼
│   ├── processed/                 # 可訓練特徵集
│   └── external/                  # 外部資料（C-MAPSS / MVTec AD）
│
├── feature_repo/                  # Feast 特徵商店 .................... [M3]
│   ├── feature_store.yaml         # 離線/線上 store 設定
│   ├── entities.py                # machine_id 等實體
│   ├── data_sources.py            # 感測器時序來源
│   └── features.py                # feature view（滾動均值、振動 std…）
│
├── notebooks/                     # 僅探索與教學 ...................... [全模組]
│   ├── 01_eda_sensor.ipynb
│   ├── 02_eda_defect_images.ipynb
│   └── 03_pointintime_leakage_demo.ipynb
│
├── src/                           # 正式原始碼（可測試、可 import）
│   ├── data/                      # 載入、清洗、驗證 ................. [M2]
│   ├── features/                  # 特徵工程、Feast 介接 ............ [M3]
│   ├── models/
│   │   ├── tabular/               # XGBoost 預測維護 ............... [M2]
│   │   ├── timeseries/            # LSTM/Prophet 需求預測 .......... [M3]
│   │   └── vision/               # PyTorch ResNet 瑕疵檢測 ........ [M4]
│   ├── training/                  # 訓練迴圈 + MLflow logging ....... [M2/M4]
│   ├── tuning/                    # Optuna objective + study + pruner [M2]
│   ├── serving/                   # 推論邏輯、前後處理 .............. [M4]
│   ├── monitoring/                # 漂移偵測、指標收集 .............. [M6]
│   └── utils/                     # 共用工具、logging、seed
│
├── pipelines/                     # 編排 DAG（ZenML/Prefect）......... [M5]
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── deployment_pipeline.py
│
├── services/                      # BentoML 服務定義 ................. [M4]
│   ├── service.py                 # 表格 + 影像雙模型一個 API
│   ├── bentofile.yaml
│   └── runners.py
│
├── tests/ .................................................. [M5]
│   ├── unit/                      # 函式級
│   ├── integration/               # pipeline / API 級
│   └── data/                      # Great Expectations 資料契約
│
├── monitoring/                    # 監控設定 ......................... [M6]
│   ├── evidently/                 # 漂移報告設定
│   └── grafana/                   # dashboard JSON + Prometheus 規則
│
├── governance/                    # 治理與合規 ....................... [M6]
│   ├── model_cards/               # 各模型 Model Card
│   ├── datasheets/                # 資料說明書
│   └── ai_act_risk_assessment.md  # EU AI Act 風險分級
│
├── models/                        # 本地產物，DVC/registry 管理（gitignore）
│
├── infra/                         # IaC（選配，進階）................. [M5]
│   └── terraform/
│
└── docs/                          # 課程教材（依模組）
    ├── M1_overview/
    ├── M2_experiment_tracking/
    ├── M3_feature_store/
    ├── M4_serving/
    ├── M5_cicd/
    └── M6_monitoring/
```

---

## 3. 資料夾 ↔ 模組對照

| 模組 | 主要落點資料夾 | 學員產出 |
| :--- | :--- | :--- |
| **M1 全景** | `README`, `Makefile`, `conf/`, `docker-compose` | 環境跑起來、成熟度 gap 表 |
| **M2 實驗追蹤＋調參** | `src/data`, `src/models/tabular`, `src/training`, `src/tuning`, `conf/hpo`, `.dvc`, `dvc.yaml` | XGBoost run + **Optuna 自動調參** + DVC 資料版本 + registry |
| **M3 特徵商店** | `feature_repo/`, `src/features`, `pipelines/feature_pipeline` | Feast feature view + point-in-time 訓練集 |
| **M4 服務化** | `src/models/vision`, `services/`, `docker/` | PyTorch 瑕疵模型 + BentoML API + image |
| **M5 CI/CD/CT** | `.github/workflows`, `pipelines/`, `tests/` | 自動化管線 + 品質門檻 + canary |
| **M6 監控治理** | `monitoring/`, `governance/`, `src/monitoring` | 漂移報告 + dashboard + Model Card |

---

## 4. 三資料型態的並存策略

同一套骨架支撐三型態，差異只在三個地方收斂，其餘共用：

| 型態 | data source | model | service runner |
| :--- | :--- | :--- | :--- |
| 結構化（預測維護） | `data/raw/sensors.csv` | `src/models/tabular` | XGBoost runner |
| 時序（需求預測） | Feast time-series view | `src/models/timeseries` | LSTM/Prophet runner |
| 影像（瑕疵檢測） | `data/raw/images/` | `src/models/vision` | PyTorch runner |

> 教學重點：**MLflow、Feast、BentoML、Evidently、CI/CD 這層完全共用**——讓學員體會「MLOps 是與模型型態無關的平台能力」。

---

## 5. 建置順序建議

依模組順序由外而內填滿，每階段都保持「可跑」：

1. **M1**：先建骨架 + `docker-compose` + `Makefile`，`make up` 能起 MLflow。
2. **M2**：填 `src/data` + `src/models/tabular` + DVC，`make train` 能產 run。
3. **M3**：填 `feature_repo/` + `src/features`，特徵管線可跑。
4. **M4**：填 `src/models/vision` + `services/`，`make serve` 起 API。
5. **M5**：填 `.github/workflows` + `pipelines/` + `tests/`，push 觸發自動化。
6. **M6**：填 `monitoring/` + `governance/`，閉環完成。

> 詳細工作分解、工時與相依見 [development-wbs.md](./development-wbs.md)。
