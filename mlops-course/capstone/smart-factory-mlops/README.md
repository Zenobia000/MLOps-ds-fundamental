# Smart Factory MLOps — 工業級端到端範例

> **這是教學 repo 的 Layer 3 完整範例（capstone）。**
> 到 M6 才解鎖：此時你對每個工具都已親手用過，這裡把它們組成一條**生產級的端到端 pipeline**。
> 重點不再是「工具怎麼用」，而是「**怎麼設計、為什麼這樣組**」。

本專案以**智慧工廠**為主題，同一套 MLOps 骨架同時支撐三種資料型態，
證明一件事：**MLOps 是與模型型態無關的平台能力**。

---

## 1. 三個子場景（三資料型態並存）

| 子場景 | 資料型態 | 輸入 → 輸出 | 模型 | 落點 |
| :--- | :--- | :--- | :--- | :--- |
| **設備預測性維護** | 結構化 + 時序 | 感測器（temperature / vibration / current）→ 故障機率 | XGBoost | `src/models/tabular` |
| **產能需求預測** | 時序 | 歷史需求序列 → 未來 N 步需求 | LSTM / 簡單 forecaster | `src/models/timeseries` |
| **產線視覺瑕疵檢測** | 影像 | 產品影像 → good / defect | 預訓練 ResNet 微調 + ONNX | `src/models/vision` |

> 三者差異只收斂在 **data source / model / service runner** 三處；
> MLflow、Feast、BentoML、Evidently、CI/CD 這層**完全共用**。

---

## 2. 架構圖（文字）

```
                         ┌────────────────────────────────────────────┐
                         │              conf/ (config-driven)          │
                         │  config.yaml · data/ · model/ · train/ · hpo/│
                         └───────────────────────┬────────────────────┘
                                                 │ 載入 (src/utils/config.py)
   ┌──────────┐   DVC    ┌──────────┐   Feast   ┌──────────────┐   MLflow   ┌────────────┐
   │  data/   ├─────────▶│ src/data ├──────────▶│ src/features ├──────────▶│ src/training│
   │ raw/...  │  版本化  │ 清洗/驗證 │  特徵商店  │ point-in-time│  追蹤/調參 │  + tuning   │
   └──────────┘          └──────────┘           └──────────────┘            └─────┬──────┘
                                                                                  │ 註冊
                                                                                  ▼
   ┌────────────┐  Evidently  ┌──────────────┐   BentoML / FastAPI   ┌──────────────────┐
   │ monitoring │◀───────────┤  src/serving │◀─────────────────────┤  Model Registry  │
   │ 漂移/儀表板 │   閉環回饋   │  推論+前後處理 │      services/        │  (MLflow)        │
   └────────────┘            └──────────────┘                       └──────────────────┘
        ▲                                                                     ▲
        │                          GitHub Actions (CI / CD / CT)              │
        └──────────── ci.yml → train.yml → deploy.yml ──── pipelines/ (Prefect)┘
```

---

## 3. MLOps 生命週期對應

| 生命週期階段 | 本專案實作 | 模組 |
| :--- | :--- | :--- |
| 資料版本化 | DVC（`dvc.yaml` / `params.yaml`） | M2 |
| 特徵管理 | Feast（`feature_repo/`，point-in-time join 防洩漏） | M3 |
| 實驗追蹤 | MLflow tracking（`src/training`） | M2 |
| 自動調參 | Optuna（`src/tuning` + `conf/hpo`） | M2 |
| 模型註冊 | MLflow Model Registry | M2 |
| 服務化 | BentoML + FastAPI + ONNX（`services/`） | M4 |
| 編排 | Prefect（`pipelines/`） | M5 |
| CI/CD/CT | GitHub Actions（`.github/workflows`） | M5 |
| 資料/測試契約 | Great Expectations + pytest（`tests/`） | M5 |
| 監控 | Evidently 漂移 + Grafana（`monitoring/`） | M6 |
| 治理 | Model Card / Datasheet / AI Act 風險分級（`governance/`） | M6 |

---

## 4. 技術棧

| 領域 | 工具 |
| :--- | :--- |
| 語言 / 套件管理 | Python ≥ 3.10, `pyproject.toml` |
| 資料版本 | DVC |
| 特徵商店 | Feast |
| 實驗追蹤 / 註冊 | MLflow |
| 自動調參 | Optuna |
| 表格模型 | XGBoost, scikit-learn |
| 時序模型 | PyTorch LSTM |
| 影像模型 | PyTorch + torchvision ResNet → ONNX Runtime |
| 服務 | BentoML, FastAPI, Uvicorn |
| 編排 | Prefect |
| 資料驗證 | Great Expectations, Pydantic |
| 監控 | Evidently |
| 設定 | PyYAML（config-driven） |
| 品質 | Ruff, Black, pre-commit, detect-secrets, pytest |

---

## 5. 快速開始（Quickstart）

```bash
# 0) 安裝（建議虛擬環境）
pip install -e ".[dev]"

# 1) 設定環境變數（複製範本後填值；.env 不進 Git）
cp .env.example .env

# 2) 啟動本地平台（MLflow + Feast + 監控）
make up

# 3) 套用 Feast 特徵定義
make feast-apply

# 4) 訓練三種模型（任選；smoke 跑玩具資料）
make train-tabular     # XGBoost 預測維護
make train-ts          # LSTM 需求預測
make train-vision      # ResNet 瑕疵檢測 + ONNX 匯出

# 5) Optuna 自動調參
make tune

# 6) 端到端 DVC 管線（prepare→features→train→evaluate）
make pipeline

# 7) 啟動推論服務 / 跑漂移監控
make serve
make monitor

# 8) 測試與 lint
make test
make lint
```

> `make help` 列出所有指令。所有指令從 **repo 根**執行；套件根 = `src/`。
> Canonical import：`from src.utils.config import load_config`。

---

## 5b. 實測指令與預期輸出（已驗證）

> 以下為在乾淨環境實際執行各 `make` target 的輸出摘要（玩具資料 smoke）。
> MLflow 預設用本地 `sqlite:///mlflow.db`（免啟 server）；產物（mlruns/、models/、
> *.parquet、mlflow.db）皆 gitignore。**11/11 target 全數驗證通過。**

| 指令 | 結果 | 關鍵輸出 |
| :--- | :--- | :--- |
| `make train-tabular` | ✅ | `f1=0.737 roc_auc=0.919 aucpr=0.849`；品質門檻 **通過**；存 `models/tabular/model.xgb` |
| `make train-ts` | ✅ | LSTM；`rmse=0.708`（gate 用 rmse≤2.0）**通過**；存 `models/timeseries/lstm.pt` |
| `make train-vision` | ✅ | ResNet 微調 smoke + ONNX 匯出（opset 17）；存 `models/vision/model.onnx` |
| `make tune` | ✅ | Optuna 30 trials；`best_value≈0.874`（PR-AUC） |
| `make pipeline` | ✅ | `dvc repro` 跑完 prepare→features→train→evaluate；二次跑顯示 *up to date* |
| `make serve` | ✅ | `bentoml serve`；`POST /healthz` → 200、`POST /predict_maintenance` → 200 |
| `make monitor` | ✅ | Evidently 漂移報告 `dataset_drift=True`（2/3 特徵漂移） |
| `make feast-apply` | ✅ | 建立 entity / feature view / feature service |
| `make test` | ✅ | `26 passed` |
| `make lint` | ✅ | ruff + black 全綠 |
| `make up` | ✅* | `docker compose config` 驗證通過（*未實際啟動全棧） |

**`make train-tabular` 實際輸出（節錄）**
```
src.training.evaluate | 品質門檻：f1=0.7368 vs 門檻 0.7000（maximize）→ 通過
__main__ | metrics={'accuracy':0.867,'precision':0.778,'recall':0.70,
                    'f1':0.737,'roc_auc':0.919,'aucpr':0.849}
```

**`make serve` 實測（另一個終端機）**
```bash
curl -X POST http://localhost:3000/predict_maintenance -H 'Content-Type: application/json' \
  -d '{"request":{"readings":[{"machine_id":"machine_01","temperature":92,"vibration":5.2,"current":12.5},
                              {"machine_id":"machine_02","temperature":56,"vibration":2.0,"current":10.1}]}}'
# → HTTP 200
# {"model_name":"smart_factory_tabular","model_version":"local","threshold":0.5,
#  "predictions":[{"machine_id":"machine_01","failure_probability":0.994,"will_fail":true},
#                 {"machine_id":"machine_02","failure_probability":0.0001,"will_fail":false}]}
```

> 依賴：`make train-*` / `tune` 需 `pip install -e .`（含 mlflow/optuna/dvc/xgboost）；
> `train-vision` 另需 torch+torchvision+onnx+onnxscript；`serve` 需 bentoml+pillow；
> `monitor` 需 evidently；`pipeline` 需 dvc（`dvc init` 已完成）。

---

## 6. 資料夾導覽

```
smart-factory-mlops/
├── conf/            # 集中設定（config.yaml + data/ model/ train/ hpo/）── 本層核心
├── src/             # 正式原始碼（data/ features/ models/ training/ tuning/ serving/ monitoring/ utils/）
├── feature_repo/    # Feast 特徵商店（entity = machine_id）
├── pipelines/       # Prefect 編排 DAG
├── services/        # BentoML 服務（表格 + 影像雙模型一個 API）
├── tests/           # unit / integration / data 契約
├── monitoring/      # Evidently 漂移 + Grafana 儀表板
├── governance/      # Model Card / Datasheet / AI Act 風險分級
├── docker/          # 訓練/服務映像 + docker-compose（本地平台）
├── .github/         # CI / CD / CT workflows
├── data/            # DVC 追蹤（raw/interim/processed/external），不進 Git
├── models/          # 本地模型產物（DVC/registry 管理），不進 Git
├── dvc.yaml         # 管線：prepare→features→train→evaluate
├── params.yaml      # DVC 追蹤的超參
├── Makefile         # 指令封裝
└── pyproject.toml   # 依賴與工具設定
```

> 完整結構與模組對照見 [`../../docs/project-structure.md`](../../docs/project-structure.md)。

---

## 7. 設定契約（重要）

切資料集 / 模型只改 `conf/`，不改程式：

- `conf/config.yaml`：`project` · `seed` · `paths{...}` · `mlflow{...}` · `active_model`
- `conf/data/*.yaml`：各資料來源（entity = `machine_id`，時間 = `event_timestamp`）
- `conf/model/<name>.yaml`：`name` · `params{...}`
- `conf/hpo/<name>.yaml`：`n_trials` · `direction` · `pruner` · `search_space{...}`

> 真實資料集（NASA C-MAPSS / MVTec AD）放 `data/external/`，取得方式見 `data/README`。
> 重量級訓練（真實大資料上微調 ResNet）以 TODO 標註；骨架在小樣本/玩具資料上即可跑。
