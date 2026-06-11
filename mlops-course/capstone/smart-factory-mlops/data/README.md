# data/ — 資料分層說明

> 本資料夾由 **DVC** 追蹤、**不進 Git**（只存 `.dvc` 指標檔與 `.gitkeep`）。
> 分層遵循「raw → interim → processed」單向流，每層只讀上一層、寫下一層，
> 確保管線可重現、可回溯。

---

## 四層用途

| 層 | 用途 | 範例內容 | 誰寫入 |
| :--- | :--- | :--- | :--- |
| `raw/` | 原始、不可變的落地資料 | `sensors.csv`、瑕疵影像、需求歷史 | DVC `prepare` 階段 |
| `interim/` | 清洗 / 對齊中繼結果 | 去重、補時間軸後的感測器 | `src/data` |
| `processed/` | 可直接訓練的特徵集 | `sensor_features.parquet` | `src/features` |
| `external/` | 第三方外部資料 | NASA C-MAPSS、MVTec AD | 手動 / 腳本下載 |

> **單向依賴鐵律**：`processed` 可讀 `interim`/`raw`，反之絕不允許。
> 不可變層（`raw`）一旦落地不得就地修改，要改就往下游另存。

---

## 玩具資料後援（smoke test / CI）

真實資料未就緒時，`src/data/loaders.py` 會自動後援到課程共用玩具資料
`../../datasets/toy_sensors.csv`（5 台機台 × 150 時點）。因此即使
`data/raw/` 為空，`python -m feature_repo.bootstrap_features`、單元測試與
notebooks 仍可跑通。正式跑時把真實 `sensors.csv` 放入 `data/raw/` 即可。

---

## 如何取得真實資料集

### 1. 結構化 + 時序：NASA C-MAPSS（預測性維護）

渦輪引擎 run-to-failure 感測時序，是 RUL / 故障預測的經典基準。

```bash
# 來源：NASA Prognostics Center of Excellence Data Repository
#   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
# 下載 "Turbofan Engine Degradation Simulation Data Set"（CMAPSSData.zip）
mkdir -p data/external/cmapss
unzip CMAPSSData.zip -d data/external/cmapss
# train_FD001.txt 等：unit_number、time_cycles、3 個 operational settings、21 個感測器
```

> 對齊本專案契約：把 `unit_number → machine_id`、`time_cycles` 轉為
> `event_timestamp`（以基準日 + cycles 小時數推算），即可餵入相同管線。

### 2. 影像：MVTec AD（產線視覺瑕疵檢測）

工業表面異常檢測標準資料集（good / 各類 defect）。

```bash
# 來源（須同意學術授權）：
#   https://www.mvtec.com/company/research/datasets/mvtec-ad
mkdir -p data/external/mvtec
tar -xf mvtec_anomaly_detection.tar.xz -C data/external/mvtec
# 目錄佈局：<class>/train/good/*.png、<class>/test/<defect>/*.png
```

> `src/data/loaders.load_images()` 預期 `<split>/<label>/<file>` 佈局，
> 將 MVTec 的 `test/good`、`test/<defect>` 直接掃成 manifest。

### 3. 時序：產能需求

無公開檔時，`loaders.load_demand()` 會由感測器資料聚合出每日筆數作為
需求 proxy，供 forecasting smoke test 使用；正式資料放
`data/raw/demand.csv`（欄位 `event_timestamp,demand`）並於 conf 指定。

---

## DVC 提示

```bash
dvc add data/raw/sensors.csv      # 追蹤大檔，產生 sensors.csv.dvc
git add data/raw/sensors.csv.dvc data/raw/.gitignore
dvc push                          # 推到遠端 storage
```
