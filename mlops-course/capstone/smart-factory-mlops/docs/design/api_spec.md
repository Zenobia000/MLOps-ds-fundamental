# API 規格 — Smart Factory 推論服務

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **語域:** L3（技術契約）
>
> **定位**：推論服務的對外契約。機器可讀版本見 [`openapi.yaml`](openapi.yaml)。
> 契約的**權威來源是程式碼**：`src/serving/schemas.py`（Pydantic）與 `services/service.py`（端點）。
> 本文件是它們的說明，兩者不一致時以程式碼為準並回頭修本文件。
> **實例:** 每服務一份（本專案只有一個服務 `smart_factory`）

## 目錄

- [1. 服務概觀](#1-服務概觀)
- [2. 端點](#2-端點)
- [3. 共用約定](#3-共用約定)
- [4. 錯誤處理](#4-錯誤處理)
- [5. 模型版本語意](#5-模型版本語意)
- [6. 契約對齊規則](#6-契約對齊規則)

---

## 1. 服務概觀

| 項目 | 值 |
| :--- | :--- |
| 服務名 | `smart_factory` |
| 框架 | BentoML（`@bentoml.service`），見 [ADR-001](../architecture/adr/ADR-001-bentoml-as-serving-framework.md) |
| 啟動 | `make serve` |
| 預設位址 | `http://localhost:3000` |
| 掛載模型 | 表格（XGBoost）＋ 影像（ONNX），**兩個獨立降級** |

BentoML 會依 `@bentoml.api` 的方法名自動產生路由，因此端點路徑即方法名。

---

## 2. 端點

### 2.1 `POST /predict_maintenance` — 預測性維護（批次）

一次送多筆感測讀數，回傳每台設備的故障機率。

**請求** `MaintenanceRequest`

| 欄位 | 型別 | 必填 | 約束 |
| :--- | :--- | :---: | :--- |
| `readings` | `SensorReading[]` | ✓ | **1–1000 筆** |

`SensorReading`：

| 欄位 | 型別 | 必填 | 約束 |
| :--- | :--- | :---: | :--- |
| `machine_id` | string | ✓ | 設備實體 ID，例 `machine_01` |
| `event_timestamp` | datetime \| null | — | ISO 8601。純表格推論可省略 |
| `temperature` | float | ✓ | `-50 ≤ x ≤ 500` |
| `vibration` | float | ✓ | `x ≥ 0` |
| `current` | float | ✓ | `x ≥ 0` |

**回應** `MaintenanceResponse`

| 欄位 | 型別 | 說明 |
| :--- | :--- | :--- |
| `model_name` | string | 實際命中的 registry 模型名稱 |
| `model_version` | string | 版本或 stage，見 [§5](#5-模型版本語意) |
| `threshold` | float | 本次判定用的門檻 |
| `predictions[].machine_id` | string | 對應輸入 |
| `predictions[].failure_probability` | float | `0 ≤ p ≤ 1` |
| `predictions[].will_fail` | bool | `p >= threshold` |

**範例**

```bash
curl -X POST http://localhost:3000/predict_maintenance \
  -H "Content-Type: application/json" \
  -d '{"readings":[
        {"machine_id":"machine_01","temperature":78.5,"vibration":4.2,"current":11.3},
        {"machine_id":"machine_02","temperature":61.0,"vibration":2.8,"current":9.4}]}'
```

```json
{"model_name":"smart_factory_tabular","model_version":"local","threshold":0.5,
 "predictions":[{"machine_id":"machine_01","failure_probability":0.994,"will_fail":true},
                {"machine_id":"machine_02","failure_probability":0.0001,"will_fail":false}]}
```

---

### 2.2 `POST /predict_defect` — 瑕疵檢測（單張影像）

**請求**：`multipart/form-data`，欄位 `image`，內容為影像檔（BentoML 會轉成 PIL Image）。
服務端一律 `convert("RGB")` 並 resize 到模型輸入尺寸。

**回應** `DefectResponse`

| 欄位 | 型別 | 說明 |
| :--- | :--- | :--- |
| `model_name` / `model_version` / `threshold` | — | 同上 |
| `prediction.label` | `"good"` \| `"defect"` | 列舉，不會有其他值 |
| `prediction.defect_probability` | float | `0 ≤ p ≤ 1` |

```bash
curl -X POST http://localhost:3000/predict_defect -F image=@sample.png
```

---

### 2.3 `POST /healthz` — 健康檢查

無請求主體。**兩個模型分別回報**，這是刻意的：一個模型掛掉不代表整個服務不可用。

| 欄位 | 型別 | 說明 |
| :--- | :--- | :--- |
| `status` | `"ok"` \| `"degraded"` | **兩個模型都就緒才是 `ok`** |
| `tabular_model_loaded` | bool | 表格模型是否就緒 |
| `vision_model_loaded` | bool | 影像模型是否就緒 |

> `degraded` 表示服務活著但功能不完整——容器探活（liveness）不該因此重啟它，
> 但就緒探測（readiness）與告警應該注意。處理見 [runbook](../ops/runbook-serving-degraded.md)。

---

## 3. 共用約定

| 約定 | 值 | 為什麼 |
| :--- | :--- | :--- |
| 實體欄位 | `machine_id` | 與 Feast entity、`toy_sensors.csv` 對齊 |
| 時間欄位 | `event_timestamp` | 與 Feast 的 point-in-time join 對齊 |
| 表格特徵 | `temperature` / `vibration` / `current` | 與訓練特徵同名同序 |
| 批次上限 | 1000 筆 | 防止單一請求耗盡記憶體 |
| 門檻 | 由 `conf/config.yaml` 的 `serving.*_threshold` 提供，預設 `0.5` | 門檻是設定不是硬編碼，可不改程式調整 |

**所有外部輸入在進入推論邏輯前都先過 Pydantic 驗證**（fail-fast）。
數值範圍不是裝飾——`temperature` 限 `-50~500` 是為了擋掉感測器故障時的離譜值進入模型。

---

## 4. 錯誤處理

| 情況 | 回應 | 來源 |
| :--- | :--- | :--- |
| 欄位缺失 / 型別錯 / 超出範圍 | **422**，訊息指出哪個欄位 | Pydantic |
| `readings` 為空或超過 1000 | 422 | Pydantic 的 `min_length` / `max_length` |
| 對應模型未就緒 | **503** `ServiceUnavailable` | `services/service.py` 明確拋出 |
| 模型 registry 與本地產物皆不可用 | 啟動時該模型為 `None` → 呼叫時 503；`healthz` 回 `degraded` | `src/serving/model_loader.py` |

**設計取捨**：模型載入失敗**不會讓整個服務啟動失敗**。
兩個端點各自降級，另一個仍可服務。代價是「服務活著但功能殘缺」，所以 `healthz` 必須被監控。

---

## 5. 模型版本語意

`model_version` 有兩種可能，**意義完全不同**：

| 值 | 意義 | 該有的反應 |
| :--- | :--- | :--- |
| `Production` / `Staging` | 從 MLflow Registry 以 `models:/<name>/<stage>` 取得 | 正常 |
| `local` | **registry 不可用，回退讀本地檔案** | 生產環境視為告警 |

回退機制是為了離線 / 教學環境能 demo（見 [`../architecture/sad.md` §5.2](../architecture/sad.md#52-推論時的模型解析)）。
但在生產環境看到 `local`，代表 registry 連不上——此時服務提供的可能是**過期的模型**，且沒有任何錯誤。
**建議：監控回應中的 `model_version`，出現 `local` 即告警。**

---

## 6. 契約對齊規則

改欄位時必須同步四處，缺一就會產生訓練/服務不一致：

1. `src/serving/schemas.py` — Pydantic 定義（**權威**）
2. `feature_repo/` — Feast 特徵定義（欄名要對齊）
3. `src/features/build_features.py` — 訓練期特徵計算
4. 本文件 ＋ [`openapi.yaml`](openapi.yaml)

> 特徵欄名在訓練與服務兩側必須**同名同序**。XGBoost 依欄位順序取值，
> 順序錯了不會報錯，只會靜默給出錯誤預測——這是最難查的一類 bug。
