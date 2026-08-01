# M3 應用指導手冊：Feast 特徵商店

## 這個模組解決什麼工程問題

M3 解決的是訓練與線上推論之間的特徵一致性問題。很多模型離線分數很好，上線後卻失效，原因不是模型本身，而是訓練時偷用了未來資訊，或線上服務時拿不到與訓練一致的特徵。

Feature store 的核心不是「存特徵」，而是「用同一份定義，在正確時間點，為訓練與推論提供一致特徵」。

## 哪邊應用

M3 適用在模型開始依賴可重用特徵、而且特徵有時間或線上查詢需求的階段。只要訓練與推論可能各自實作一份特徵邏輯，就有 training-serving skew 風險。

常見應用：

- 風控與醫療：依照事件時間取當時已知特徵，避免資料洩漏。
- 推薦系統：線上依 `user_id`、`item_id` 查最新特徵。
- 預測性維護：依 `machine_id` 查感測器彙總特徵。
- 多模型平台：多個模型共用同一套特徵契約。

## 怎麼用

使用順序是先定義主體，再定義來源與特徵契約，最後串訓練與服務：

1. 在離線資料中準備 entity key 與 `event_timestamp`。
2. 定義 Entity，例如 `patient`、`machine`、`customer`。
3. 定義 FileSource，指向 Parquet 或其他離線來源。
4. 定義 FeatureView，包含 schema、ttl、online 設定。
5. 執行 `feast apply` 寫入 registry。
6. 訓練用 `get_historical_features` 做 point-in-time join。
7. 推論前用 `materialize` 寫入 online store。
8. 線上服務用 `get_online_features` 查即時特徵。

## 本模組元件

| 元件 | 在 MLOps 裡的職責 | 本課使用方式 |
| :--- | :--- | :--- |
| Feast | 特徵商店框架 | 管理 Entity、FeatureView、offline/online 查詢 |
| Entity | 特徵主體 | `patient`，join key 是 `patient_id` |
| FileSource | 離線特徵來源 | Parquet 檔案 |
| FeatureView | 特徵契約 | predictors 與 target 兩組 feature view |
| Registry | 特徵定義登記簿 | `registry.db` |
| Online Store | 即時查特徵的儲存層 | `online_store.db` SQLite |
| Point-in-time join | 時間點正確的訓練資料組裝 | `get_historical_features` |

## Feast 怎麼應用

Feast 讓你用程式碼定義 feature contract，然後讓訓練與服務都透過這份 contract 取特徵。

典型資料流：

```text
offline source -> FeatureView -> registry
FeatureView + entity_df -> get_historical_features -> training dataset
FeatureView -> materialize -> online store -> get_online_features -> inference
```

工程應用場景：

- 多個模型共用同一批特徵。
- 訓練與推論特徵邏輯容易不一致。
- 特徵有時間概念，需要避免 data leakage。
- 線上推論需要低延遲查詢最新特徵。

## Entity 怎麼設計

Entity 是特徵附著的主體。設計 entity 的問題是：「模型推論時，我會用什麼 key 找到這筆資料？」

例子：

| 場景 | Entity | Join key |
| :--- | :--- | :--- |
| 糖尿病風險 | patient | `patient_id` |
| 預測性維護 | machine | `machine_id` |
| 推薦系統 | user、item | `user_id`、`item_id` |
| 詐欺偵測 | account、transaction | `account_id`、`transaction_id` |

設計原則：

- join key 名稱要和資料欄位一致。
- entity 不應該是模型或資料表名稱，而是業務主體。
- 線上服務收到的 key 應能對應到 entity key。

## FileSource 與 event timestamp 怎麼設計

Feast 的 FileSource 指向離線資料來源。本課使用 Parquet，並要求每筆資料有 `event_timestamp`。

為什麼 `event_timestamp` 重要：

- 它表示特徵是在什麼時間被觀測到。
- point-in-time join 會用它避免取到未來資料。
- 沒有它，訓練資料可能發生時間穿越。

資料最低要求：

| 欄位 | 用途 |
| :--- | :--- |
| entity key | 例如 `patient_id`，用來 join 主體 |
| event timestamp | 表示特徵觀測時間 |
| feature columns | 真正提供給模型的特徵 |

本課資料關係：

| 檔案 | 內容 |
| :--- | :--- |
| `predictors_df.parquet` | `patient_id`、`event_timestamp`、8 個 predictor features |
| `target_df.parquet` | `patient_id`、`event_timestamp`、`Outcome` label |

## FeatureView 怎麼設計

FeatureView 是一組特徵的契約，定義 schema、來源、entity、ttl、是否可進 online store。

本課範例：

```python
predictors_fv = FeatureView(
    name="predictors_feature_view",
    entities=[patient],
    ttl=timedelta(days=2),
    schema=[
        Field(name="Pregnancies", dtype=Int64),
        Field(name="Glucose", dtype=Int64),
        Field(name="BMI", dtype=Float64),
    ],
    source=predictors_source,
    online=True,
)
```

設計判斷：

| 設計項 | 判斷方式 |
| :--- | :--- |
| `name` | 用業務可理解的特徵組名稱 |
| `entities` | 這組特徵屬於哪個 entity |
| `schema` | 明確列出欄位與型別 |
| `ttl` | 特徵多久後不應再被使用 |
| `online` | 是否需要線上即時查詢 |

## Registry 怎麼應用

Registry 是 Feast 的特徵定義登記簿。`feast apply` 會掃描 `feature_definition.py`，把 Entity、FileSource、FeatureView 寫入 registry。

工程應用：

- 讓團隊知道目前可用的特徵定義。
- 讓訓練與服務端使用同一份 feature contract。
- 作為 feature store 狀態的版本化基礎。

操作流程：

```bash
cd feature_repo
feast apply
```

## Online Store 怎麼應用

Online store 是服務線上推論的低延遲特徵儲存。本課用 SQLite，正式環境常用 Redis 或雲端線上儲存。

使用流程：

```bash
feast materialize <start-time> <end-time>
```

然後服務端用：

```python
store.get_online_features(
    features=[
        "predictors_feature_view:Glucose",
        "predictors_feature_view:BMI",
    ],
    entity_rows=[{"patient_id": 123}],
)
```

工程判斷：

- 批次訓練用 offline data。
- 即時推論用 online store。
- materialize 是把 offline feature 搬到 online store 的動作。
- online store 空表時，`get_online_features` 查不到可用值。

## Point-in-time join 怎麼應用

Point-in-time join 是 feature store 的核心。它保證訓練資料只使用當下已經存在的特徵。

錯誤做法：

```text
直接用 patient_id join predictors 和 target，不看時間
```

風險：

- 可能使用未來才產生的特徵。
- 離線指標虛高。
- 上線後模型拿不到同樣資訊，表現崩盤。

正確做法：

```python
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "predictors_feature_view:Glucose",
        "predictors_feature_view:BMI",
        "target_feature_view:Outcome",
    ],
).to_df()
```

## 最小落地流程

1. 準備離線資料，確認有 entity key 與 `event_timestamp`。
2. 定義 Entity，例如 `patient` 或 `machine`。
3. 定義 FileSource，指向 Parquet 或其他離線來源。
4. 定義 FeatureView，包含 schema、ttl、online 設定。
5. 執行 `feast apply` 寫入 registry。
6. 用 `get_historical_features` 建訓練資料。
7. 用 `materialize` 將特徵送進 online store。
8. 用 `get_online_features` 模擬線上推論查特徵。

## 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 資料沒有 `event_timestamp` | 無法做正確時間 join | 在資料準備階段補上 |
| 只用 entity key join | 可能偷看到未來資料 | 使用 `get_historical_features` |
| ttl 設太短 | 訓練資料出現大量 NaN | 根據業務時效調整 ttl |
| ttl 設太長 | 過舊特徵仍被使用 | 定義特徵有效期限 |
| 忘記 materialize | online store 查不到資料 | 部署前執行 materialize |
| 訓練與服務各寫一份特徵邏輯 | training-serving skew | 兩邊都透過 Feast 定義取特徵 |

## 工程驗收標準

- `feast apply` 成功建立 registry。
- FeatureView schema 與來源資料欄位一致。
- `get_historical_features` 可產生訓練表。
- 訓練資料沒有使用未來 timestamp。
- `materialize` 後 online store 可查到指定 entity 的特徵。
- 訓練與推論使用同一份 feature definition。

## 你應該帶走的能力

完成 M3 後，工程師應該能判斷什麼情境需要 feature store，並能設計 entity、feature view、offline source、online store 的最小可用架構，避免訓練與服務特徵不一致。
