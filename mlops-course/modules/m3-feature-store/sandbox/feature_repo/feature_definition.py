"""
feature_definition.py — Feast 的「特徵契約」。

這個檔示範什麼:
    用程式碼把「特徵長什麼樣、從哪來、屬於誰」寫死下來,讓訓練與服務
    讀的是「同一份定義」,這是 Feast 解決『訓練/服務不一致』的核心。
    本檔只「宣告」,不執行任何運算;真正生效是在你跑 `feast apply` 時。

怎麼跑:
    不要直接 python 這個檔。它是被 feast CLI 掃描的定義檔:
        cd feature_repo
        feast apply
    Feast 會掃描本資料夾所有 .py,找出 Entity / FeatureView 並登記到 registry。

三個核心名詞:
    Entity      = 特徵掛在「誰」身上(這裡是 patient,join key = patient_id)
    FileSource  = 離線特徵的來源檔(parquet),內含 event_timestamp
    FeatureView = 一組特徵的集合 + 它的來源 + 存活時間(ttl)
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

# --- Entity:特徵的主體 ---
# join_keys 是 join 時用來對齊的欄位名,必須和 parquet 裡的欄位一致。
patient = Entity(
    name="patient",
    join_keys=["patient_id"],
    description="一位糖尿病篩檢病患",
)

# --- 來源 1:預測特徵(predictors)---
# event_timestamp_column 告訴 Feast「這筆特徵是何時量到的」,
# 這是 point-in-time join 能正確運作的關鍵。
predictors_source = FileSource(
    name="predictors_source",
    path="data/predictors_df.parquet",        # 相對 feature_repo/ 的路徑
    event_timestamp_column="event_timestamp",
)

# --- 來源 2:標籤(target / Outcome)---
target_source = FileSource(
    name="target_source",
    path="data/target_df.parquet",
    event_timestamp_column="event_timestamp",
)

# --- Feature View 1:8 個輸入特徵 ---
# ttl(time-to-live):一筆特徵在事件時間之後多久內仍算「有效」。
# 設 2 天表示 join 時,只回溯找 2 天內的特徵;太舊的視為過期。
predictors_fv = FeatureView(
    name="predictors_feature_view",
    entities=[patient],
    ttl=timedelta(days=2),
    schema=[
        Field(name="Pregnancies", dtype=Int64),
        Field(name="Glucose", dtype=Int64),
        Field(name="BloodPressure", dtype=Int64),
        Field(name="SkinThickness", dtype=Int64),
        Field(name="Insulin", dtype=Int64),
        Field(name="BMI", dtype=Float64),
        Field(name="DiabetesPedigreeFunction", dtype=Float64),
        Field(name="Age", dtype=Int64),
    ],
    source=predictors_source,
    online=True,            # True 才能 materialize 到線上 store 供即時查詢
    tags={"team": "mlops-course", "stage": "m3"},
)

# --- Feature View 2:標籤 ---
# 把標籤也做成 FeatureView,訓練時就能用一次 join 同時取得特徵 + 對應時刻的標籤。
target_fv = FeatureView(
    name="target_feature_view",
    entities=[patient],
    ttl=timedelta(days=2),
    schema=[
        Field(name="Outcome", dtype=Int64),
    ],
    source=target_source,
    online=True,
    tags={"team": "mlops-course", "stage": "m3"},
)
