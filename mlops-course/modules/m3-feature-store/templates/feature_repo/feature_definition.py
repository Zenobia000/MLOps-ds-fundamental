"""
workspace/feature_repo/feature_definition.py — M3 特徵契約模板

在 feature_repo/ 執行：feast apply
對照：modules/m3-feature-store/sandbox/feature_repo/feature_definition.py

填空方式：搜尋 TODO(M3- 把 ??? 換成正確值；改完再 feast apply。
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

# TODO(M3-5): join_keys 應與 parquet 欄位一致（提示：["patient_id"]）
patient = Entity(
    name="patient",
    join_keys=["???"],  # ← 改這裡
    description="一位糖尿病篩檢病患",
)

# TODO(M3-6): path 相對 feature_repo/；先跑 prepare_features.py 產出 parquet
predictors_source = FileSource(
    name="predictors_source",
    path="data/???",  # ← 改成 predictors_df.parquet
    event_timestamp_column="event_timestamp",
)

target_source = FileSource(
    name="target_source",
    path="data/???",  # ← 改成 target_df.parquet
    event_timestamp_column="event_timestamp",
)

# TODO(M3-7): ttl 天數對照 sandbox（提示：timedelta(days=2)）；online=True
predictors_fv = FeatureView(
    name="predictors_feature_view",
    entities=[patient],
    ttl=timedelta(days=0),  # ← 改成正確 ttl
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
    online=True,
    tags={"team": "mlops-course", "stage": "m3-workspace"},
)

target_fv = FeatureView(
    name="target_feature_view",
    entities=[patient],
    ttl=timedelta(days=0),  # ← 同上
    schema=[
        Field(name="Outcome", dtype=Int64),
    ],
    source=target_source,
    online=True,
    tags={"team": "mlops-course", "stage": "m3-workspace"},
)
