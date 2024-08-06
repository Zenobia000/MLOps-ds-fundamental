# This is an example feature definition file

from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

# patient = Entity(name = "patient_id",
#                      value_type = ValueType.INT64,
#                  description = "ID of the patient")
patient = Entity(name="patient", join_keys=["patient_id"])
## Predictors Feature View
file_source = FileSource(path = r"data/predictors_df.parquet",
                         event_timestamp_column = "event_timestamp",)

df1_fv = FeatureView(
    name = "predictors_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [patient],
    schema = [
    Field(name = "Pregnancies", dtype = Int64),
    Field(name = "Glucose", dtype = Int64),
    Field(name = "BloodPressure", dtype = Int64),
    Field(name = "SkinThickness", dtype = Int64),
    Field(name = "Insulin", dtype = Int64),
    Field(name = "BMI", dtype = Float64),
    Field(name = "DiabetesPedigreeFunction", dtype = Float64),
    Field(name = "Age", dtype = Int64),       
    ],
    source = file_source,
    online = True,
    tags= {},
)

## Target FEature View

target_source = FileSource(path = r"data/target_df.parquet",
                         event_timestamp_column = "event_timestamp",)

target_fv = FeatureView(
    name = "target_df_feature_view",
    ttl = timedelta(seconds = 86400*2),
    entities = [patient],
    schema = [
    Field(name = "Outcome", dtype = Int64),       
    ],
    source = target_source,
    online = True,
    tags= {},
)

