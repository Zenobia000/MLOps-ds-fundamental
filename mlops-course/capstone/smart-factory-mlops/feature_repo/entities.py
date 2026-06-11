"""Feast 實體定義（entities）。

實體（entity）是特徵的關聯鍵。本專案唯一實體為設備 ``machine_id``，
與全專案契約一致（loaders / build_features / data sources 皆用此鍵）。
"""

from feast import Entity, ValueType

# 設備實體：以 machine_id 作為所有感測器特徵的 join key。
machine = Entity(
    name="machine",
    join_keys=["machine_id"],
    value_type=ValueType.STRING,
    description="產線設備（machine_id 為唯一識別鍵）。",
)
