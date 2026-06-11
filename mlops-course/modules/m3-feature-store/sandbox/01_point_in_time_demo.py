"""
01_point_in_time_demo.py — Feast 的四個最小可用動詞,一次跑完。

這個檔示範什麼:
    用同一份 diabetes 特徵,走完特徵商店的完整生命週期:
        1. get_historical_features  → 做 point-in-time join,組「訓練集」(離線)
        2. materialize              → 把最新特徵搬進線上 store(sqlite)
        3. get_online_features      → 用 patient_id 秒查,模擬「即時推論」(線上)
    全程的重點是一句話:訓練時和服務時拿到的特徵,必須是同一套邏輯算出來的。

【為什麼要 point-in-time?「時間穿越」陷阱】
    想像你要預測「某病患在 T 時刻會不會被診斷為糖尿病」。
    錯誤做法:直接把整張特徵表 join 標籤表,不看時間。
        → 你可能把「T 之後」才量到的特徵也餵進訓練集。
        → 模型在訓練/驗證時「偷看到未來」,離線指標被灌得很漂亮(虛高),
          但上線後根本拿不到未來特徵,實戰表現直接崩盤。
    正確做法:get_historical_features 會嚴格只取「entity 事件時間 <= 該時刻」
        且仍在 ttl 內的最新特徵,徹底杜絕時間穿越。

怎麼跑:
    先準備資料與 registry:
        python 00_prepare_data.py
        cd feature_repo && feast apply && cd ..
    再跑本檔:
        python 01_point_in_time_demo.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from feast import FeatureStore

# 設定 seed:雖然本檔不直接用亂數,仍宣告以符合本 repo「有隨機性就設 seed」慣例。
SEED = 42

# feature_repo 的路徑(相對本檔),FeatureStore 要指到含 feature_store.yaml 的資料夾。
HERE = Path(__file__).resolve().parent
REPO_PATH = HERE / "feature_repo"

# ★ 關鍵:Feast 解析 FileSource 的相對路徑(data/xxx.parquet)是相對「當前工作目錄」,
#   不是 repo_path。所以我們先把工作目錄切進 feature_repo,確保不論從哪裡執行本檔
#   都能正確找到 data/ 下的 parquet。(等同手動 cd feature_repo 再跑)
os.chdir(REPO_PATH)


def build_training_set(store: FeatureStore) -> pd.DataFrame:
    """動詞 1:get_historical_features — 用 point-in-time join 組訓練集。"""
    # entity_df 是「我想在這些時刻、針對這些病患取特徵」的查詢清單。
    # 每一列 = (要哪位病患, 在哪個時刻) → Feast 回頭去找「那個時刻」的正確特徵。
    #
    # ★ 真實世界的做法:時間點來自「標籤被觀測到的時刻」。
    #   所以這裡直接讀 target_df.parquet 的 event_timestamp 當查詢時刻 —— 也就是
    #   「在每位病患被確診的那一刻,我當時手上有的特徵長什麼樣?」
    #   這樣每個查詢時刻都緊貼該病患的特徵時間,落在 ttl(2 天)內,join 得到資料。
    target_df = pd.read_parquet("data/target_df.parquet")  # 已 cd 進 feature_repo
    entity_df = target_df.loc[
        target_df["patient_id"].isin([0, 1, 2, 3, 4]),
        ["patient_id", "event_timestamp"],
    ].copy()
    # event_timestamp 在 parquet 裡已是 UTC tz-aware,直接拿來用,
    # Feast 只會撈出「事件時間 <= 此標籤觀測時刻」的特徵,絕不穿越到未來。

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "predictors_feature_view:Glucose",
            "predictors_feature_view:BMI",
            "predictors_feature_view:Age",
            "target_feature_view:Outcome",   # 標籤也一起 join,訓練資料一次到位
        ],
    ).to_df()
    return training_df


def main() -> None:
    # 動詞 0(前置):apply 已在 CLI 做過(feast apply)。
    # 這裡載入那份 registry,拿到可操作的 store 物件。
    store = FeatureStore(repo_path=str(REPO_PATH))

    print("=" * 64)
    print("步驟 1) get_historical_features:point-in-time join 組訓練集")
    print("=" * 64)
    training_df = build_training_set(store)
    # 注意:每位病患的 Glucose/BMI/Age 都是「在該病患標籤觀測時刻或之前」最新的一筆,
    #       絕不會混進那個時刻「之後」才量到的數值 → 沒有時間穿越。
    print(training_df.to_string(index=False))

    print("\n" + "=" * 64)
    print("步驟 2) materialize:把最新特徵搬進線上 store(sqlite)")
    print("=" * 64)
    # materialize 會把 [start, end] 區間內、每個 entity 的「最新」特徵
    # 寫進 online_store.db,讓線上推論能 O(1) 秒查。
    # 用帶 UTC 時區的時間,與 parquet 的 event_timestamp 保持一致(都是 UTC)。
    store.materialize(
        start_date=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    print("materialize 完成:online_store.db 已更新。")

    print("\n" + "=" * 64)
    print("步驟 3) get_online_features:用 patient_id 秒查,模擬即時推論")
    print("=" * 64)
    # 線上查詢「不帶時間戳」:它永遠回傳 materialize 進去的「最新」特徵。
    # ★ 一致性保證:這裡的特徵欄位與步驟 1 訓練時完全相同
    #   (同一個 feature view、同一份定義),所以「訓練/服務一致」。
    online_features = store.get_online_features(
        features=[
            "predictors_feature_view:Glucose",
            "predictors_feature_view:BMI",
            "predictors_feature_view:Age",
        ],
        entity_rows=[{"patient_id": 0}, {"patient_id": 1}, {"patient_id": 2}],
    ).to_dict()
    print(pd.DataFrame(online_features).to_string(index=False))

    print("\n完成。對照一下:")
    print("  - 步驟 1 是『離線、回溯歷史時刻』取特徵 → 給訓練用")
    print("  - 步驟 3 是『線上、取當下最新』取特徵   → 給推論用")
    print("  兩者欄位定義同源,這正是特徵商店避免訓練/服務不一致的價值。")


if __name__ == "__main__":
    main()
