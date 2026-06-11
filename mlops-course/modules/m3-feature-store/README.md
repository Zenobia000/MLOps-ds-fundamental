# m3 — 特徵商店 Feast(技能階梯 階4)

## 1. 本模組學什麼

學會用 **Feast** 把「特徵」當成一個獨立、可治理的資產來管理:用 **point-in-time(時間點正確性)join** 組訓練集,並讓**訓練時**與**線上服務時**讀到的特徵來自**同一份定義**,從根本上消除「訓練/服務不一致」。對應技能階梯 **階4(Feature Store)**。

> 一句話心法:特徵商店要解決的不是「存特徵」,而是「在正確的時間點、給訓練和服務同一套特徵」。

### 為什麼這件事這麼重要?(錯誤做法 → 指標虛高)

| | 錯誤做法(無時間概念的 join) | 正確做法(point-in-time join) |
| :-- | :-- | :-- |
| 怎麼做 | 把整張特徵表直接 join 標籤表,不看時間 | 只取「entity 事件時間 ≤ 標籤觀測時刻」且仍在 ttl 內的最新特徵 |
| 後果 | 訓練集混進「未來才量到」的特徵 → 模型偷看到答案 | 訓練集只含「當下能拿到」的特徵 |
| 指標 | 離線 AUC/Accuracy **虛高**,看起來超棒 | 離線指標誠實,接近上線真實表現 |
| 上線後 | 線上根本拿不到未來特徵 → **效能崩盤** | 訓練/服務一致,表現穩定 |

這個「時間穿越(time travel / data leakage)」陷阱,就是 `sandbox/01_point_in_time_demo.py` 註解裡反覆標出的重點。

---

## 2. 沙盒步驟(Layer 1:照編號逐個跑,只學最小可用動詞)

四個最小可用動詞:**apply → get_historical_features → materialize → get_online_features**。

先進到沙盒並裝依賴(建議用乾淨 venv):

```bash
cd modules/m3-feature-store/sandbox
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

然後照順序跑:

```bash
# (1) 準備資料:把 diabetes.csv 補上 patient_id + event_timestamp,拆成兩張 parquet
python 00_prepare_data.py
#   → 產出 feature_repo/data/predictors_df.parquet 與 target_df.parquet

# (2) apply:讓 Feast 掃描 feature_definition.py,把 entity / feature view 登記到 registry
cd feature_repo
feast apply
cd ..

# (3) 跑完整生命週期 demo:
#     get_historical_features(離線/訓練) → materialize(搬到線上) → get_online_features(線上/推論)
python 01_point_in_time_demo.py
```

各檔在教什麼:

| 檔案 | 學到的最小動詞/概念 |
| :-- | :-- |
| `sandbox/00_prepare_data.py` | 離線來源要有 **entity 欄位 + event_timestamp** 才能做 point-in-time |
| `sandbox/feature_repo/feature_store.yaml` | `local` provider + `sqlite` 線上 store 的最小設定 |
| `sandbox/feature_repo/feature_definition.py` | **Entity / FileSource / FeatureView** 三個核心物件 |
| `sandbox/01_point_in_time_demo.py` | `get_historical_features` / `materialize` / `get_online_features` |

> 提醒:`sandbox/` 的每個檔都能獨立理解,彼此不互相 import。

---

## 3. 整合任務(Layer 2:把 Feast 接到 workspace/)

到 `workspace/`,把「特徵」這一層接進你正在長大的專案主線。目標:讓訓練腳本不再直接讀 CSV,而是**透過 Feast 取特徵**。

TODO 提示:

- [ ] **TODO 1**:在 `workspace/` 建一個 `feature_repo/`,把本沙盒的 `feature_store.yaml` 與 `feature_definition.py` 搬過去(路徑改成 workspace 的相對路徑)。
- [ ] **TODO 2**:寫一支 `prepare_features.py`,用 m1/m2 已經在用的玩具資料產出帶 `event_timestamp` 的 parquet(沿用 `00_prepare_data.py` 的補時間戳手法)。
- [ ] **TODO 3**:把 workspace 既有的「讀 CSV → 訓練」流程,改成 `store.get_historical_features(...)` 取訓練集;確認欄位與原本一致。
- [ ] **TODO 4**:在訓練完後呼叫 `store.materialize(...)`,並寫一支極簡推論函式用 `store.get_online_features(...)` 取線上特徵。
- [ ] **TODO 5**(進階):把 m2 的 MLflow run 串起來 —— 記錄「這次訓練用的 feature view 名稱與版本」,讓實驗可追溯到特徵定義。

驗收標準:訓練與推論讀的是**同一組 feature view**,且訓練集是 point-in-time join 出來的(沒有時間穿越)。

---

## 4. 卡住怎麼辦

從上一個模組的已知良好快照重置,確保起點乾淨:

```
checkpoints/after-m2/
```

把它的內容覆蓋回你的 `workspace/`,即可回到「m2 結束、m3 還沒開始」的狀態重來。

常見坑:

- `feast apply` 找不到 parquet → 你忘了先跑 `python 00_prepare_data.py`。
- `get_historical_features` 回傳特徵全是 NaN → entity_df 的 `event_timestamp` 早於特徵事件時間,或超過 ttl(本例 ttl=2 天)。這正是 point-in-time 在保護你。
- `feast` 指令找不到 → 確認 venv 已啟動且 `pip install -r requirements.txt` 成功。

---

## 5. 檢核題(自我確認)

1. 為什麼離線來源(FileSource)一定要有 `event_timestamp` 欄位?少了它會發生什麼事?
2. 「無時間概念的 join」為什麼會讓離線指標**虛高**?用「時間穿越」一詞解釋。
3. `get_historical_features` 與 `get_online_features` 的差別是什麼?各自服務於訓練還是推論?
4. `materialize` 在整個流程裡扮演什麼角色?不做它的話 `get_online_features` 會怎樣?
5. 「訓練/服務一致」在本模組是靠什麼機制保證的?(提示:同一份 _______ 定義)
