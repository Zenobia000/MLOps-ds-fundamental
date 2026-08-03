# ADR-002：導入 Feast 當特徵商店

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 已接受
> **Owner:** Capstone 維護者
> **實例:** 每決策一份

## 脈絡

預測性維護要回答的是：「在 T 時刻**之前**，判斷設備在 T 會不會故障」。
這帶來兩個用一般 DataFrame 很難保證的性質：

1. **時間點正確性**：訓練集只能含 `event_timestamp <= 標籤觀測時刻` 的特徵。錯了就是時間穿越，
   離線指標會虛高、上線後崩盤，而且**離線階段不會有任何錯誤訊息告訴你**。
2. **訓練/服務一致**：訓練時算的特徵與線上推論時取的特徵，必須來自同一份定義。
   兩邊各寫一次就會慢慢漂移（training-serving skew）。

候選：不用（自己寫 pandas）、Feast、Tecton、Featureform。

## 決策

**採用 Feast**，`local` provider + 檔案 offline store（parquet）+ SQLite online store。
特徵定義在 `feature_repo/`，讀寫封裝在 `src/features/feast_io.py`。

## 理由

**為什麼不「自己寫 pandas」**——這是最誘人的選項，也是最危險的。
`shift(1)` 這種防穿越的寫法，只要有人手滑忘了寫，洩漏就悄悄回來了，**而且離線指標會變好看**，
不會有人來抱怨。把「只取過去」從「工程師要記得的紀律」變成**基礎設施層級的保證**，才是 feature store 的價值。

（這個對照在課程 m3 有可執行的示範：洩漏版 AUC 0.940 vs 正確版 0.791，
見 [`02_leakage_viz.ipynb`](../../../../../modules/m3-feature-store/sandbox/02_leakage_viz.ipynb)。）

| 候選 | 為什麼不選 |
| :--- | :--- |
| 自己寫 pandas | 沒有基礎設施保證，靠人記得。見上 |
| Tecton | 商業產品，需帳號與雲端資源，違反「本機零外部依賴就能跑完」的教學前提 |
| Featureform | 概念相近但社群規模小，教材選型偏向學員日後最可能遇到的 |

Feast 的 `local` provider 讓整條流程**零外部服務**：offline 是 parquet 檔、online 是 SQLite 檔。
學員 `feast apply` 之後立刻能跑，不用先架 Redis 或雲端資料庫。

## 後果

**正面**

- `get_historical_features` 天生只回溯不穿越，point-in-time 正確性由工具保證
- `materialize` → `get_online_features` 讓線上推論拿到與訓練同源的特徵定義
- entity（`machine_id`）與時間欄（`event_timestamp`）成為跨模組的契約，schema 也照這個對齊

**負面 / 代價**

- 多一層抽象與一個要維護的 registry；小專案會覺得「殺雞用牛刀」
- SQLite online store **無法水平擴展、無高可用**。教學規模夠用，生產要換 Redis/DynamoDB
  （`docker-compose.yml` 已預留 redis 服務）
- Feast 版本演進快（本專案鎖 0.47）。0.40 → 0.47 期間 `Entity` 的 `value_type` 從可省略變成即將必填，
  已在 `feature_repo/feature_definition.py` 顯式宣告以免未來升版壞掉

## 相關

- [`../sad.md` §6 資料架構](../sad.md#6-資料架構)
- 課程 m3 模組（Feast 的四個最小可用動詞）
