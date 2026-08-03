# ADR-003：MLflow 與 DVC 並存，各管一半

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 已接受
> **Owner:** Capstone 維護者
> **實例:** 每決策一份

## 脈絡

「可重現」需要四樣東西同時被固定：**程式碼版本、資料版本、環境版本、隨機種子**。
Git 管程式碼、seed 由 `src/utils/seed.py` 管，剩下資料與環境。

常見的疑問是：MLflow 也能存 artifact、DVC 也能記 metrics，**為什麼要兩個工具？**
這個 ADR 就是回答它——因為新人看到兩套版本控制通常會想砍掉一個。

## 決策

**兩者並存，職責明確切開**：

| 工具 | 管什麼 | 不管什麼 |
| :--- | :--- | :--- |
| **MLflow** | 實驗的**過程與結果**：params、metrics、model artifact、signature、Model Registry 的生命週期（Staging/Production） | 不管輸入資料的版本 |
| **DVC** | **資料與管線**的版本：`data/` 的內容、`dvc.yaml` 的 stage 依賴、哪個 commit 對應哪份資料 | 不管實驗比較與模型註冊 |

分界線一句話：**MLflow 回答「這次實驗做了什麼、結果多好」，DVC 回答「這次實驗吃的是哪一份資料」。**

## 理由

**為什麼不只用 MLflow**
MLflow 能把資料集當 artifact 上傳，但那是「複製一份存起來」，不是版本控制——
沒有 `dvc checkout` 那種「切回舊 commit 就拿回當時資料」的能力，而且大檔案重複上傳很快就爆。

**為什麼不只用 DVC**
DVC 的 `metrics` 能記分數、`dvc exp` 也能比較實驗，但它沒有 Model Registry：
沒有 Staging→Production 的生命週期、沒有 alias、沒有給服務端 `models:/name/stage` 這種解析 URI。
`src/serving/model_loader.py` 正是靠這個 URI 取模型的。

**兩者的接點**
`dvc.yaml` 的 `train` stage 呼叫 `python -m src.training.train`，而該腳本內部開 MLflow run。
於是：DVC 保證「這個 stage 的輸入是哪一版資料」，MLflow 記錄「這次跑出什麼」。
`evaluate` stage 把主指標寫成 DVC metrics，讓 `dvc metrics diff` 能跨 commit 比較。

## 後果

**正面**

- 「同一個 git commit 永遠對應同一份資料 + 同一組實驗紀錄」，四要素齊全
- 服務端能用 registry 的 stage 解析模型，與訓練解耦
- `dvc repro` 只重跑受影響的 stage，省時間

**負面 / 代價**

- **學習成本加倍**：學員要同時理解兩套心智模型，這是本專案最常見的困惑點。
  課程刻意把它拆成 m2 的階 1（MLflow）與階 3（DVC）分開教，就是為了降低這個負荷
- 兩處都要設定 remote（MLflow tracking URI、DVC remote），環境設定變複雜
- 職責邊界要靠紀律維持。**明確禁止**：不要用 DVC 存模型 artifact，也不要用 MLflow 當資料版本控制

## 現況備註

本專案的 MLflow 預設用 **SQLite backend**（`sqlite:///mlflow.db`）而非 file store，
因為 MLflow 3.x 已把 file store 列為維護模式。設 `MLFLOW_TRACKING_URI` 可覆蓋成遠端 server
（`make up` 會起一個）。DVC remote 用本地資料夾即可，不需雲端帳號。

## 相關

- [`../sad.md` §5.1 訓練 → 註冊](../sad.md#51-訓練--註冊含品質門檻)
- [`../sad.md` §6 資料架構](../sad.md#6-資料架構)
