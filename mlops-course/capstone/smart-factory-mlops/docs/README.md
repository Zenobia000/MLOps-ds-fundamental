# Smart Factory MLOps — 工程文件

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **定位**：本專案的工程文件索引。說明**寫了哪幾份、為什麼是這幾份、哪些刻意不寫**。
> 模板來源：[`VibeCoding_Workflow_Templates`](../../../VibeCoding_Workflow_Templates/INDEX.md)

---

## 1. 這裡有什麼

| 文件 | 回答什麼問題 | 對應模板 |
| :--- | :--- | :--- |
| [`architecture/sad.md`](architecture/sad.md) | 系統由哪些 runtime 組成、邊界在哪、資料怎麼流、怎麼部署 | `03_architecture/sad.md` |
| [`architecture/adr/`](architecture/adr/) | **為什麼選這個工具、放棄了什麼** | `03_architecture/adr.md`（每決策一份） |
| [`design/api_spec.md`](design/api_spec.md) ＋ [`openapi.yaml`](design/openapi.yaml) | 推論服務的對外契約 | `04_design/api_spec.md` |
| [`qa/test_plan.md`](qa/test_plan.md) | 測什麼、測到什麼程度、品質門檻怎麼擋 | `05_qa/test_plan.md` |
| [`ops/`](ops/) | 出事了怎麼辦（漂移告警、門檻擋下、服務降級） | `06_ops/runbook.md`（每症狀一份） |

既有文件不重複造：模型卡與合規自評在 [`../governance/`](../governance/)，資料說明在 [`../data/README.md`](../data/README.md)，
安裝與指令在 [`../README.md`](../README.md)。本目錄只補「那些檔案沒回答的問題」。

---

## 2. 為什麼是這五份（模板有 15 份）

模板庫的 Pilot 階段建議是 15 份全寫。**這個專案刻意不照做**，因為它有兩個特性讓部分模板不適用：

**它是 ML 系統，不是一般 CRUD 應用。** 一般軟體的「正確性」看程式邏輯；ML 系統多了資料與模型兩個會自己腐壞的東西。所以最該被寫下來的不是欄位規格，而是**決策理由**（ADR）與**出事怎麼辦**（runbook）。

**它是教材的參考實作，不是交付客戶的產品。** 沒有付費客戶、沒有商業案例、沒有驗收輪次，所以需求層與 UAT 層的模板對它是空轉。

| 略過的模板 | 為什麼 |
| :--- | :--- |
| `brd` / `prd` / `srs` | 沒有客戶與商業案例。這個專案的「需求」是課程學習目標，已在 [`docs/mlops-course-outline.md`](../../../../docs/mlops-course-outline.md) |
| `ux_research` / `information_architecture` / `ui_spec` | **完全沒有 UI**。對外介面只有三個 HTTP 端點，契約寫在 `api_spec` |
| `db_design` | 沒有關聯式資料庫。持久化只有 Feast 的 SQLite online store 與 MLflow 的 SQLite backend，資料架構寫在 [`sad.md` §6](architecture/sad.md#6-資料架構) |
| `uat_plan` | 沒有客戶驗收輪次。品質把關由 CI 的自動門檻負責，見 [`test_plan.md`](qa/test_plan.md) |
| `lld` | `src/` 每個模組都有完整 docstring，再寫一份低階設計只會變成第二份會過期的真相。模組邊界寫在 `sad.md` §2 |
| `deployment_and_operations` | 部署拓撲寫在 [`sad.md` §7](architecture/sad.md#7-部署視圖)；日常操作寫在 `../README.md` 的 Quickstart。等真的有多環境（stage/prod）再單獨成篇 |

> 模板庫自己的規則就是「**不按序填滿**、只讀與當前範圍相關的章節」。
> 這份清單是那條規則的執行結果，不是偷懶。

---

## 3. 讀的順序

初次接觸這個專案：

1. [`../README.md`](../README.md) — 它是什麼、怎麼跑起來
2. [`architecture/sad.md`](architecture/sad.md) — 系統長什麼樣
3. [`architecture/adr/`](architecture/adr/) — 為什麼長這樣（**教學價值最高**）
4. 要接 API → [`design/api_spec.md`](design/api_spec.md)；要維運 → [`ops/`](ops/)

---

## 4. 已知的教學簡化

這是參考實作，有幾處刻意留成佔位。**文件不會假裝它們是生產就緒的**：

| 位置 | 現況 | 影響 |
| :--- | :--- | :--- |
| `pipelines/deployment_pipeline.py` 的 `canary_probe` | 直接回傳 `1.0`，沒有真的探測 | canary 判定永遠 promote，見 [ADR-005](architecture/adr/ADR-005-deployment-placeholders.md) |
| `infra/terraform/` | 只有 README，無實際 IaC | 無法一鍵起雲端環境 |
| 三個模型的訓練資料 | 玩具 / 合成資料 | 指標不具業務意義（vision 的 f1 常為 0，品質門檻會正確擋下註冊）|

修正這些不是 bug fix，是把教學骨架接上真實資料源——屬於學員的 Capstone 練習範圍。
