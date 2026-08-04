# Smart Factory MLOps — 工程文件

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **定位**：本專案的工程文件索引。說明**寫了哪幾份、為什麼是這幾份、哪些刻意不寫**。
> 模板來源：VibeCoding 工程文件模板庫（`01_requirements`–`06_ops` 六層結構）。
> 該模板庫**不隨本 repo 發布**——它是撰寫這些文件時的參考格式，不是執行期依賴，
> 所以本目錄的文件都是自足的，讀它們不需要拿到模板庫。

---

## 1. 這裡有什麼

| 文件 | 回答什麼問題 | 對應模板 |
| :--- | :--- | :--- |
| [`architecture/sad.md`](architecture/sad.md) | 系統由哪些 runtime 組成、邊界在哪、資料怎麼流、怎麼部署 | `03_architecture/sad.md` |
| [`architecture/adr/`](architecture/adr/) | **為什麼選這個工具、放棄了什麼** | `03_architecture/adr.md`（每決策一份） |
| [`design/api_spec.md`](design/api_spec.md) ＋ [`openapi.yaml`](design/openapi.yaml) | 推論服務的對外契約 | `04_design/api_spec.md` |
| [`design/lld.md`](design/lld.md) | codebase 長什麼樣、誰依賴誰、狀態怎麼流轉 | `04_design/lld.md` |
| [`qa/test_plan.md`](qa/test_plan.md) | 測什麼、測到什麼程度、品質門檻怎麼擋 | `05_qa/test_plan.md` |
| [`ops/`](ops/) | 出事了怎麼辦（漂移告警、門檻擋下、服務降級） | `06_ops/runbook.md`（每症狀一份） |

既有文件不重複造：模型卡與合規自評在 [`../governance/`](../governance/)，資料說明在 [`../data/README.md`](../data/README.md)，
安裝與指令在 [`../README.md`](../README.md)。本目錄只補「那些檔案沒回答的問題」。

---

## 2. 為什麼是這六份（模板有 15 份）

模板庫的 Pilot 階段建議是 15 份全寫。**這個專案刻意不照做**，因為它有兩個特性讓部分模板不適用：

**它是 ML 系統，不是一般 CRUD 應用。** 一般軟體的「正確性」看程式邏輯；ML 系統多了資料與模型兩個會自己腐壞的東西。所以最該被寫下來的不是欄位規格，而是**決策理由**（ADR）與**出事怎麼辦**（runbook）。

**它是教材的參考實作，不是交付客戶的產品。** 沒有付費客戶、沒有商業案例、沒有驗收輪次，所以需求層與 UAT 層的模板對它是空轉。

| 略過的模板 | 為什麼 |
| :--- | :--- |
| `brd` / `prd` / `srs` | 沒有客戶與商業案例。這個專案的「需求」是課程學習目標，已在 [`docs/mlops-course-outline.md`](../../../../docs/mlops-course-outline.md) |
| `ux_research` / `information_architecture` / `ui_spec` | **完全沒有 UI**。對外介面只有三個 HTTP 端點，契約寫在 `api_spec` |
| `db_design` | 沒有關聯式資料庫。持久化只有 Feast 的 SQLite online store 與 MLflow 的 SQLite backend，資料架構寫在 [`sad.md` §6](architecture/sad.md#6-資料架構) |
| `uat_plan` | 沒有客戶驗收輪次。品質把關由 CI 的自動門檻負責，見 [`test_plan.md`](qa/test_plan.md) |
| `deployment_and_operations` | 部署拓撲寫在 [`sad.md` §7](architecture/sad.md#7-部署視圖)；日常操作寫在 `../README.md` 的 Quickstart。等真的有多環境（stage/prod）再單獨成篇 |

> 模板庫自己的規則就是「**不按序填滿**、只讀與當前範圍相關的章節」。
> 這份清單是那條規則的執行結果，不是偷懶。

---

## 3. 讀的順序

初次接觸這個專案：

1. [`../README.md`](../README.md) — 它是什麼、怎麼跑起來
2. [`architecture/sad.md`](architecture/sad.md) — 系統長什麼樣
3. [`architecture/adr/`](architecture/adr/) — 為什麼長這樣（**教學價值最高**）
4. [`design/lld.md`](design/lld.md) — 要改程式前先看：模組依賴、狀態機契約
5. 要接 API → [`design/api_spec.md`](design/api_spec.md)；要維運 → [`ops/`](ops/)

---

## 4. 已知的邊界

這是參考實作。以下是**目前的真實邊界**，文件不會把它們說成生產就緒：

| 位置 | 現況 | 影響 |
| :--- | :--- | :--- |
| 三個模型的訓練資料 | 玩具 / 合成資料 | 指標不具業務意義（vision 的 f1 常為 0，品質門檻會正確擋下註冊）|
| `infra/terraform/` | **示意性 IaC**：用 `local`/`random` provider 產出資源描述檔，非真實雲端資源 | 結構、變數化、秘密處理都可用；換成雲商 provider 才會真的開資源 |
| `resolve_model` 的最終後援 | registry 完全不可用時落到硬編 URI `models:/smart-factory/latest` | 真實部署應在此失敗而非猜測（[ADR-005](architecture/adr/ADR-005-deployment-placeholders.md)）|
| canary 探測 | 驗的是**服務健康**，不是模型品質 | 載得起來但預測很爛的模型仍會通過；模型品質由訓練期門檻負責 |

接上真實資料源、換成雲商 provider——這些不是 bug fix，是學員的 Capstone 練習範圍。

> **已解除的舊限制**（v1 曾列在這裡）：`canary_probe` 寫死 `1.0` 已改為真實 HTTP 探測；
> 三個 `soft_import` 指向的缺失模組已補齊。另外 v1 曾誤稱「`infra/terraform/` 只有 README」——
> 那是錯的，`.tf` 檔一直都在，是當初用只搜 `*.md` 的掃描做了錯誤推論。
