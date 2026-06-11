# 教學排序設計：單點精熟 → 漸進整合 → 完整專案

> **問題**：原 capstone 大綱以「完整專案視角」鋪陳，會在課程一開始就攤開大量尚未學過的工具（MLflow + Feast + BentoML + Prefect + Actions + Evidently），造成初學者**認知超載**——還不會用任何一個，卻被要求把六個接起來。
> **原則**：完整專案是**專家的心智模型**，不是**初學者的學習路徑**。本文件把課程重排為漸進式，降低每一步的認知負荷。
> **參照**：本 repo 的 `MLflow/basic/` 01→16 號 notebook 已示範正確做法——一次只加一個概念。本設計把同樣原則推廣到所有工具。

---

## 1. 教學三定律

| 定律 | 內容 | 反例（要避免） |
| :--- | :--- | :--- |
| **一次一工具** | 每個練習只引入**一個**新工具，其餘環境都已熟悉 | 第一堂就 Feast + MLflow + Docker 一起上 |
| **玩具資料先行** | 新工具的初次接觸用**最小玩具資料**（iris / 單一 CSV），把注意力留給「工具怎麼用」 | 一開始就用智慧工廠三資料集，學生忙著理解資料而非工具 |
| **延後整合** | 整合（把工具接起來）**只在最後**做；前面都是孤立沙盒 | 每個模組都要求接回主專案 |

> 核心：**先會用（usage）→ 再會接（integration）**。把「學工具」和「組系統」拆成兩件事，不要混在一起教。

---

## 2. 三層學習法

```
┌─ Layer 1：單工具沙盒（Atomic Sandbox）────────────────────┐
│  每個工具一個 10–30 行的最小範例，玩具資料，零整合         │
│  目標：認識這個工具的「核心動詞」與心智模型               │
│  產出：能獨立跑起來、能說出它解決什麼問題                 │
└──────────────────────────────────────────────────────────┘
                      ↓ 熟了才往上
┌─ Layer 2：漸進整合（Incremental Integration）────────────┐
│  每次只把「一個已學會的工具」接到既有成品上               │
│  目標：理解工具之間的介面與資料流                         │
│  產出：一條逐步長大的 pipeline                            │
└──────────────────────────────────────────────────────────┘
                      ↓ 接順了才收尾
┌─ Layer 3：完整專案（Capstone）───────────────────────────┐
│  智慧工廠端到端。此時所有工具都已熟悉，重點是「設計與權衡」 │
│  目標：自己決定怎麼組、為什麼這樣組                       │
│  產出：能跑、能維運、能講清楚決策的系統                   │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 工具技能階梯（每一階只加一個新工具）

> 順序依「依賴關係 + 認知負荷」排定。每一階假設前面都已熟悉。

| 階 | 新工具 | 玩具沙盒練習（Layer 1） | 既有基礎 |
| :--- | :--- | :--- | :--- |
| 0 | 純 Python + Git | sklearn 訓一個 baseline，commit | — |
| 1 | **MLflow** | 同一支腳本加 `log_param/metric/model`，看 UI | 階 0 |
| 2 | **Optuna（HPO 自動調參）** | 玩具資料寫 `objective`，`study.optimize` 自動找超參；**每個 trial = 一個 MLflow run** | 階 1 |
| 3 | **DVC** | 對單一 CSV `add/push/checkout` | 階 1 |
| 4 | **Feast** | 玩具資料定 entity + feature view，取 historical/online | 階 1 |
| 5 | **Docker** | 把訓練腳本 `build` 成 image 跑起來 | 階 0 |
| 6 | **FastAPI** | 一個 `POST /predict` 包既有模型 | 階 0 |
| 7 | **BentoML** | 同一個模型改用 Bento service（對比 FastAPI 差異） | 階 6 |
| 8 | **PyTorch 服務** | 預訓練 ResNet → ONNX → 用 Bento 服務 | 階 7 |
| 9 | **Prefect/ZenML** | 把兩支既有函式串成一個 flow | 階 1 |
| 10 | **GitHub Actions** | push 觸發 `pytest` 一條 workflow | 階 0 |
| 11 | **Evidently** | 兩個 DataFrame 算 drift report | 階 0 |
| 12 | **整合** | 上述全部接成智慧工廠 pipeline | 全部 |

> 關鍵：**到階 12 才第一次「整合」**。在那之前，學生對每個工具都已有獨立的成功經驗，整合時面對的是「接線」而非「同時學六樣新東西」。
>
> **為什麼 Optuna 緊接 MLflow（階 2）**：HPO 的本質是「跑很多次實驗找最佳超參」，而每個 trial 就是一個 MLflow run——兩者天生互補。先學 MLflow 才有地方記錄 trial，再學 Optuna 才有東西自動產生大量 run。你 repo 的 `autoML_template` 正是這兩者的結合（HPO 是 AutoML 的引擎）。

---

## 4. 每個工具的「最小可用單元」（先只教這幾個動詞）

降低負荷的具體手法：**初次接觸只教 3–5 個核心動詞，進階功能全部延後**。

| 工具 | 先只教（最小可用） | 明確延後（之後再回來） |
| :--- | :--- | :--- |
| **MLflow** | `start_run` / `log_param` / `log_metric` / `log_model` / UI 看 run | registry alias、custom pyfunc、`evaluate`、signature 強制 |
| **Optuna（HPO）** | `create_study` / `objective(trial)` + `trial.suggest_*` / `optimize(n_trials)` / `best_params` | pruner（ASHA/Hyperband）、多目標（accuracy vs latency）、分散式（Ray Tune）、AutoML 框架（FLAML/AutoGluon） |
| **DVC** | `init` / `add` / `push` / `checkout` | `dvc.yaml` pipeline、remote 細節、實驗管理 |
| **Feast** | `apply` / `get_historical_features` / `materialize` / `get_online_features` | on-demand FV、streaming、push source |
| **Docker** | `build` / `run` / `-p` 埠對應 / volume | compose、multi-stage、healthcheck |
| **FastAPI** | 一個 `POST /predict` | async、middleware、認證 |
| **BentoML** | service + runner + `bentofile` + `serve` | adaptive batching、Yatai、雲端部署 |
| **Prefect** | `@flow` / `@task` / 本地 run | deployment、schedule、blocks |
| **GitHub Actions** | `on: push` → 跑 `pytest` | matrix、cache、OIDC、self-hosted |
| **Evidently** | `DataDriftPreset` 報告（ref vs current） | test suite、監控服務、dashboard |

> 教學話術：「**今天 Feast 你只要會四個動作就能做事，其他等你需要時我們再回來。**」——明確告訴學生「不用一次學完」，本身就是降負荷。

---

## 5. 重排後的 12 小時：每模組 A/B 兩段

每個 2h 模組拆成 **A 段（單工具沙盒）+ B 段（接回上一個成品）**，最後一模組才完整整合。

| 模組 | A 段：單工具沙盒（玩具資料） | B 段：漸進整合（接既有成品） |
| :--- | :--- | :--- |
| **M1 全景** | Git + 純 sklearn baseline（無新工具） | 建專案骨架、跑通環境 |
| **M2 MLflow + 調參** | iris 上練 `log_*` 與 UI（階 1）；Optuna 自動調參、每 trial 一個 run（階 2） | 把追蹤＋調參接到 baseline；加 DVC 版本化資料（階 3） |
| **M3 Feast** | 玩具資料練 feature view 與 point-in-time（階 4） | 把特徵餵給 M2 的訓練流程 |
| **M4 服務化** | FastAPI→Docker→BentoML 三段對比（階 5–7）；PyTorch 服務（階 8） | 把 M2/M3 的模型包成可呼叫 API |
| **M5 自動化** | Prefect 串兩函式（階 9）；Actions 跑 pytest（階 10） | 把 M3/M4 的步驟編成 pipeline + CI gate |
| **M6 監控+收尾** | Evidently 算 drift report（階 11） | **Layer 3 完整整合**：閉環 + 治理 + Capstone |

> 觀察：**整合動作集中在每模組 B 段、且逐模組加重，到 M6 才完整**。A 段永遠是「孤立、玩具、低負荷」，保護學生的初次接觸體驗。

---

## 6. 玩具資料 → 真實資料的切換時機

| 階段 | 用什麼資料 | 為什麼 |
| :--- | :--- | :--- |
| Layer 1 沙盒 | iris / 單一小 CSV / 你的 `diabetes.csv` | 資料越無聊越好，注意力全留給工具 |
| Layer 2 整合 | 智慧工廠**單一**子場景（先只結構化） | 開始面對真實資料形狀，但仍只有一條線 |
| Layer 3 Capstone | 智慧工廠三型態（結構化/時序/影像） | 最後才面對完整複雜度 |

> `diabetes.csv` 你已經有了——很適合當 Layer 1 的玩具資料貫穿 MLflow 與 Feast 沙盒，學生不必每個工具換一份資料。

---

## 7. 認知負荷檢核表（設計每個 Lab 時自問）

- [ ] 這個 Lab 是否**只引入一個**新工具？
- [ ] 初次接觸是否用**玩具資料**，而非真實複雜資料？
- [ ] 是否只教該工具的**最小可用動詞**，延後進階功能？
- [ ] 學生在這步**之前**，是否對所有其他元件都已有成功經驗？
- [ ] 「整合」是否被推遲到學生已熟悉各零件之後？
- [ ] 是否明確告訴學生「現在不用學完，之後會回來」？

---

## 8. 與其他文件的關係

- 本文件**重排**了 [mlops-course-outline.md](./mlops-course-outline.md) 的 Lab 順序與資料策略；觀念骨架與模組主題不變。
- [project-structure.md](./project-structure.md) 的完整資料夾**只在 Layer 3 才全部出現**；Layer 1/2 階段學生只接觸當前工具對應的那一兩個資料夾。
- 建議授課時：**先發 Layer 1 沙盒的最小 repo，每模組才逐步「解鎖」更多資料夾**，避免一開始就看到完整骨架而焦慮。
