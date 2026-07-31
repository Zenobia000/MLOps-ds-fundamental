# 介質選擇：什麼該是 notebook，什麼該是 script

> **這份文件回答什麼**：本課程 6 個模組、33 個沙盒單元，哪些用 `.ipynb`、哪些用 `.py`／`.md`，以及**為什麼**。
> **狀態**：已實作（4 本 notebook 已建立，其餘維持腳本）。
> **相關**：[`project-structure.md`](project-structure.md)（`notebooks ≠ src` 原則）、[`teaching-progression.md`](teaching-progression.md)（技能階梯）。

---

## 1. 問題

課程大綱把「把模型**從 notebook 變成**服務」寫成 Module 4 的學習目標，於是教學骨架 `mlops-course/modules/` 一開始全部採用 `.py` 腳本，一本 notebook 都沒有。

但這對**資料科學家**的學習體驗是有代價的：表格資料的探索、特徵建構、分布漂移——這些的理解**來自看見**。用 `print()` 吐一面文字牆去教「時間穿越」，是把該用圖說的事情硬寫成散文。

兩邊都對，但它們講的不是同一件事。需要一個能同時容納兩者的判準。

---

## 2. 判準

不是照**資料型態**切（「表格用 notebook、影像用腳本」），也不是照**模組編號**切（「前三個 notebook、後三個腳本」）。兩種切法都會切錯。真正的分界是：

> ### 人在看資料 → notebook；機器無人值守在跑 → 檔案。

展開成可判定的條件：

**選 notebook，當這個單元同時滿足：**

1. **學習對象是資料本身**——dataframe、分布、時間軸，而不是一個服務或一條流程。
2. **理解來自看見中間狀態**——尤其是「錯誤做法 vs 正確做法」的並排對照。
3. **產物是拋棄式的**——重跑一次就好，不需要有人 import 它、不需要有檔案留在特定路徑。

**選檔案（`.py` / `.yml` / `.md`），當出現任一條：**

1. **產物必須是某個路徑上的檔案**——`uvicorn app:app`、`bentoml serve service.py`、`pytest` 的 `test_*.py`、Docker build context。notebook 當不了這個交付物。
2. **執行的是長跑行程**——server 會卡住 cell，`--reload`、埠口生命週期在 notebook 裡整個失真。
3. **教學重點是 CLI 與檔案系統狀態機**——`dvc init/add/checkout` 搭配 `git commit`，**工作目錄的狀態就是要教的東西**；包進 `!` cell 反而把它藏起來。
4. **這一課的主題與 notebook 的弱點正面衝突**——見下節。

---

## 3. 一個特別的排除理由：自打嘴巴

M2 的主題是**可重現性**。notebook 最有名的失敗模式正是可重現性：隱藏狀態、亂序執行、「在我機器上跑得動」。

用一個「亂序執行會給出不同答案」的介質，去教「今天跑、明天跑、別人跑都要一樣」，教學訊息會自我矛盾。所以 **M2 的 MLflow 段刻意留在腳本**——而且這個對比值得在課堂上明講，它本身就是一課。

同理，MLflow 的 run lifecycle 在 notebook 是經典陷阱：cell 執行到一半報錯會留下未關閉的 run，下一次 `start_run()` 就錯誤地巢狀進去。教學現場很難除錯。

---

## 4. 逐單元裁決

### 用 notebook（4 本）

| 模組 | Notebook | 為什麼是 notebook |
| :--- | :--- | :--- |
| **m1** | `sandbox/00_eda_iris.ipynb` | DS 動手的第一件事是看資料：`head()`、類別平衡、特徵分布、相關性。原本 `01_baseline_iris.py` 對這段只有一行 `讀到資料：150 筆、3 個類別`。 |
| **m2** | `sandbox/optuna/04_visualization.ipynb` | `optuna.visualization` 的 optimization history／param importances／parallel coordinate 是 Plotly 圖，inline 直出。DS 靠這幾張圖決定搜尋範圍該怎麼調。 |
| **m3** | `sandbox/02_leakage_viz.ipynb` | **本課最強的 notebook 案例**。point-in-time 洩漏是天生的對照式教材：錯誤 join → 指標虛高 vs 正確 join → 誠實指標。原腳本只能 `print(df.to_string())`，docstring 被迫用 15 行散文解釋「時間穿越」——那是圖該做的事。 |
| **m6** | `sandbox/evidently/drift_explore.ipynb` | Evidently 本來就產 HTML 報告，原本要存檔再切瀏覽器開。notebook inline 顯示嚴格更好，少一次 context switch；而漂移分析本質就是資料分析。 |

### 維持腳本／指令（其餘全部）

| 模組 | 單元 | 擋點 |
| :--- | :--- | :--- |
| **m1** | `01_baseline_iris.py` | 主題是「固定 seed、可重現」，是 notebook 的反面教材——刻意保留成對照組。 |
| **m2** | `mlflow/01–03` | run lifecycle 陷阱；且模組主題就是可重現性（見 §3）。 |
| **m2** | `dvc/01_version_a_csv.md` | CLI + git 的工作目錄狀態就是教學內容。 |
| **m3** | `00_prepare_data.py` | 產出 parquet 給 `feast apply` 吃，是檔案交付物。 |
| **m3** | `feature_repo/feature_definition.py` | **Feast 必須 import 這個模組**，它不能是 notebook。 |
| **m4** | `01_fastapi/app.py` | `uvicorn app:app` 需要模組路徑；server 阻塞 cell。 |
| **m4** | `02_docker/Dockerfile` | build context 需要真實檔案。 |
| **m4** | `03_bentoml/service.py` | `bentoml serve service.py` 需要檔案。 |
| **m4** | `04_pytorch_onnx/*.py` | `.pt` → ONNX → serve 是檔案鏈。 |
| **m5** | `prefect/flow.py` | flow 存在的意義就是被排程執行。 |
| **m5** | `github-actions/ci.yml` | YAML，給 runner 跑。 |
| **m5** | `tests/test_sample.py` | pytest 要 `test_*.py` 才會被發現。 |
| **m6** | `evidently/drift_report.py` | 保留成「同一個分析進 CI 自動跑」的腳本版（見 §5）。 |
| **m6** | `governance/*.md` | 是文件填空練習，不是程式。 |

---

## 5. 分水嶺在 M4，而那正是課程要講的話

大綱 Module 4 的學習目標原文：「把模型**從 notebook 變成**別人能用 HTTP 呼叫的服務」。

所以 **M4 之後不用 notebook 不是工具限制，是課程訊息本身**。學員在 m1–m3 用 notebook 探索得很舒服，走到 m4 被要求交出 `app.py`——那個摩擦感是刻意的，它就是「上線」的體感。

但**不是乾淨的 m1–m3 vs m4–m6**。**m6 會盪回來**：漂移分析本質是資料分析（人在看），所以用 notebook；但「CI 裡自動跑的漂移檢查」還是腳本。同一個模組並存兩種介質，這個對比可以直接拿來教：

```
m6/sandbox/evidently/
├── drift_explore.ipynb   ← 人在看：調參數、換欄位、觀察分布怎麼被推開
└── drift_report.py       ← 機器在跑：同一套邏輯，進 CI 當自動門檻
```

一句話總結給學員：**「你在 notebook 想清楚要監控什麼，然後把結論寫成腳本讓機器每天替你看。」**

---

## 6. 工程前提（notebook 進 repo 的代價）

notebook 不是免費的。導入前補齊三件事，否則會侵蝕既有的品質保證：

| 風險 | 對策 | 落在哪 |
| :--- | :--- | :--- |
| **outputs 進 git**，diff 變髒、`checkpoints/` 重置機制受污染 | `nbstripout --verify` 在 CI 擋下帶 output 的 notebook | `.github/workflows/ci.yml`、`make nb-strip` |
| **notebook 不被測**，「verified commands」的保證在 notebook 上破功 | `pytest --nbmake` 實際執行每一本 | `.github/workflows/ci.yml`、`make nb-test` |
| **cell 重跑的副作用**（m3 原本在 import 時 `os.chdir`） | 改成 scoped context manager，離開就還原 | `m3/sandbox/01_point_in_time_demo.py` |

> 教學 repo 的 notebook 必須**每一本都能從頭到尾重跑**。做不到這點的 notebook，正是它自己在示範的壞味道。

---

## 7. 決策摘要

- 判準是**誰在看**，不是資料型態，也不是模組編號。
- 4 本 notebook：m1 EDA、m2 Optuna 視覺化、m3 洩漏對照、m6 漂移探索。
- 其餘 29 個單元維持腳本／指令／文件。
- M2 的 MLflow 段**刻意**不用 notebook，理由是教學一致性而非技術限制。
- 介質選擇本身就是 MLOps 的一課，在各模組 README 明講，不要讓學員以為是隨意的。
