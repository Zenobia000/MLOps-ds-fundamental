# MLOps 入門教學 Repo（學習地圖）

> **一句話定位**：用「一次一工具、玩具資料先行、延後整合」的漸進式路線，帶你從一支純 sklearn 腳本，逐步長成一條可追蹤、可調參、可版本化、可服務化、可自動化、可監控的 MLOps pipeline。

這個 repo **不是**一份攤開全部工具的生產骨架，而是一條**循序解鎖**的學習路徑。每一步只引入一個新工具，其餘環境都已熟悉，把你的注意力留給「這個工具怎麼用」。

---

## 1. 三層學習法（沙盒 → 漸進整合 → 完整專案）

整個課程把「學工具」和「組系統」**刻意拆成兩件事**，分三層推進：

| 層 | 住在哪 | 性質 | 你的心態 |
| :--- | :--- | :--- | :--- |
| **Layer 1 單工具沙盒** | `modules/mN-*/sandbox/` | 孤立、玩具資料、可丟可重來 | 「我在學這**一個**工具怎麼用」 |
| **Layer 2 漸進整合** | `workspace/`（單一、跨模組累積） | 一條逐步長大的真主線 | 「我把學會的工具**接到**我的專案」 |
| **Layer 3 完整專案** | `capstone/smart-factory-mlops/` | 生產級完整骨架，最後才解鎖 | 「我自己決定**怎麼組、為什麼**」 |

> 核心節奏：**先在 `sandbox/` 用玩具資料把工具玩熟（會用）→ 再回 `workspace/` 接到自己長大的專案（會接）→ 最後在 `capstone/` 端到端整合（會設計）。**
>
> - **Layer 1 是多個孤立小範例**：每個檔案能獨立執行、彼此不 import、零跨檔依賴。卡住就整個丟掉重來，不影響任何人。
> - **Layer 2 只有一份 `workspace/`**：它跨模組持續長大，每個模組的整合任務都往這裡加**一個**工具，你會親眼看著專案變大。
> - **Layer 3 的 `capstone/` 最後才碰**：前期刻意不讓你看到完整骨架而焦慮；等所有零件都親手用過，整合時面對的是「接線」而非「同時學六樣新東西」。

---

## 2. 六模組總表（學什麼工具 + 對應技能階梯第幾階）

如果你還不熟各工具的應用面，先讀 [`modules/component-introduction.md`](./modules/component-introduction.md)，再回來照模組順序動手。

| 模組 | 主題 | A 段：單工具沙盒（玩具資料） | 對應技能階梯 |
| :--- | :--- | :--- | :--- |
| **m1** | 全景 / 基礎 | Git + 純 sklearn baseline（無新工具） | 階 0（純 Python + Git） |
| **m2** | 追蹤・調參・版本化 | MLflow `log_*` 與 UI → Optuna 自動調參 → DVC 版本化 CSV | 階 1 MLflow + 階 2 Optuna + 階 3 DVC |
| **m3** | 特徵商店 | Feast：entity / feature view / point-in-time / online | 階 4 Feast |
| **m4** | 服務化 | FastAPI → Docker → BentoML → PyTorch+ONNX | 階 5 Docker, 階 6 FastAPI, 階 7 BentoML, 階 8 PyTorch 服務 |
| **m5** | 自動化 | Prefect 串 flow → GitHub Actions 跑 pytest | 階 9 Prefect + 階 10 GitHub Actions |
| **m6** | 監控・治理・收尾 | Evidently 算 drift report → 治理 → 解鎖 Capstone | 階 11 Evidently + 整合收尾 |

> 技能階梯的完整定義見 `docs/`（`teaching-progression.md`）。每一階只加**一個**新工具，且假設前面都已熟悉。每個工具初次接觸**只教 3–5 個核心動詞**，進階功能全部延後——「現在不用學完，之後需要時會回來」。

### 沙盒裡的兩種介質：notebook 與腳本

沙盒不是清一色的 `.py`。有 4 個單元刻意做成 notebook，判準只有一句話：

> **人在看資料 → notebook；機器無人值守在跑 → 檔案。**

| Notebook | 在哪 | 為什麼是 notebook |
| :--- | :--- | :--- |
| `00_eda_iris.ipynb` | m1 | 分布、平衡、可分性——理解來自看見 |
| `04_visualization.ipynb` | m2 | Optuna 的互動圖是用來**決定下一輪搜尋範圍**的 |
| `02_leakage_viz.ipynb` | m3 | 時間穿越是對照式教材：洩漏 AUC 0.940 vs 正確 0.791 |
| `drift_explore.ipynb` | m6 | 漂移分析本質就是資料分析 |

其餘 29 個單元維持腳本／指令。**m4 是分水嶺**——它的主題正是「把模型**從 notebook 變成**服務」，
所以那之後不用 notebook 不是限制，是課程訊息本身。**m6 會盪回來**，是唯一兩種介質並存的模組。

逐一裁決與工程前提見 [`docs/notebook-vs-script.md`](../docs/notebook-vs-script.md)。
notebook 的品質由 CI 保證：`make nb-test` 實際執行每一本，`make nb-verify` 擋下帶輸出的檔案。

---

## 3. 如何「漸進解鎖」（不要一開始就看到全部）

降低焦慮的關鍵是**只看當前模組**：

1. **README 只引導當前模組**：照 `m1 → m2 → … → m6` 的順序，一次只打開一個 `modules/mN-*/`，讀它的 README、跑它的 `sandbox/`。不要提前跳去後面的模組。
2. **`workspace/` 跨模組只有一份**：每個模組的「整合任務」都往同一個 `workspace/` 加一個工具，你的成品在這裡持續長大。
3. **`checkpoints/` 是各時間點的存檔**：每個模組結束有一份已知良好快照（`checkpoints/after-mN/`）。卡住、缺課、改壞了，就一鍵重置（見下方「卡住怎麼辦」）。
4. **`capstone/` 最後才碰**：它是生產級完整骨架（= 智慧工廠端到端）。到 **m6** 才解鎖；在那之前刻意不展開，避免初學焦慮。等你走到這裡，每個零件都已親手用過，重點變成「設計與權衡」。

> 教學的終點 = 生產的起點：你在 `capstone/` 第一次看到生產級結構時，會發現「啊，原來這些我都會」。

---

## 4. 資料夾導覽

```
mlops-course/
├── README.md          # 本檔：學習地圖
├── SETUP.md           # 環境安裝、啟動 MLflow UI、無 GPU 後援
├── Makefile           # 常用指令封裝（make help 看全部）
├── pyproject.toml     # 全課依賴（一次裝好，避免每模組重裝的摩擦）
│
├── datasets/          # 共用玩具資料（Layer 1 專用）：iris.csv / diabetes.csv / toy_sensors.csv
│
├── modules/           # ★ 教學主體，依上課順序排列（路徑即進度）
│   ├── m1-foundations/                    # 純 sklearn baseline + Git
│   ├── m2-tracking-tuning-versioning/     # MLflow + Optuna + DVC
│   ├── m3-feature-store/                  # Feast
│   ├── m4-serving/                        # FastAPI / Docker / BentoML / PyTorch+ONNX
│   ├── m5-automation/                     # Prefect + GitHub Actions
│   └── m6-monitoring-governance/          # Evidently + 治理 + 收尾
│       每個 mN-*/ 內部一致：README.md（五段格式）+ sandbox/（最小可跑範例）
│       sandbox/ 內多為 .py 腳本；m1/m2/m3/m6 各有一本 .ipynb（見上方「兩種介質」）
│
├── workspace/         # ★ Layer 2：你的漸進整合主線（跨模組累積長大，只有一份）
│
├── checkpoints/       # 各模組結束的 workspace 已知良好快照（救援 / 補課）
│   ├── after-m1/  after-m2/  after-m3/  after-m4/  after-m5/
│
└── capstone/          # ★ Layer 3：完整智慧工廠（= 生產結構，最後一模組才解鎖）
    └── smart-factory-mlops/
```

| 資料夾 | 用途 | 什麼時候進去 |
| :--- | :--- | :--- |
| `datasets/` | 共用玩具資料 | Layer 1 沙盒一律從這裡讀，模組內**不複製**資料檔 |
| `modules/` | 教學主體，六模組 | 依 `m1→m6` 順序，一次一個 |
| `workspace/` | 你長大的專案 | 每個模組的整合任務都往這裡加工具 |
| `checkpoints/` | 已知良好快照 | 卡住時 `cp -r checkpoints/after-mN/* workspace/` 重置 |
| `capstone/` | 完整生產骨架 | **m6 才解鎖** |

### 指令從哪裡下

課程文件裡的指令有兩種起點，**搞混是最常見的卡關原因**：

| 指令 | 起點 | 例子 |
| :--- | :--- | :--- |
| `make ...`、`python workspace/...`、`cp -r checkpoints/...` | **`mlops-course/`**（放 `Makefile` 的這一層） | `make workspace-m1` |
| 沙盒範例（`python 01_*.py`、`uvicorn app:app`、`feast apply`） | **該沙盒自己的資料夾** | 先 `cd modules/m1-foundations/sandbox` |

規則就一句：**各模組 README 裡的路徑一律相對 `mlops-course/`**。
`cd` 進沙盒跑完之後，記得 `cd` 回 `mlops-course/` 再接下一段——
不回來的話，`make workspace-mN` 會因為那層沒有 `Makefile` 而失敗：

```bash
cd mlops-course                       # ← 每一段的起點
cd modules/m1-foundations/sandbox     # 進沙盒
python 01_baseline_iris.py
cd ../../..                           # ← 回到 mlops-course/（三層：sandbox → m1 → modules）
make workspace-m1                     # 這時才找得到 Makefile
```

> 隨時用 `pwd` 確認你在哪一層；`make help` 只有在 `mlops-course/` 下才跑得動，
> 可以拿它當「我站對地方了嗎」的快速檢查。

**Git 是唯一的例外。** git 的 repo 根目錄是**再上一層**的 `MLOps-ds-fundamental/`，
`mlops-course/` 只是它底下的一個子資料夾（沒有自己的 `.git`）。這代表：

- `git status` 會列出**整個 repo** 的變更，不只 `mlops-course/` 的——看到別的目錄出現在清單裡是正常的。
- git 指令**在任何一層下都可以**，但你給的檔案路徑是相對「你當下所在的位置」。
  在 `mlops-course/` 下 `git add workspace/train.py` 是對的；在 repo 根則要寫
  `git add mlops-course/workspace/train.py`。
- 開分支、commit 影響的是整個 repo，不是單一模組。

---

## 5. 從這裡開始

1. 先讀 **[SETUP.md](./SETUP.md)** 把環境一次裝好：

   ```bash
   cd mlops-course
   make setup       # uv 建 .venv + 依 uv.lock 裝全課依賴（Python 3.11 自動取得）
   make check-env   # 確認站對地方
   ```

   > 全課共用**一個** `.venv`，版本由 `uv.lock` 鎖死——你、同學、CI 拿到的是同一組版本。
   > 各模組沙盒**不要**再自己 `pip install`（m4 的兩份 `requirements.txt` 是容器映像用的，不是給你的環境）。
   > `make` 指令會自動用 `.venv`，不必手動 activate。

2. `make help` 看可用指令；`make m1` 印出第一個模組的 README 路徑。
3. `cd modules/m1-foundations`，讀它的 README，照五段格式走：學什麼 → 跑 sandbox → 整合任務 → 卡住怎麼辦 → 檢核題。
4. 走完一個模組就 `make checkpoint-mN` 存檔，再前進下一個。

> 卡住別慌：每個模組 README 都有「卡住怎麼辦」，指向對應的 `checkpoints/after-m(N-1)/` 一鍵重置。
