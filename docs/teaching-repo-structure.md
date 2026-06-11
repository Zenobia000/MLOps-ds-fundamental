# 教學 Repo 資料夾結構（依學習順序組織）

> **核心差異**：教學 repo 與生產 repo 的組織原則**相反**。
> - [project-structure.md](./project-structure.md)（生產）按**架構**組織——同類放一起（所有 model 在 `src/models/`），方便維運。
> - **本文件（教學）按學習順序組織**——同一個「學習步驟」的東西放一起，且**一次只暴露一個工具**，方便循序漸進。
>
> 本結構是 [teaching-progression.md](./teaching-progression.md) 三層學習法（沙盒 → 漸進整合 → 完整專案）的**實體落地**。

---

## 1. 設計目標

| 目標 | 對應的結構決策 |
| :--- | :--- |
| **一次一工具** | 每模組獨立資料夾，`sandbox/` 內的範例彼此不 import，零跨檔依賴 |
| **路徑即順序** | 模組用 `m1-`…`m6-` 編號，學生看路徑就知道進度（仿你的 `MLflow/basic/` 01–16） |
| **孤立的初次接觸** | Layer 1 沙盒用 `datasets/` 共用玩具資料，可獨立丟掉重來 |
| **整合是逐步長大的** | Layer 2 只有**一個** `workspace/`，跨模組累積，學生親眼看專案長大 |
| **容錯/可補課** | `checkpoints/` 存各模組結束的已知良好快照，卡住或缺課可一鍵重置 |
| **延後完整複雜度** | `capstone/` 獨立且「最後才解鎖」，前期不讓學生看到完整骨架而焦慮 |

---

## 2. 完整教學 Repo 樹

```
mlops-course/
├── README.md                       # 學習地圖 + 三層說明 + 如何逐步解鎖
├── SETUP.md                        # 環境安裝、疑難排解
├── Makefile                        # make m2 / make checkpoint-m2 / make capstone-up
├── pyproject.toml                  # 全課依賴（一次裝好，避免每模組重裝的摩擦）
│
├── datasets/                       # 共用玩具資料（Layer 1 專用）............ [一次一資料]
│   ├── iris.csv
│   ├── diabetes.csv                # 你已有，沿用
│   └── toy_sensors.csv             # 極簡時序樣本
│
├── modules/                        # ★ 教學主體，依上課順序排列
│   ├── m1-foundations/
│   │   ├── README.md               # 目標／步驟／沙盒指引／整合任務／檢核題
│   │   └── sandbox/                # Layer 1：純 sklearn baseline + Git（無新工具）
│   │
│   ├── m2-tracking-tuning-versioning/  # 階 1 MLflow + 階 2 Optuna + 階 3 DVC
│   │   ├── README.md
│   │   └── sandbox/
│   │       ├── mlflow/
│   │       │   ├── 01_first_run.py        # 仿你的 01–16 漸進
│   │       │   ├── 02_params_metrics.py
│   │       │   └── 03_log_model.py
│   │       ├── optuna/                    # 自動調參，每 trial = 一個 MLflow run
│   │       │   ├── 01_objective_basic.py  # study + objective + optimize
│   │       │   ├── 02_mlflow_callback.py  # trial 自動寫進 MLflow
│   │       │   └── 03_pruning_asha.py     # 提早砍爛 trial（選配多目標）
│   │       └── dvc/
│   │           └── 01_version_a_csv.md
│   │
│   ├── m3-feature-store/           # 階 4 Feast
│   │   ├── README.md
│   │   └── sandbox/                # 用 diabetes.csv 練 entity / feature view / point-in-time
│   │
│   ├── m4-serving/                 # 階 5–8
│   │   ├── README.md
│   │   └── sandbox/
│   │       ├── 01_fastapi/         # 先最熟悉的
│   │       ├── 02_docker/          # 再包起來
│   │       ├── 03_bentoml/         # 再升級 ML 原生
│   │       └── 04_pytorch_onnx/    # 最後 CV + ONNX
│   │
│   ├── m5-automation/              # 階 9 Prefect + 階 10 GitHub Actions
│   │   ├── README.md
│   │   └── sandbox/
│   │       ├── prefect/
│   │       └── github-actions/
│   │
│   └── m6-monitoring-governance/   # 階 11 Evidently + 治理 + 收尾
│       ├── README.md
│       └── sandbox/
│
├── workspace/                      # ★ Layer 2：學生的漸進整合主線（跨模組累積長大）
│   └── README.md                   # 「你的成品長在這裡；每模組 B 段往這裡加一個工具」
│
├── checkpoints/                    # 各模組結束的 workspace 已知良好快照（救援/補課）
│   ├── after-m1/
│   ├── after-m2/
│   ├── after-m3/
│   ├── after-m4/
│   └── after-m5/
│
├── capstone/                       # ★ Layer 3：完整智慧工廠（= project-structure.md）
│   └── smart-factory-mlops/        # 最後一模組才解鎖
│
└── docs/                           # 講師教材、投影片、本規劃系列
```

---

## 3. 三層 ↔ 資料夾的對應

| 學習層 | 住在哪 | 性質 | 學生心態 |
| :--- | :--- | :--- | :--- |
| **Layer 1 單工具沙盒** | `modules/mN/sandbox/` | 孤立、玩具資料、可丟可重來 | 「我在學這一個工具怎麼用」 |
| **Layer 2 漸進整合** | `workspace/`（單一、累積） | 跨模組長大的真主線 | 「我把學會的工具接到我的專案」 |
| **Layer 3 完整專案** | `capstone/smart-factory-mlops/` | 生產級完整骨架 | 「我自己決定怎麼組、為什麼」 |

> 關鍵：**Layer 1 用 `sandbox/`（多個孤立小範例），Layer 2 用 `workspace/`（一個長大的專案）**。兩者實體分開，學生不會把「練工具」和「組系統」搞混。

---

## 4. 每模組統一內部結構（一致性也是降負荷）

每個 `modules/mN-*/` 長得一樣,學生不必每模組重新適應結構:

```
mN-<主題>/
├── README.md          # 五段固定格式（見下）
└── sandbox/           # Layer 1：本模組工具的最小可跑範例（玩具資料）
```

**README.md 固定五段**（每模組同一模板）：

1. **本模組學什麼**（1–2 句 + 對應技能階梯第幾階）
2. **沙盒步驟**（Layer 1：照 `sandbox/` 的編號逐個跑，只學最小可用動詞）
3. **整合任務**（Layer 2：到 `workspace/` 把這個工具接上去，含 TODO 提示）
4. **卡住怎麼辦**（指向 `checkpoints/after-m(N-1)/` 重置 + 對照下一個 checkpoint）
5. **檢核題**（3–5 題自我確認）

> 解答策略：沙盒範例**本身就是可跑的正解**（學生照打、改參數）；整合任務的正解 = `checkpoints/after-mN/`。不另設 `solution/`,減少資料夾數量。

---

## 5. 「漸進解鎖」機制：學生怎麼前進

降低焦慮的關鍵是**不要一開始就看到全部**。三種由簡到繁的做法,擇一:

| 機制 | 做法 | 適合 |
| :--- | :--- | :--- |
| **A. 純資料夾紀律（推薦入門）** | README 只引導到「當前模組」;`capstone/` 開課時說明「最後才碰」 | 最簡單,零工具負擔 |
| **B. Makefile 閘門** | `make m3` 才複製 m3 教材到 workspace;`make capstone-up` 才起完整環境 | 想要儀式感與防呆 |
| **C. Git 分支/標籤（進階）** | 每模組一個 `checkpoint/after-mN` tag,`git checkout` 切換 | 想順便教 Git 工作流 |

> 共通原則:**`workspace/` 跨模組只有一份、持續長大;`checkpoints/` 是它在各時間點的存檔**。學生卡住就 `cp -r checkpoints/after-m2/* workspace/` 重置,不影響別人。

---

## 6. 學生的旅程(走一遍資料夾)

```
開課    cd modules/m1-foundations     → 讀 README → 跑 sandbox(純 sklearn)
        → 到 workspace 建 baseline    → 對照 checkpoints/after-m1 確認

M2      cd modules/m2-tracking-...    → 跑 sandbox/mlflow/01→03(玩具資料)
        → 回 workspace 把追蹤接到 baseline + DVC 版本化
        → 卡住就 checkpoints/after-m2 重置

M3–M5   同樣節奏:sandbox 練單工具 → workspace 接上去 → checkpoint 確認

M6      cd modules/m6-...             → Evidently 沙盒
        → 解鎖 capstone/              → 此時所有工具都熟,專注「設計與權衡」
```

每一步:**先在 `sandbox/` 用玩具資料把工具玩熟,再回 `workspace/` 接到自己長大的專案**。整合複雜度被推遲、被分攤。

---

## 7. 教學 Repo vs 生產 Repo（同樣的程式,不同的組織原則）

| 維度 | 教學 Repo（本文件） | 生產 Repo（project-structure.md） |
| :--- | :--- | :--- |
| **組織原則** | 依**學習順序**（時間軸） | 依**架構**（關注點分離） |
| **頂層資料夾** | `modules/m1…m6`、`workspace`、`capstone` | `src/`、`conf/`、`pipelines/`、`services/`… |
| **同一工具的程式** | 散在「學它的那一週」 | 收斂在它的職責資料夾 |
| **複雜度曝光** | 逐步解鎖、刻意延後 | 一次全攤開(維運者需要全局視野) |
| **最終交集** | `capstone/smart-factory-mlops/` **就是**生產結構 | 同一份 |

> 教學的終點 = 生產的起點。學生在 `capstone/` 第一次看到生產級結構時,因為每個零件都已親手用過,不再是負擔而是「啊,原來這些我都會」。

---

## 8. 命名與紀律(維護者守則)

- `modules/` 一律 `mN-<kebab-主題>`;路徑數字 = 上課順序。
- `sandbox/` 內的範例**禁止 import 其他模組**——保持孤立、可獨立執行。
- 玩具資料只放 `datasets/`,模組內**不複製**資料檔。
- 真實智慧工廠資料**只**出現在 `workspace/`(Layer 2 後期)與 `capstone/`(Layer 3)。
- 一個 `sandbox/` 範例只示範**一個**新動詞/概念;寧可多檔、每檔短。
- `checkpoints/after-mN/` 必須是「乾淨環境跑得起來」的快照——每次改教材要同步更新。

---

## 9. 與其他文件的關係

- 結構落地 [teaching-progression.md](./teaching-progression.md) 的三層學習法。
- `capstone/smart-factory-mlops/` 內部 = [project-structure.md](./project-structure.md) 的完整生產結構。
- 各模組教學內容對應 [mlops-course-outline.md](./mlops-course-outline.md) 的 M1–M6。
- 開發此教學 repo 的工時與順序見 [development-wbs.md](./development-wbs.md)(Phase 1 骨架先做 `capstone/`,再由它「降階」拆出各模組 sandbox)。
