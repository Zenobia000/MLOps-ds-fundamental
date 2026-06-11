# M1 · Foundations（階 0：純 Python + Git）

## 1. 本模組學什麼

用你最熟悉的 sklearn 訓一個 baseline（iris + LogisticRegression），並用 Git 把它版控起來，建立貫穿整門課的核心心智：**可重現（reproducibility）**。
對應技能階梯 **階 0**——本模組**不引入任何新工具**，只把「baseline + Git + 固定 seed」這三件事練熟，作為之後 MLflow / Optuna / DVC… 的乾淨起點。

---

## 2. 沙盒步驟（Layer 1：跑 `sandbox/`）

到本模組的 `sandbox/`，照編號逐個跑，一次只學一個最小可用動詞：

```bash
cd modules/m1-foundations/sandbox
python 01_baseline_iris.py
```

`01_baseline_iris.py` 做的事（也只做這些）：

1. 讀 `datasets/iris.csv`（玩具資料，越無聊越好）。
2. `train_test_split` 並**設 seed**（`random_state=42`）。
3. 訓 `LogisticRegression`，在 test set 印出 accuracy。

**驗收重點**：把腳本跑兩次，accuracy 必須**一模一樣**。
若兩次不同，代表某處的隨機性沒被固定——這正是本模組要你體會的「可重現」。
這支腳本**完全不碰 MLflow**，先把純 sklearn 的手感建立起來。

---

## 3. 整合任務（Layer 2：到 `workspace/` 接上去）

`workspace/` 是你**跨模組會一直長大**的主線專案。本模組的任務：把上面的 baseline 搬進 `workspace/`，當作整條 pipeline 的**主線起點**。之後每個模組的 B 段，都會在這份 baseline 上「接一個新工具」。

到 `workspace/` 建立你的第一版訓練腳本（TODO 提示）：

```text
workspace/
└── train.py          # ← 你要建立的主線訓練腳本
```

`train.py` 需要做到（把沙盒學到的搬過來，並稍微「專案化」）：

- [ ] **TODO**：讀 `datasets/iris.csv`（沿用沙盒的相對路徑寫法，別把資料複製進 workspace）。
- [ ] **TODO**：`train_test_split` 設 `random_state=42`（seed 集中成一個常數，方便日後統一管理）。
- [ ] **TODO**：訓 `LogisticRegression` 並印出 test accuracy（先求能跑、能重現，不求最佳）。
- [ ] **TODO**：把這份 baseline 用 Git 版控起來（見下方 cheatsheet）——先開分支，再 commit。
- [ ] **TODO（預留接口）**：在訓練結束處留一行註解 `# M2: 之後在這裡接 MLflow log_*`，標好下一個工具的接點。

> 為什麼搬進 workspace？沙盒是「練工具、可丟可重來」；workspace 是「你真正長大的成品」。兩者實體分開，避免把「練習」和「組系統」搞混。

### 極簡 Git 工作流 cheatsheet

階 0 只需要四個動詞就能把 baseline 安全版控起來：

```bash
git status                      # 我現在改了什麼？（動手前後都先看一眼）
git branch feat/m1-baseline     # 開一條功能分支（鐵律：別在 main 上直接改）
git switch feat/m1-baseline     # 切到該分支（舊版 Git 用 git checkout）
git add workspace/train.py      # 把要納管的檔案放進暫存區
git commit -m "feat(m1): add iris baseline as workspace mainline"
```

| 動詞 | 一句話 | 什麼時候用 |
| :--- | :--- | :--- |
| `git status` | 看工作區現況 | 任何動作前後都先看 |
| `git branch <name>` | 建立分支 | 開始一個新任務時 |
| `git add <file>` | 把變更放進暫存區 | commit 前挑選要納管的檔案 |
| `git commit -m "..."` | 把暫存區存成一個版本 | 完成一個可獨立 review 的小步 |

> 鐵律：**先開分支，再動程式碼**。永遠不要在 `main` 上直接改。commit message 用祈使句、講清楚 WHY，想像一個沒看過此 repo 的人在讀。

---

## 4. 卡住怎麼辦

- 還沒有上一個模組，本模組是起點——若 `workspace/` 被你改亂了，用 M1 結束的乾淨快照重置：

  ```bash
  cp -r checkpoints/after-m1/* workspace/
  ```

- 沙盒腳本若報錯：
  - `FileNotFoundError: iris.csv` → 確認 `mlops-course/datasets/iris.csv` 存在；路徑由腳本自動往上找，不必手動改。
  - `ModuleNotFoundError: sklearn` → 先裝課程依賴（見 repo 根的 `SETUP.md` / `pyproject.toml`）。

- 完成本模組後，拿你的 `workspace/` 對照 `checkpoints/after-m1/`，確認結果一致，再進 M2。

---

## 5. 檢核題（自我確認）

1. 為什麼要設 `random_state` / seed？不設會發生什麼？
2. `train_test_split` 把資料切成 train / test 各做什麼用？為什麼 accuracy 要在 **test** 上算？
3. 「先開分支再改碼」的鐵律解決了什麼問題？直接在 `main` 上 commit 有何風險？
4. 沙盒（`sandbox/`）與主線（`workspace/`）的差別是什麼？baseline 為什麼要搬進 `workspace/`？
5. 一個好的 commit message 應該講清楚什麼？（提示：WHY 比 WHAT 重要）
