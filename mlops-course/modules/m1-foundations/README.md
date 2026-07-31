# M1 · Foundations（階 0：純 Python + Git）

## 1. 本模組學什麼

用你最熟悉的 sklearn 訓一個 baseline（iris + LogisticRegression），並用 Git 把它版控起來，建立貫穿整門課的核心心智：**可重現（reproducibility）**。
對應技能階梯 **階 0**——本模組**不引入任何新工具**，只把「baseline + Git + 固定 seed」這三件事練熟，作為之後 MLflow / Optuna / DVC… 的乾淨起點。

---

## 2. 沙盒步驟（Layer 1：跑 `sandbox/`）

> **指令起點**：本檔所有路徑都相對 **`mlops-course/`**（放 `Makefile` 的那一層）。
> `cd` 進沙盒跑完，記得 `cd` 回 `mlops-course/` 再接下一段。
> Git 是例外——repo 根在再上一層，詳見[課程 README「指令從哪裡下」](../../README.md#指令從哪裡下)。

到本模組的 `sandbox/`，照編號逐個跑，一次只學一個最小可用動詞：

```bash
cd mlops-course/modules/m1-foundations/sandbox   # 從 repo 根出發；已在 mlops-course/ 則去掉開頭

jupyter lab 00_eda_iris.ipynb   # 先「看」資料：分布、類別平衡、可分性
python 01_baseline_iris.py      # 再「固定」結果：可重現的 baseline

cd ../../..                     # ★ 回到 mlops-course/（sandbox → m1-foundations → modules）
```

### 2-0　為什麼這裡有一本 notebook、旁邊卻是腳本？

這是本課程第一次示範**介質選擇**，值得停下來想三十秒：

| | `00_eda_iris.ipynb` | `01_baseline_iris.py` |
| :--- | :--- | :--- |
| 誰在用 | **人**，互動式 | **機器**，無人值守 |
| 目的 | 看懂資料長什麼樣 | 產出今天/明天/別人跑都一樣的數字 |
| 為什麼是這個介質 | 理解來自**看見**——分布、重疊、相關性都是圖 | 主題是**可重現**，而亂序執行正是 notebook 最弱的地方 |

> 判準一句話：**人在看資料 → notebook；機器無人值守在跑 → 檔案。**
> 全課 33 個沙盒單元的逐一裁決見 [`docs/notebook-vs-script.md`](../../../docs/notebook-vs-script.md)。

`00_eda_iris.ipynb` 會帶你看四件事，每一件都直接影響下一步的決策：
類別平衡（決定 accuracy 能不能用）、特徵分布、花瓣散布圖（決定 baseline 選線性模型就夠）、相關性（花瓣長寬相關 0.96，特徵有冗餘）。

`01_baseline_iris.py` 做的事（也只做這些）：

1. 讀 `datasets/iris.csv`（玩具資料，越無聊越好）。
2. `train_test_split` 並**設 seed**（`random_state=42`）。
3. 訓 `LogisticRegression`，在 test set 印出 accuracy。

**驗收重點**：把腳本跑兩次，accuracy 必須**一模一樣**。
若兩次不同，代表某處的隨機性沒被固定——這正是本模組要你體會的「可重現」。
這支腳本**完全不碰 MLflow**，先把純 sklearn 的手感建立起來。

---

## 3. 整合任務（Layer 2：到 `workspace/` 接上去）

`workspace/` 是你**跨模組會一直長大**的主線專案。本模組：把 sandbox 學到的 baseline **填進主線模板**，當作整條 pipeline 的起點。

在 `mlops-course/` 解鎖填空骨架（**不要從空白檔硬寫**）：

```bash
pwd                 # 應該結尾是 /mlops-course，不是 .../sandbox
make workspace-m1
# 產生 workspace/train.py（已存在則跳過，不會蓋掉你的進度）
```

> 若出現 `make: *** No rule to make target` 或 `No such file or directory: Makefile`，
> 就是你還停在沙盒資料夾裡——`cd` 回 `mlops-course/` 再跑一次。

打開 `workspace/train.py`，搜尋 `TODO(M1-` 依序填（對照 `sandbox/01_baseline_iris.py`）：

- [ ] **TODO(M1-1)**：設定 `SEED`
- [ ] **TODO(M1-2)**：`find_iris_csv`（注意 `parents` 層數與 sandbox 不同）
- [ ] **TODO(M1-3)**：`load_iris`
- [ ] **TODO(M1-4～6)**：`train_test_split` → 訓練 → 印 test accuracy
- [ ] **Git**：先開分支再 commit（見下方 cheatsheet）
- [ ] 保留檔尾 `# M2: 之後在這裡接 MLflow log_*`（下一模組接點）

```bash
python workspace/train.py   # 跑兩次，accuracy 必須一樣
```

> 為什麼搬進 workspace？沙盒是「練工具、可丟可重來」；workspace 是「你真正長大的成品」。模板只給骨架，關鍵邏輯仍要你自己填。

### 極簡 Git 工作流 cheatsheet

階 0 只需要四個動詞就能把 baseline 安全版控起來。

> **先搞清楚 git 在管哪個範圍**：`mlops-course/` **不是**獨立的 git repo，
> 它沒有自己的 `.git`。真正的 repo 根是**再上一層**的 `MLOps-ds-fundamental/`。
> 所以：
> - `git status` 會列出**整個 repo** 的變更（可能包含 `docs/`、`.github/` 等），不只 `mlops-course/`。看到別的目錄是正常的，不要以為自己改壞了什麼。
> - git 指令在哪一層下都可以，但**檔案路徑相對你當下的位置**。下面這組是站在 `mlops-course/` 時的寫法。
> - 開分支與 commit 影響整個 repo，不是只有這個模組。
>
> 用 `git rev-parse --show-toplevel` 可以隨時印出 repo 根在哪。

```bash
cd mlops-course                 # 以下路徑以這一層為準（在 repo 根則路徑要加 mlops-course/ 前綴）

git status                      # 我現在改了什麼？（動手前後都先看一眼）
git branch feat/m1-baseline     # 開一條功能分支（鐵律：別在 main 上直接改）
git switch feat/m1-baseline     # 切到該分支（舊版 Git 用 git checkout）
git add workspace/train.py      # 把要納管的檔案放進暫存區（相對當前目錄）
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
