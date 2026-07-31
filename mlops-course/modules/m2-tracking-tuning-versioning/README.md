# M2 — 追蹤、調參、版本化（Tracking / Tuning / Versioning）

## 1. 本模組學什麼

學會把實驗「記下來、自動找最佳超參、連資料一起版本化」三件事：
**MLflow**（技能階梯 **階 1**）負責記錄每次實驗的 params/metrics/model；
**Optuna**（**階 2**）自動掃超參、找最佳；**DVC**（**階 3**）讓資料和程式碼一起被版本控制。

> **本模組的靈魂：每個 Optuna trial = 一個 MLflow run。**
> HPO 與追蹤天生互補——Optuna 負責「自動產生大量實驗」，MLflow 負責「記錄並比較這些實驗」。
> 先有 MLflow 才有地方記 trial，再有 Optuna 才有東西自動產生大量 run。兩者合起來就是 AutoML 的引擎。

---

## 2. 沙盒步驟（Layer 1：照編號逐個跑，只學最小可用動詞）

> **指令起點**：本檔所有路徑都相對 **`mlops-course/`**（放 `Makefile` 的那一層）。
> `cd` 進沙盒跑完，記得 `cd` 回 `mlops-course/` 再接下一段（見 2-4）。
> Git 是例外——repo 根在再上一層，詳見[課程 README「指令從哪裡下」](../../README.md#指令從哪裡下)。

玩具資料一律用 `datasets/iris.csv`（150 列、4 個數值特徵 + `target` 標籤）。
每個 `.py` 都能**獨立執行**、彼此不互相 import。先裝工具：

```bash
pip install scikit-learn "mlflow>=2.9" optuna dvc
```

### 2-1　MLflow（階 1）— 最小可用動詞：`start_run / log_param / log_metric / log_model`

```bash
cd mlops-course/modules/m2-tracking-tuning-versioning/sandbox/mlflow   # 從 repo 根出發

python 01_first_run.py      # 一個 run + log_param + log_metric（最小起手式）
python 02_params_metrics.py # set_tracking_uri / set_experiment + 一次記多組 params/metrics
python 03_log_model.py      # log_model + infer_signature，之後 load_model 讀回來驗證

mlflow ui                   # 在此資料夾開 UI，瀏覽器進 http://127.0.0.1:5000 看你剛記的 run
```

### 2-2　Optuna（階 2）— 最小可用動詞：`create_study / objective+suggest / optimize / best_params`

```bash
cd ../optuna

python 01_objective_basic.py  # 寫 objective(trial)，study.optimize(n_trials=20) 自動找超參
python 02_mlflow_callback.py  # ★核心：每個 trial 寫成一個 MLflow run（nested run），UI 比較收斂
python 03_pruning_asha.py     # 加 pruner 提早砍掉爛 trial；註解附「多目標」與 ASHA 選配

mlflow ui                     # 進實驗 iris-optuna-hpo，展開 parent run 看 20 個 child run

jupyter lab 04_visualization.ipynb   # 讀懂搜尋過程：收斂/重要性/平行座標/slice
```

`04_visualization.ipynb` 是本模組唯一的 notebook，因為 `optuna.visualization` 產出的是
**互動式 Plotly 圖**，而且這四張圖的用途不是好看，是**決定下一輪要怎麼調搜尋範圍**：

| 圖 | 它回答的問題 | 你該做的事 |
| :--- | :--- | :--- |
| optimization history | 還有沒有進步空間？ | 紅線走平就停，別再燒預算 |
| param importances | 誰在決定分數？ | 不重要的超參固定住 |
| parallel coordinate | 好 trial 長什麼樣？ | 把範圍縮到深色線集中的區間 |
| slice plot | 範圍設得對嗎？ | 最佳點貼邊界 → 往外擴 |

> **為什麼 MLflow 那三支刻意不做成 notebook？**
> 兩個理由。技術上，MLflow 的 run lifecycle 在 notebook 是經典陷阱——cell 中途報錯會留下
> 未關閉的 run，下一次 `start_run()` 就錯誤地巢狀進去，教學現場很難除錯。
> 更重要的是教學一致性：**本模組的主題是「可重現」，而亂序執行正是 notebook 最有名的失敗模式**。
> 用一個會給出不同答案的介質去教「每次都要一樣」，訊息會自我矛盾。
> 完整判準見 [`docs/notebook-vs-script.md`](../../../docs/notebook-vs-script.md)。

### 2-3　DVC（階 3）— 最小可用動詞：`init / add / push / checkout`

純指令教學，照著做：

```bash
cd ../dvc
# 打開 01_version_a_csv.md，照步驟把 iris.csv 做版本化，
# 體會「同一個 git commit 永遠拉到同一份資料」。
```

### 2-4　跑完回到 `mlops-course/`

上面三段一路 `cd` 進了 `sandbox/dvc/`，下一節的 `make` 需要回到有 `Makefile` 的那層：

```bash
cd ../../../..      # dvc → sandbox → m2-… → modules → mlops-course
pwd                 # 確認結尾是 /mlops-course
```

> 小提醒：新版 MLflow 對「本地檔案紀錄（`./mlruns`）」預設會擋，沙盒腳本已在檔頭用
> `MLFLOW_ALLOW_FILE_STORE=true` 明確允許，零設定即可跑。

---

## 3. 整合任務（Layer 2：到 `workspace/` 把工具接上去）

先解鎖填空模板（需已有 `workspace/train.py`）：

```bash
make workspace-m2
# 新增 train_tracking.py、conf/params.yaml（已存在則 skip）
```

打開 `workspace/train_tracking.py`，搜尋 `TODO(M2-`（對照本模組 sandbox）：

- [ ] **TODO(M2-1)**：`start_run` + `log_param` / `log_metric` / `log_model` + signature
- [ ] **TODO(M2-2～3)**：Optuna `objective` + nested run；`best_params` 寫進 parent run
- [ ] **TODO（DVC，CLI）**：對訓練資料 `dvc init / add`，把 `.dvc` commit 進 Git
- [ ] **選填**：讓訓練改讀 `conf/params.yaml`

```bash
python workspace/train_tracking.py
make mlflow-ui
```

> 驗收：MLflow UI 看得到 experiment；`git log` 的 commit 能對應 DVC 資料版本。
---

## 4. 卡住怎麼辦

- 想要乾淨的起點：從上一個模組的快照重置 workspace
  ```bash
  cp -r checkpoints/after-m1/* workspace/
  ```
- 想對照本模組做完後「應該長怎樣」：看 `checkpoints/after-m2/`（本模組整合任務的正解）。
- 沙盒範例本身就是可跑的正解——照打、改參數、再跑一次，是最快的除錯方式。
- 常見錯誤：
  - `MlflowException: ... maintenance mode` → 你自己寫的腳本沒設
    `MLFLOW_ALLOW_FILE_STORE=true`（沙盒腳本已內建）。
  - `FileNotFoundError: iris.csv` → 先到 `datasets/` 跑 `python make_datasets.py` 產生玩具資料。

---

## 5. 檢核題（做完問自己）

1. 一個 **MLflow run** 記了哪三類東西？`log_param`、`log_metric`、`log_model` 各對應什麼？
2. `set_experiment` 和 `start_run` 的關係是什麼？為什麼要把相關 run 收進同一個 experiment？
3. Optuna 的 `objective(trial)` 回傳值是什麼？`create_study(direction=...)` 的 direction 怎麼決定？
4. 為什麼說「**每個 Optuna trial = 一個 MLflow run**」？nested run 在 UI 上幫你看到什麼？
5. `git checkout` 一個舊 commit 後，為什麼還要 `dvc checkout` 資料才真的回到那一版？
