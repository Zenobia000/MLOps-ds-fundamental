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

玩具資料一律用 `datasets/iris.csv`（150 列、4 個數值特徵 + `target` 標籤）。
每個 `.py` 都能**獨立執行**、彼此不互相 import。先裝工具：

```bash
pip install scikit-learn "mlflow>=2.9" optuna dvc
```

### 2-1　MLflow（階 1）— 最小可用動詞：`start_run / log_param / log_metric / log_model`

```bash
cd modules/m2-tracking-tuning-versioning/sandbox/mlflow

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
```

### 2-3　DVC（階 3）— 最小可用動詞：`init / add / push / checkout`

純指令教學，照著做：

```bash
cd ../dvc
# 打開 01_version_a_csv.md，照步驟把 iris.csv 做版本化，
# 體會「同一個 git commit 永遠拉到同一份資料」。
```

> 小提醒：新版 MLflow 對「本地檔案紀錄（`./mlruns`）」預設會擋，沙盒腳本已在檔頭用
> `MLFLOW_ALLOW_FILE_STORE=true` 明確允許，零設定即可跑。

---

## 3. 整合任務（Layer 2：到 `workspace/` 把工具接上去）

把這三個工具接到你在 M1 建好的 baseline 上。到 `mlops-course/workspace/` 裡：

- [ ] **TODO（MLflow）**：把 M1 的訓練腳本包進 `with mlflow.start_run():`，
      `log_param` 記超參、`log_metric` 記 accuracy/f1、`log_model` 存模型 + signature。
- [ ] **TODO（Optuna）**：把固定超參改成 `objective(trial)`，用 `study.optimize` 自動找；
      **每個 trial 用 nested run 寫進 MLflow**，最後把 `best_params` 記在 parent run。
- [ ] **TODO（DVC）**：對 `workspace/` 用到的訓練資料 `dvc init / add / push`，
      把 `.dvc` 指標檔 commit 進 Git，讓「程式碼版本」對應「資料版本」。

> 驗收：在 `mlflow ui` 看得到你 workspace 的實驗；`git log` 的每個 commit 都能用
> `dvc checkout` 還原出當時的資料。

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
