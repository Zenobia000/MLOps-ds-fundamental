# M2 應用指導手冊：MLflow、Optuna、DVC

## 這個模組解決什麼工程問題

M2 把「我跑過一些實驗」升級成「團隊可以追蹤、比較、重現實驗」。它處理三個核心問題：實驗紀錄散落各處、超參數靠人工亂試、資料版本和程式碼版本對不起來。

本模組的工程心法是：每一次訓練都要留下紀錄；每一次自動搜尋都要能被比較；每一個 commit 都要知道它對應哪份資料。

## 哪邊應用

M2 適用在 baseline 已經可重跑、但實驗開始變多的階段。當你開始問「哪個模型最好」「那次分數怎麼來的」「這個 commit 用哪份資料」時，就需要 M2 的元件。

常見應用：

- 模型比較：比較不同模型、特徵、參數與資料版本。
- 自動調參：讓系統批次嘗試超參數，而不是人工手動亂試。
- 實驗審查：讓 reviewer 能看到 params、metrics、artifact。
- 可重現訓練：讓舊 commit 能找回當時使用的資料版本。

## 怎麼用

使用順序是先記錄、再搜尋、最後鎖住資料版本：

1. 在訓練腳本加入 MLflow，記錄 params、metrics、model。
2. 確認 MLflow UI 能比較不同 run。
3. 將訓練流程包成 Optuna objective。
4. 每個 Optuna trial 建立 nested MLflow run。
5. 用 Optuna 視覺化分析搜尋空間。
6. 用 DVC 追蹤資料檔，Git 只 commit `.dvc` 指標檔。

## 本模組元件

| 元件 | 在 MLOps 裡的職責 | 本課使用方式 |
| :--- | :--- | :--- |
| MLflow | 記錄 params、metrics、model artifacts | 每次訓練建立 run |
| Optuna | 自動搜尋超參數 | 每個 trial 對應一個 MLflow run |
| DVC | 管理資料與模型檔版本 | Git commit 對應 `.dvc` 指標檔 |
| Plotly | 觀察 HPO 搜尋過程 | Optuna visualization notebook |

## MLflow 怎麼應用

MLflow 用來回答：「這次模型是怎麼訓出來的？」工程上要記三類東西：輸入設定、輸出結果、產物。

最小紀錄範圍：

| 類型 | 例子 | 用途 |
| :--- | :--- | :--- |
| params | model type、C、max_iter、seed | 重現訓練條件 |
| metrics | accuracy、f1、AUC、loss | 比較實驗好壞 |
| artifacts | model file、conf、plots | 保存可交付產物 |

最小實作骨架：

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-baseline")

with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("seed", 42)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name="model")
```

工程應用場景：

- 比較不同模型、不同資料版本、不同特徵組合。
- 保存模型 artifact，讓後續服務化能讀回。
- 訓練流程失敗後保留可查線索。
- 把實驗結果交給團隊 review。

設計建議：

- 每個獨立訓練任務都建立一個 run。
- 實驗名稱應該對應任務或模型線，例如 `iris-optuna-hpo`。
- 必要設定檔也要 log 成 artifact。
- 不要只記最佳結果，失敗與普通結果也有比較價值。

## Optuna 怎麼應用

Optuna 用來自動產生超參數組合，並根據目標分數尋找更好的參數。它不取代你的訓練流程，而是包住訓練流程反覆嘗試。

最小心智模型：

| 概念 | 意義 |
| :--- | :--- |
| `study` | 一次超參數搜尋任務 |
| `trial` | 一次參數嘗試 |
| `objective` | 給 trial 一組參數，回傳要最大化或最小化的分數 |
| `suggest_*` | 定義搜尋空間 |
| `best_params` | 搜尋後找到的最佳參數 |

最小實作骨架：

```python
import optuna

def objective(trial):
    c = trial.suggest_float("C", 1e-3, 10.0, log=True)
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    return accuracy_score(y_valid, pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(study.best_params)
```

與 MLflow 搭配：

- parent run 記錄整次 HPO 任務。
- nested child run 記錄每個 trial。
- trial params 與 metrics 全部寫進 MLflow。
- 最佳參數寫回 parent run，方便 UI 上快速定位。

工程應用場景：

- baseline 已穩定，需要系統化改善模型。
- 模型有多個重要超參數，人工試參數效率低。
- 需要保留完整搜尋過程，而不是只知道最後答案。

## DVC 怎麼應用

DVC 用來管理大型資料與模型檔。它的核心價值是讓 Git commit 能對應到正確資料版本。

Git 與 DVC 分工：

| 工具 | 管什麼 | 不適合管什麼 |
| :--- | :--- | :--- |
| Git | 程式碼、小型設定、`.dvc` 指標檔 | 大型 CSV、Parquet、模型權重 |
| DVC | 資料內容、模型檔、特徵輸出 | 程式碼 review 流程 |

最小工作流：

```bash
dvc init
dvc add data/iris.csv
git add .dvc .gitignore data/iris.csv.dvc
git commit -m "data: track iris dataset with dvc"
dvc push
```

回到舊版本時：

```bash
git checkout <old-commit>
dvc checkout
```

工程應用場景：

- 資料集會更新，但舊實驗必須能重現。
- 模型 artifact 太大，不適合放 Git。
- CI 或訓練機要拉到與程式碼一致的資料。

## Plotly 與 HPO 視覺化怎麼應用

Optuna 的視覺化不是為了好看，而是幫你決定下一輪搜尋策略。

| 圖 | 回答的工程問題 | 下一步行動 |
| :--- | :--- | :--- |
| optimization history | 分數是否還在進步 | 紅線走平就停止或換搜尋空間 |
| param importances | 哪些參數最重要 | 不重要的參數固定住 |
| parallel coordinate | 好 trial 集中在哪些範圍 | 縮小搜尋空間 |
| slice plot | 最佳點是否貼邊界 | 貼邊界就擴大範圍 |

## 最小落地流程

1. 從 M1 的可重現 baseline 開始。
2. 加入 MLflow，記錄 params、metrics、model。
3. 確認 MLflow UI 看得到每次 run。
4. 將模型訓練包進 Optuna objective。
5. 每個 Optuna trial 用 nested MLflow run 記錄。
6. 用 Optuna visualization 分析搜尋是否值得繼續。
7. 用 DVC 將訓練資料版本化。
8. 在 Git commit 中保存程式碼與 `.dvc` 指標檔。

## 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 只記最佳分數 | 無法理解搜尋過程 | 每個 run 都記錄 |
| params 記不完整 | 實驗無法重現 | seed、資料版本、模型設定都要記 |
| Optuna trial 沒接 MLflow | HPO 結果不可追蹤 | trial 內建立 nested run |
| 把大型資料 commit 進 Git | repo 膨脹、clone 變慢 | 用 DVC add |
| `git checkout` 後忘記 `dvc checkout` | 程式碼回舊版但資料沒回去 | 兩個指令成對使用 |

## 工程驗收標準

- MLflow UI 看得到實驗、run、params、metrics、model。
- Optuna 每個 trial 都能對應到 MLflow run。
- 最佳參數能從 MLflow 與 Optuna 查到。
- 資料檔由 DVC 管理，Git 只保存 `.dvc` 指標。
- 舊 commit 加 `dvc checkout` 能取回對應資料。

## 你應該帶走的能力

完成 M2 後，工程師應該能建立一套可追蹤的訓練流程：知道每次模型怎麼產生、為什麼變好或變差、使用哪份資料，以及如何回到過去的實驗狀態。
