# M1 應用指導手冊：可重現 baseline 與 Git 工作流

## 這個模組解決什麼工程問題

M1 的目標不是追求高分模型，而是建立一個工程團隊能信任的起點：同一份程式、同一份資料、同一組參數，今天、明天、別人機器上都能跑出一致結果。

如果 baseline 不可重現，後面的 MLflow、Optuna、DVC、Feast 都只是把混亂記錄得更完整。M1 先把訓練流程整理成可執行、可版控、可比較的最小單元。

## 哪邊應用

M1 適用在任何 ML 專案的第一天：你還不需要完整平台，也不需要複雜 pipeline；你需要的是一條乾淨 baseline，讓後續每個改善都有比較基準。

常見應用：

- 新專案起步：先用簡單模型確認資料有沒有訊號。
- 教學與 onboarding：讓新工程師理解資料、模型、評估的最小流程。
- 原型驗證：在導入 MLflow、DVC、Feast 前，先確認訓練腳本可重跑。
- 除錯基準：複雜模型壞掉時，回到 baseline 判斷問題在資料、模型還是環境。

## 怎麼用

使用順序是先探索、再固定、最後版控：

1. 用 notebook 看資料分布、類別平衡與特徵可分性。
2. 用 Python 腳本固定讀資料、切分、訓練、評估流程。
3. 所有隨機步驟都設定 seed。
4. 重跑兩次確認 metric 一致。
5. 用 Git branch 與 commit 保存這個 baseline。

## 本模組元件

| 元件 | 在 MLOps 裡的職責 | 本課使用方式 |
| :--- | :--- | :--- |
| Python | 將資料處理與訓練流程寫成可重複執行的腳本 | `01_baseline_iris.py` |
| scikit-learn | 快速建立傳統 ML baseline | LogisticRegression + train/test split |
| Git | 管理程式碼版本與協作流程 | branch、commit、status |
| 固定 seed | 控制隨機性，讓結果可重現 | `random_state=42` |
| Notebook | 給人理解資料分布與可分性 | `00_eda_iris.ipynb` |

## Python 在這裡怎麼應用

Python 在 MLOps 專案裡有兩種用法：探索與生產化。探索階段可以用 notebook 互動式看資料；一旦流程需要被別人重跑，就應該整理成 `.py` 腳本。

工程應用場景：

- 將資料讀取、切分、訓練、評估串成一支可執行腳本。
- 將人腦操作轉成機器可重跑的流程。
- 為後續 MLflow 追蹤與 CI 自動化提供穩定入口。

設計原則：

- 腳本應該從固定資料路徑讀資料。
- 隨機性必須顯式設定。
- 執行結果應該能從終端機清楚看到。
- 不要把訓練流程留在 notebook 裡當唯一版本。

## scikit-learn 在這裡怎麼應用

scikit-learn 適合用來建立第一個可靠 baseline。baseline 的價值不是最好，而是作為之後所有實驗比較的參考線。

典型使用方式：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))
```

工程判斷：

| 問題 | 建議 |
| :--- | :--- |
| 資料是表格型，想快速知道有沒有訊號 | 先用 scikit-learn baseline |
| 模型分數不穩定 | 先檢查 split seed 與資料前處理是否固定 |
| 模型很複雜但沒有 baseline | 先補 baseline，否則不知道複雜度是否值得 |

## Git 在這裡怎麼應用

Git 管的是程式碼與小型文字設定，不是大型資料本體。它的工程價值是讓你知道每一次變更的脈絡，並且讓團隊能安全協作。

最小工作流：

```bash
git status
git branch feat/m1-baseline
git switch feat/m1-baseline
python modules/m1-foundations/sandbox/01_baseline_iris.py
git add modules/m1-foundations/sandbox/01_baseline_iris.py
git commit -m "feat(m1): add reproducible iris baseline"
```

工程規則：

| 規則 | 原因 |
| :--- | :--- |
| 先開 branch 再改 | 避免直接污染主線 |
| commit 要小 | 方便 review 與回滾 |
| message 說明原因 | 未來排查問題時能理解當時決策 |
| `git status` 常看 | 避免誤 commit 不相關檔案 |

## 固定 seed 怎麼應用

固定 seed 是 MLOps 最基本的可重現控制。常見隨機來源包括 train/test split、模型初始化、shuffle、抽樣、HPO trial。

實務做法：

- 在檔案開頭集中定義 `SEED = 42`。
- 所有支援 `random_state` 的 API 都使用同一個 seed。
- 如果用 NumPy 或 PyTorch，也要同步設定它們的 seed。
- 在 README 或 MLflow params 中記錄 seed。

驗收方式：

```bash
python modules/m1-foundations/sandbox/01_baseline_iris.py
python modules/m1-foundations/sandbox/01_baseline_iris.py
```

兩次輸出的 test accuracy 應該一致。

## Notebook 與腳本怎麼分工

| 介質 | 適合工作 | 不適合工作 |
| :--- | :--- | :--- |
| Notebook | EDA、視覺化、互動理解資料 | 當作唯一訓練入口 |
| Python script | 訓練、評估、CI、排程、自動化 | 大量互動式探索 |

工程判斷句：人在看資料時用 notebook；機器要重跑時用腳本。

## 最小落地流程

1. 用 notebook 看資料欄位、分布、類別平衡。
2. 決定第一個 baseline 模型與評估指標。
3. 把資料讀取、split、train、evaluate 寫成 `.py`。
4. 固定 seed，重跑兩次確認分數一致。
5. 用 Git branch 與 commit 保存這個乾淨起點。
6. 在下一模組把 MLflow 接到同一支訓練流程上。

## 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 沒固定 seed | 每次分數不同，無法比較實驗 | 所有隨機 API 顯式設 seed |
| 只留 notebook | CI 與排程很難重跑 | 將核心流程整理成腳本 |
| baseline 太複雜 | 後面難判斷改善來源 | 第一版模型保持簡單 |
| 直接在 main 改 | 協作與回滾風險高 | 開 feature branch |
| 把資料硬塞進 Git | repo 變大、clone 變慢 | 後續用 DVC 管資料 |

## 工程驗收標準

- 腳本可從乾淨環境執行。
- 兩次執行得到相同 metric。
- 資料切分與模型訓練有固定 seed。
- Git commit 只包含本任務相關檔案。
- README 能說清楚資料、模型、metric、如何重跑。

## 你應該帶走的能力

完成 M1 後，工程師應該能把一個臨時 ML 實驗整理成可版控、可重跑、可交接的 baseline。這是後續所有 MLOps 工具能發揮作用的前提。
