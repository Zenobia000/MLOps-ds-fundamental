# M5 應用指導手冊：Prefect、pytest、GitHub Actions

## 這個模組解決什麼工程問題

M5 解決的是「流程能不能自動且可靠地跑」。一個 ML 專案如果只能靠工程師手動照順序執行指令，就很難維運。自動化的目標是把資料載入、驗證、訓練、評估、品質門檻、測試全部變成可觀測、可重跑、可失敗阻擋的流程。

本模組的工程心法是：會跑不夠，還要能自動跑；自動跑不夠，還要失敗時能明確阻止壞結果進入下一步。

## 哪邊應用

M5 適用在流程已經不只一支手動腳本，而是有多個步驟需要穩定重跑的階段。當資料更新、程式 push、模型重訓、品質檢查都需要被自動化時，就進入 M5 的範圍。

常見應用：

- 持續訓練：新資料到來後自動跑訓練 pipeline。
- Pull request 品質門檻：每次改程式都自動跑測試。
- 資料驗證：資料壞掉時在訓練前就失敗。
- 模型上線前檢查：分數不達標就阻止部署。

## 怎麼用

使用順序是先拆流程，再加測試，最後放到 CI：

1. 將流程拆成 load、validate、train、evaluate、gate。
2. 用 Prefect `@task` 包每個步驟。
3. 用 Prefect `@flow` 串完整 pipeline。
4. 用 pytest 測資料假設、metric 與 pipeline smoke test。
5. 在本機跑 `python -m pytest -v`。
6. 將 GitHub Actions workflow 放到 `.github/workflows/`。
7. 讓 push 或 PR 自動跑測試與品質門檻。

## 本模組元件

| 元件 | 在 MLOps 裡的職責 | 本課使用方式 |
| :--- | :--- | :--- |
| Prefect | 工作流程編排 | `@task`、`@flow` 串 pipeline |
| pytest | Python 測試框架 | 單元測試與資料驗證 |
| GitHub Actions | CI/CD 自動化 | push 後自動跑 pytest |
| CI / CD / CT | 自動化流程概念 | 區分程式、部署、重訓 |
| 品質門檻 gate | 阻擋不合格模型或資料 | accuracy gate、data validation |
| Canary / Blue-Green | 安全部署策略 | 小流量試行或快速切換 |

## Prefect 怎麼應用

Prefect 將普通 Python 函式組成可觀測的 workflow。它不要求你重寫訓練邏輯，而是把既有函式標記成 task，再用 flow 串起來。

最小心智模型：

| 概念 | 意義 |
| :--- | :--- |
| `@task` | pipeline 裡的一個步驟 |
| `@flow` | 一條由多個 task 組成的流程 |
| state | 每個 task 的執行狀態，例如 Running、Completed、Failed |
| deployment | 將 flow 註冊到 server 或 cloud，供排程與遠端觸發 |

最小實作骨架：

```python
from prefect import flow, task

@task
def load_data():
    return data

@task
def validate_data(data):
    if data.empty:
        raise ValueError("empty data")
    return data

@task
def train_model(data):
    return model, metrics

@task
def quality_gate(metrics):
    if metrics["accuracy"] < 0.85:
        raise ValueError("model quality below threshold")

@flow
def training_flow():
    data = load_data()
    data = validate_data(data)
    model, metrics = train_model(data)
    quality_gate(metrics)

if __name__ == "__main__":
    training_flow()
```

工程應用場景：

- 訓練流程有多個步驟，需要明確知道哪一步失敗。
- 需要每日、每週、資料到達後自動重訓。
- 需要把漂移偵測結果接到重訓流程。
- 需要讓 pipeline 狀態可觀測。

設計原則：

- task 粒度應對應可理解的工程步驟。
- 每個 task 要有清楚輸入與輸出。
- validation 與 quality gate 應該在流程中早點失敗。
- 不要把整支訓練腳本包成一個巨大 task。

## pytest 怎麼應用

pytest 用來自動確認程式與資料假設仍然成立。ML 專案除了測函式邏輯，還要測資料與模型品質。

測試類型：

| 類型 | 測什麼 | 例子 |
| :--- | :--- | :--- |
| 單元測試 | 小函式邏輯 | metric 算法是否正確 |
| 資料驗證 | 輸入資料是否符合預期 | 欄位存在、型別正確、缺值比例 |
| Pipeline smoke test | 流程是否可跑 | 小資料跑完整流程 |
| 品質門檻 | 模型是否達標 | accuracy >= 0.85 |

最小測試骨架：

```python
def test_accuracy_gate_passes():
    metrics = {"accuracy": 0.9}
    assert metrics["accuracy"] >= 0.85

def test_required_columns_exist(dataframe):
    required = {"sepal_length", "sepal_width", "petal_length", "petal_width", "target"}
    assert required.issubset(dataframe.columns)
```

工程應用場景：

- PR 合併前自動檢查。
- pipeline 改動後確認核心流程沒壞。
- 資料來源換版後確認 schema 沒變。

## GitHub Actions 怎麼應用

GitHub Actions 讓每次 push 或 pull request 都自動跑檢查。

最小 workflow：

```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pytest
      - run: python -m pytest -v
```

工程應用場景：

- PR 自動跑測試。
- 測試失敗就阻止 merge。
- 後續可加 Docker build、lint、deploy。
- 保證團隊不是靠口頭提醒跑測試。

設計原則：

- 本機先跑過同一條 pytest 指令再 push。
- CI 裡的步驟應該盡量接近生產流程。
- workflow 檔案必須放在 repo 根目錄 `.github/workflows/`。
- 不要在 CI 裡跑超長或昂貴訓練，除非它是必要品質門檻。

## CI / CD / CT 怎麼判斷

| 名稱 | 觸發來源 | 主要問題 | 典型產出 |
| :--- | :--- | :--- | :--- |
| CI | 程式碼 push / PR | 程式碼有沒有壞 | 測試通過 |
| CD | CI 通過 | 服務能不能上線 | 部署版本 |
| CT | 新資料 / 漂移 / 排程 | 模型是否需要重訓 | 新模型 artifact |

ML 專案比一般軟體多 CT，因為即使程式碼沒變，資料分布也會變，模型品質可能退化。

## 品質門檻 gate 怎麼設計

品質門檻 gate 是自動化流程中最重要的保護機制。

常見 gate：

| Gate | 阻擋什麼 |
| :--- | :--- |
| Data schema gate | 欄位缺失、型別錯誤 |
| Data quality gate | 缺值過多、值域異常 |
| Model metric gate | 模型低於最低品質 |
| Regression gate | 新模型比舊模型差太多 |
| Drift gate | 上線資料偏離 reference |

設計原則：

- 門檻要從業務風險倒推，不要隨便寫一個漂亮數字。
- gate 失敗要讓 pipeline 明確失敗。
- 失敗訊息要足夠清楚，方便工程師修。
- 初期可以簡單，後期再引入更完整的 validation framework。

## Canary 與 Blue-Green 怎麼應用

模型部署要控制風險，不能每次新模型都直接吃 100% 流量。

| 策略 | 做法 | 優點 | 風險 |
| :--- | :--- | :--- | :--- |
| Canary | 先給 1% 到 5% 流量 | 影響小、可觀察 | 需要流量切分能力 |
| Blue-Green | 舊版與新版並行，切換流量 | 回滾快 | 需要維持兩套環境 |

工程應用：

- 高風險模型先 canary。
- 需要快速切回舊版時用 blue-green。
- 新模型上線期間要看系統指標與模型指標。

## 最小落地流程

1. 將訓練流程拆成 load、validate、train、evaluate、gate。
2. 用 Prefect `@task` 包每個步驟。
3. 用 Prefect `@flow` 串完整 pipeline。
4. 寫 pytest 測核心函式與資料假設。
5. 本機執行 `python -m pytest -v`。
6. 將 GitHub Actions workflow 放到 `.github/workflows/`。
7. push 後確認 Actions 自動跑測試。
8. 將品質門檻接進 pipeline，失敗就停止。

## 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| pipeline 全寫在一個函式 | 失敗時不知道哪一步壞 | 拆成明確 task |
| 只測程式不測資料 | schema 變了才在訓練時爆 | 加資料驗證測試 |
| CI 路徑放錯 | GitHub 不會執行 workflow | 放到 `.github/workflows/` |
| gate 只印 warning | 壞模型仍可能往下走 | gate 失敗要 raise error |
| CI 跑太重 | PR feedback 太慢 | 區分快速測試與長訓練 |

## 工程驗收標準

- Prefect flow 可本機執行完整 pipeline。
- 每個 task 有清楚狀態與錯誤位置。
- pytest 可在本機通過。
- GitHub Actions push 後自動跑測試。
- 品質門檻失敗時 pipeline 會停止。
- README 寫清楚如何本機重跑 CI 同款指令。

## 你應該帶走的能力

完成 M5 後，工程師應該能把手動 ML 流程轉成可觀測的自動 pipeline，並用測試與 CI 阻擋壞資料、壞程式、壞模型進入下一步。
