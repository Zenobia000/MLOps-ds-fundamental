# Prefect 入門：工作流程編排（Workflow Orchestration）

> 本頁回答：**Prefect 是什麼、為什麼叫編排、`@task` / `@flow` 差在哪、本課學到哪裡為止。**  
> 動手實作：[`sandbox/prefect/`](./sandbox/prefect/README.md)  
> 概念圖：[`assets/m5-prefect-orchestration.png`](./assets/m5-prefect-orchestration.png)  
> Pipeline 五步示意（含 quality gate）：[`assets/m5-prefect-pipeline_20260801_131741.png`](./assets/m5-prefect-pipeline_20260801_131741.png)

---

## 1. 一句話定位

**Prefect 是工作流程編排（workflow orchestration）工具**——把你「已經會跑的函式」串成一條有順序、有狀態、失敗可定位的 pipeline，而不是重寫一套新框架。

| 沒有編排 | 有 Prefect |
| :--- | :--- |
| `python load.py && python train.py && python eval.py` | `@flow` 裡依序呼叫 `@task` |
| 失敗了不知道卡在哪一步 | 每個 task 有 `Running` / `Completed` / `Failed` |
| 重跑要人記得指令順序 | 一條 flow，輸入輸出自動串接 |

> **編排 ≠ 訓練模型。**  
> 模型邏輯仍是你的 Python；Prefect 管的是「步驟怎麼接、跑到哪、壞在哪」。

---

## 2. 最小心智模型（本課只教這三個動詞）

```text
              ┌─────────── @flow (iris-train-flow) ───────────┐
              │                                               │
              │  @task load_data() ──df──▶ @task train_eval() │
              │                                               │
              └──────────────────────┬────────────────────────┘
                                     │
                              python flow.py
                           （本地 run，零 server）
```

| 動詞 | 角色 | 一句話 |
| :--- | :--- | :--- |
| **`@task`** | 一個**步驟** | 既有函式加裝飾器，就被追蹤狀態 |
| **`@flow`** | 一條**流程** | 在 flow 裡呼叫多個 task，組成執行圖 |
| **本地 `run`** | **執行** | `python flow.py` 直接跑，不必先開 server |

對應本課沙盒 [`flow.py`](./sandbox/prefect/flow.py)：

```python
@task
def load_data() -> pd.DataFrame: ...        # 讀 datasets/iris.csv

@task
def train_eval(df) -> float: ...            # 沿用 M1 的 iris baseline

@flow(name="iris-train-flow")
def main() -> float:
    df = load_data()
    return train_eval(df)

if __name__ == "__main__":
    main()
```

> 沙盒沿用 M1 的 iris baseline（同資料、同 seed、同模型），**沒有新的 ML**。
> 這樣你比對得出來的差異，就純粹是 Prefect 帶來的那一層。

---

## 3. 「編排」在 MLOps 裡解決什麼

M1–M4 你已經有 load、train、eval、serve 等函式。痛點通常是：

1. 手動照順序跑，容易漏步、難交接  
2. 失敗時只看到最後一個 traceback，不知道是資料壞還是訓練壞  
3. 之後要接 **CT（持續訓練）**：新資料 / drift / 排程 → 自動重跑整條線  

Prefect 先把「手動腳本鏈」升級成「可觀測 pipeline」；排程與遠端觸發（deployment / schedule）本課**刻意延後**。

在 M5 更大圖景裡的位置：

```text
CT 流程（常由 Prefect 編排）
  load → validate → train → evaluate → quality gate
                                           │
                              不夠好就 Failed，擋住上線
```

CI（GitHub Actions + pytest）管「程式碼有沒有壞」；  
Prefect 更常管「資料/訓練這條流水線怎麼跑」。

---

## 4. Task 狀態（你在終端會看到的）

| 狀態 | 意義 |
| :--- | :--- |
| Pending / Running | 排隊或執行中 |
| Completed | 這步成功，輸出可傳給下一步 |
| Failed | 這步失敗；flow 通常停在這裡（可再設重試） |

本課重點：**不改你的業務邏輯，卻多了狀態追蹤。**  
（重試、告警、UI 儀表板等屬進階，之後再加。）

---

## 5. 明確延後（先不要學）

| 主題 | 以後什麼時候回來 |
| :--- | :--- |
| **deployment** | 要把 flow 註冊到 server / Cloud，給遠端觸發 |
| **schedule** | 需要「每天凌晨自動重訓」 |
| **blocks** | 要重用 DB / S3 等連線設定 |

教學話術：今天只要會「`@task` 包步驟、`@flow` 串流程、`python` 跑起來」。

---

## 6. 檢核

1. `@task` 和 `@flow` 哪個是步驟、哪個是流程？  
2. 為什麼說 Prefect 是編排工具，不是另一個訓練框架？  
3. 為什麼本課強調「不必先開 server」也能 `python flow.py`？  

動手：在 [`sandbox/prefect/`](./sandbox/prefect/README.md) 跑 `python flow.py`，對照終端裡每個 task 的狀態流轉。
