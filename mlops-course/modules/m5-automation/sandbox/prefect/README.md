# sandbox/prefect — 把兩支函式串成一個 flow（階 9）

> 一次一動詞：這個沙盒只教 Prefect 的三個核心動詞 `@task` / `@flow` / 本地 `run`。
> deployment、schedule、blocks **全部延後**，等你真的要排程時再回來。
>
> 編排概念說明與概念圖：[`../../prefect-introduction.md`](../../prefect-introduction.md) · [`../../assets/m5-prefect-orchestration.png`](../../assets/m5-prefect-orchestration.png)

## 這個沙盒在教什麼

Prefect 是「工作流程編排（workflow orchestration）」工具。它的最小心智模型只有兩層：

| 動詞 | 角色 | 一句話 |
| :--- | :--- | :--- |
| `@task` | 一個**步驟** | 你既有的任何函式，加上 `@task` 就被 Prefect 追蹤 |
| `@flow` | 一條**流程** | 在 flow 裡呼叫多個 task，Prefect 自動建執行圖、記狀態 |
| 本地 `run` | **執行** | `python flow.py` 直接跑，不需要 server |

> 為什麼先學這個：你在 M1–M4 已經有一堆「會跑的函式」（load、train、eval、serve）。
> Prefect 不要求你重寫它們——只要加 `@task` 裝飾器，再用 `@flow` 串起來，
> 就從「一堆散落的腳本」升級成「一條有狀態、可重跑、可觀測的 pipeline」。

## 怎麼跑

```bash
# 依賴不用另外裝，課程統一環境已含 prefect / pandas / scikit-learn。
# 在本資料夾執行：
python flow.py
```

預期輸出（節錄）：

```
[load_data] 已載入 150 筆 iris 資料，3 個類別
[train_eval] 評估完成，accuracy = 0.967
[flow] pipeline 結束，最終 accuracy = 0.967
```

> 資料與模型**刻意沿用 M1 的 iris baseline**——同一份 `datasets/iris.csv`、
> 同一顆 `SEED = 42`、同一個 LogisticRegression，所以 accuracy 跟你在 M1 跑出來的一致。
> 這是故意的：這個沙盒沒有任何新的 ML，你才能把注意力全放在
> 「加上 `@task` / `@flow` 之後多了什麼」。

同時你會看到 Prefect 自動印出每個 task 的狀態流轉（`Running` -> `Completed`）。
這就是 Prefect 幫你做的事：**不改你的邏輯，卻多了狀態追蹤與失敗重試的能力**。

## 看一眼程式（`flow.py`）

```python
def find_iris_csv() -> Path: ...        # 純路徑計算，刻意不加 @task

@task
def load_data() -> pd.DataFrame: ...    # 步驟一：讀 datasets/iris.csv

@task
def train_eval(df) -> float: ...        # 步驟二：訓練 + 評估

@flow(name="iris-train-flow")
def main() -> float:
    df = load_data()                    # flow 內呼叫 task，Prefect 自動串接
    return train_eval(df)               # 上一個 task 的輸出 -> 下一個的輸入

if __name__ == "__main__":
    main()                              # 本地 run，零 server
```

> 注意 `find_iris_csv()` **沒有** `@task`。判準不是「是不是函式」，而是
> 「值不值得被單獨追蹤與重試」——它只是算路徑，包成 task 只會讓執行圖
> 多一個沒資訊量的節點。什麼該切成 task，是 Prefect 用起來最常見的取捨。

## 明確延後（先不要學）

- **deployment**：把 flow 註冊到 Prefect server / Cloud，讓它能被遠端觸發。
- **schedule**：用 cron 或 interval 定時自動跑。
- **blocks**：把資料庫、S3 等外部連線設定存成可重用的 block。

> 教學話術：今天你只要會「`@task` 包步驟、`@flow` 串流程、`python` 跑起來」就能做事。
> 等你需要「每天凌晨自動重訓」時，我們再回來學 deployment + schedule。

## 自我檢核

1. `@task` 和 `@flow` 的差別是什麼？哪個是「步驟」、哪個是「流程」？
2. flow 裡 `train_eval(data)` 的 `data` 是怎麼來的？Prefect 在這中間做了什麼？
3. 為什麼不需要先開 Prefect server 也能跑 `python flow.py`？
