"""
sandbox/prefect/flow.py — 階 9 Prefect 最小可用單元

這個檔示範什麼：
    只用 Prefect 的三個核心動詞 —— @task / @flow / 本地 run —— 把
    「兩支既有函式」串成一個有向流程（flow）。
    重點：你不需要 server、不需要 deployment、不需要 schedule，
    一個 `python flow.py` 就能在本地跑完整條 pipeline，並由 Prefect
    自動記錄每個 task 的執行狀態（成功/失敗/重試）。

延後（之後需要時再回來）：
    - deployment（把 flow 註冊到 Prefect server / Cloud）
    - schedule（定時觸發）
    - blocks（外部資源連線設定）

怎麼跑：
    # 先確保已安裝 prefect（課程 pyproject.toml 已含）：
    #   pip install prefect
    # 在本資料夾執行：
    python flow.py

    跑完你會在終端看到 Prefect 印出每個 task 的狀態流轉
    （Pending -> Running -> Completed），最後印出評估結果。
"""

import random

from prefect import flow, task

# 設定隨機種子：教學腳本只要用到隨機性就固定 seed，確保結果可重現。
SEED = 42
random.seed(SEED)


@task
def load_data() -> list[dict]:
    """第一支 task：載入「玩具資料」。

    為了讓沙盒零依賴（不需要外部 CSV、不需要 sklearn），
    這裡直接用程式生出一份極簡的玩具樣本：每筆 = 一個特徵 x 與標籤 y。
    在真實專案裡，這支函式才會去讀 datasets/iris.csv 之類的資料。
    """
    # y = 1 代表 x 大於 0.5，加上一點點雜訊讓資料不是完美可分。
    data = []
    for _ in range(100):
        x = random.random()
        noise = random.random() < 0.1  # 10% 機率翻轉標籤，模擬真實雜訊
        y = int(x > 0.5)
        if noise:
            y = 1 - y
        data.append({"x": x, "y": y})
    print(f"[load_data] 已載入 {len(data)} 筆玩具資料")
    return data


@task
def train_eval(data: list[dict]) -> float:
    """第二支 task：用最樸素的規則「訓練 + 評估」。

    這裡不引入 sklearn，純手寫一個閾值分類器（x > 0.5 -> 1），
    目的只是示範「task 之間如何傳資料」與「flow 如何串起來」，
    而不是教模型本身。回傳值 = 準確率（accuracy）。
    """
    correct = sum(1 for row in data if int(row["x"] > 0.5) == row["y"])
    accuracy = correct / len(data)
    print(f"[train_eval] 評估完成，accuracy = {accuracy:.3f}")
    return accuracy


@flow(name="toy-train-flow")
def main() -> float:
    """@flow：把上面兩支 task 串成一條 pipeline。

    flow 內部呼叫 task 時，Prefect 會自動建立執行圖、追蹤狀態、
    並把上一個 task 的輸出當作下一個 task 的輸入（這裡 data -> train_eval）。
    """
    data = load_data()
    accuracy = train_eval(data)
    print(f"[flow] pipeline 結束，最終 accuracy = {accuracy:.3f}")
    return accuracy


if __name__ == "__main__":
    # 本地 run：直接呼叫 flow 函式即可在本機跑完整條流程。
    # 這就是 Prefect「最小可用」的用法——先把 flow 跑起來，
    # deployment / schedule 等到你真的要排程時再學。
    main()
