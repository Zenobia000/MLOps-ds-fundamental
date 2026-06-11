# M5 自動化：Prefect 編排 + GitHub Actions CI（階 9–10）

## 1. 本模組學什麼

把「會跑的函式」自動化成「會自己跑、跑壞會擋下來」的流程：用 **Prefect**（階 9）
把既有函式串成一條可觀測的 pipeline，用 **GitHub Actions**（階 10）讓每次 push
自動跑測試當品質門檻。這一模組你第一次把「編排」與「CI」接上你的專案。

### 先搞懂三組觀念（這模組的理論骨架）

**CI/CD vs CT（持續訓練）** —— ML 比一般軟體多了一條「資料/模型」的軸線：

| 縮寫 | 全名 | 觸發來源 | 產出 |
| :--- | :--- | :--- | :--- |
| **CI** | Continuous Integration | 程式碼 push | 測試通過的程式碼 |
| **CD** | Continuous Delivery/Deployment | CI 通過 | 部署上線的服務 |
| **CT** | Continuous Training | **新資料 / 偵測到 drift / 排程** | 重新訓練的模型 |

> 一般軟體只有 CI/CD（程式碼變了才重跑）。ML 多一條 **CT**：就算程式碼沒變，
> 只要**資料變了**或模型在線上**退化（drift）**，也要自動重訓。Prefect 正是
> 用來編排 CT 流程的工具（load -> validate -> train -> eval -> 視門檻決定是否上線）。

**ML 專屬測試** —— 傳統測試測「程式邏輯」，ML 還要測「資料」與「模型品質」：

| 測試類型 | 測什麼 | 例子 |
| :--- | :--- | :--- |
| 單元測試 | metric / 前處理函式算得對不對 | `accuracy()` 給定輸入回傳正確值 |
| **資料驗證** | 進 pipeline 的資料符不符合預期 | 欄位齊全、值域正確、無大量缺值、分布沒爆走 |
| **品質門檻 gate** | 新模型夠不夠好才准上線 | `accuracy >= 0.85` 才 deploy，否則擋下 |

> 品質門檻 gate 是 ML CI/CD 的靈魂：**模型不是「能跑就上」，是「夠好才上」**。
> 沒過門檻就讓 pipeline 失敗 / 不部署，避免爛模型悄悄取代好模型。

**部署策略（canary / blue-green 概念）** —— 新模型怎麼安全地換上去：

| 策略 | 做法 | 一句話 |
| :--- | :--- | :--- |
| **Blue-Green** | 同時備好舊（blue）新（green）兩套，瞬間切流量到新版，出事秒切回 | 「整批切換、可秒回滾」 |
| **Canary** | 先放 5% 流量給新模型試水溫，觀察指標沒問題再逐步加到 100% | 「小流量先試、出事影響最小」 |

> 兩者都在回答同一個問題：**怎麼換新模型而不讓使用者一次承受全部風險。**
> 本模組只建立概念；實際切流量在 capstone 階段才動手。

## 2. 沙盒步驟（Layer 1：照編號逐個跑，只學最小可用動詞）

每個沙盒彼此**不互相 import**，都能獨立執行。先把工具玩熟，再回 workspace 接。

```bash
cd modules/m5-automation/sandbox
```

**(1) Prefect：把兩支函式串成一個 flow**（只學 `@task` / `@flow` / 本地 run）

```bash
pip install prefect
cd prefect
python flow.py
```
看終端印出每個 task 的狀態流轉與最終 accuracy。細節見 `prefect/README.md`。

**(2) pytest：先在本機跑通測試**（CI 要跑的東西，先確認本機綠）

```bash
cd ..            # 回到 sandbox/
pip install pytest
python -m pytest tests/ -v
```
`tests/test_sample.py` 含一個 metric 單元測試與一組資料驗證測試。

**(3) GitHub Actions：看懂最小 CI 範本**（只學 `on` / `runs-on` / `steps`）

打開 `github-actions/ci.yml` 對照 `github-actions/README.md`，理解
「push -> 跑 pytest -> 失敗就變紅」這條品質門檻。真實使用時把它複製到
repo 根目錄的 `.github/workflows/`，GitHub 才會執行（README 有指令）。

## 3. 整合任務（Layer 2：到 workspace 把這個工具接上去）

回到 `workspace/`，把 M3/M4 累積的步驟編成 pipeline，並加上 CI gate：

```bash
cd ../../workspace      # 你的專案主線（跨模組長大）
```

TODO 提示：

- [ ] **用 Prefect 編排訓練流程**：把 workspace 既有的「載入特徵 -> 訓練 -> 評估」
      三支函式各加 `@task`，再用一個 `@flow` 串起來（模式照搬 `sandbox/prefect/flow.py`）。
- [ ] **加資料驗證 task**：在訓練前插一個 task，檢查輸入特徵的 schema 與值域，
      不通過就讓 flow 失敗（對應「資料驗證」測試）。
- [ ] **加品質門檻 gate**：評估後若 `accuracy < 門檻`，讓 flow 報錯而非繼續，
      確保爛模型不會被往下游送（對應「品質門檻 gate」）。
- [ ] **把 workspace 的測試接上 CI**：複製 `sandbox/github-actions/ci.yml` 到
      repo 根目錄 `.github/workflows/`，把測試路徑改成 workspace 的 `tests/`，
      push 後在 GitHub Actions 分頁確認變綠。
- [ ] **（觀念題，不必實作）** 想一下：若要把新模型安全換上線，你會選 canary 還是
      blue-green？把理由寫進 workspace 的 README。

## 4. 卡住怎麼辦

- **重置到本模組起點**：把上一模組的已知良好快照覆蓋回 workspace
  ```bash
  cp -r checkpoints/after-m4/* workspace/
  ```
- **對照本模組的正解**：完成整合後，與下一個 checkpoint 比對
  ```bash
  ls checkpoints/after-m5/        # 本模組整合任務的參考終點
  diff -r workspace/ checkpoints/after-m5/
  ```
- **沙盒跑不動**：沙盒範例本身就是可跑的正解，直接照打、只改參數即可；
  Prefect 跑不起來先確認 `pip install prefect`、pytest 失敗先在本機 `-v` 看哪條 assert 掛。

## 5. 檢核題

1. CI、CD、CT 三者的**觸發來源**各是什麼？為什麼 ML 比一般軟體多了 CT？
2. 「資料驗證測試」和「單元測試」測的東西有何不同？各舉一個例子。
3. 什麼是「品質門檻 gate」？它擋下的是什麼情況？放在 pipeline 的哪一步？
4. `@task` 與 `@flow` 的職責分別是什麼？為什麼本地 `python flow.py` 不需要 server？
5. canary 與 blue-green 的核心差別是什麼？它們共同要解決的問題是哪一個？
