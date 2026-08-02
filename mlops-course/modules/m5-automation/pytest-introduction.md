# pytest 入門：怎麼用、怎麼寫、怎麼接 CI

> 本頁回答：**pytest 怎麼跑、測試函式怎麼寫、和 GitHub Actions 怎麼接成品質門檻。**  
> 動手範例：[`sandbox/tests/test_sample.py`](./sandbox/tests/test_sample.py)  
> CI 範本：[`sandbox/github-actions/`](./sandbox/github-actions/README.md)  
> 概念圖：[`assets/m5-pytest-ci.png`](./assets/m5-pytest-ci.png)

---

## 1. 一句話定位

**pytest 是 Python 測試框架**——你寫一堆 `test_*.py` / `test_*` 函式，用 `assert` 宣告「應該成立的事」；失敗就非 0 結束。  
接上 CI 後：**push → 自動跑同一條 pytest → 紅就擋、綠才過。**

| 沒有測試 | 有 pytest + CI |
| :--- | :--- |
| 靠人記得手動檢查 | 每次 push 自動檢查 |
| 改壞 metric / schema 上線才發現 | 本機或 CI 立刻紅燈 |
| 「應該沒問題吧」 | 有可重複執行的證據 |

---

## 2. 怎麼用（指令）

### 安裝

```bash
pip install pytest
# 課程環境 pyproject.toml 通常已含 pytest
```

### 本課沙盒怎麼跑

```bash
# 從 repo 根，或先 cd 到 m5-automation/sandbox
cd mlops-course/modules/m5-automation/sandbox
python -m pytest tests/ -v
```

常用旗標：

| 指令 | 意思 |
| :--- | :--- |
| `python -m pytest` | 用目前環境的 pytest（比直接打 `pytest` 穩） |
| `tests/` | 要蒐集的目錄 |
| `-v` | verbose，列出每條測試名稱與結果 |
| `-k accuracy` | 只跑名字含 `accuracy` 的測試 |
| `-x` | 第一個失敗就停 |
| `--tb=short` | 縮短失敗 traceback |

成功時會看到 `PASSED`；失敗時對應 `assert` 那一行會標紅，exit code ≠ 0。

> **口訣：CI 上跑什麼，本機就先跑什麼。**  
> 本課 CI 指令：`python -m pytest modules/m5-automation/sandbox/tests/ -v`

---

## 3. 怎麼寫（規則與範例）

### 3.1 pytest 怎麼找到你的測試

| 規則 | 說明 |
| :--- | :--- |
| 檔名 | 預設蒐集 `test_*.py` 或 `*_test.py` |
| 函式名 | 以 `test_` 開頭 |
| 類別名（可選） | `Test*` 且方法名 `test_*` |
| 斷言 | 用內建 `assert`，不必寫 `self.assertEqual` |

### 3.2 最小單元測試

對應本課 `accuracy()`：

```python
def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 與 y_pred 長度必須相同")
    if not y_true:
        raise ValueError("不能對空資料計算 accuracy")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def test_accuracy_all_correct():
    assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_accuracy_half_correct():
    assert accuracy([0, 1, 0, 1], [0, 1, 1, 0]) == 0.5
```

### 3.3 測「應該拋錯」

```python
import pytest

def test_accuracy_rejects_empty():
    with pytest.raises(ValueError):
        accuracy([], [])
```

### 3.4 ML 專屬：資料驗證

不只測程式邏輯，還要測「進 pipeline 的資料長得對嗎」：

```python
def test_toy_data_schema():
    data = load_toy_data(n=20)
    assert len(data) == 20
    for row in data:
        assert set(row.keys()) == {"x", "y"}


def test_toy_data_value_range():
    data = load_toy_data(n=20)
    for row in data:
        assert 0.0 <= row["x"] <= 1.0
        assert row["y"] in (0, 1)
```

### 3.5 本課建議測這幾類

| 類型 | 測什麼 | 本課例子 |
| :--- | :--- | :--- |
| **單元測試** | 小函式算得對不對 | `test_accuracy_*` |
| **資料驗證** | schema / 值域 | `test_toy_data_*` |
| **品質門檻（進階）** | 模型夠不夠好 | `assert accuracy >= 0.85` |
| **Pipeline smoke（進階）** | 整條流程能跑通 | 小資料跑完 flow |

### 3.6 寫測試的實務習慣

| 習慣 | 原因 |
| :--- | :--- |
| 函式名說清楚情境 | `test_accuracy_half_correct` 比 `test1` 好除錯 |
| 一測一事 | 失敗時立刻知道壞哪條假設 |
| 隨機要固定 `SEED` | 避免 CI flaky（偶發紅） |
| 本機綠了再 push | 少浪費 Actions 分鐘、少吵團隊 |

---

## 4. 怎麼與 CI 結合

### 4.1 關係圖

```text
你寫 test_*.py
      │
      │  本機：python -m pytest tests/ -v
      ▼
   綠燈？ ──否──▶ 先修，不要 push
      │是
      ▼
   git push
      │
      ▼
GitHub Actions（.github/workflows/ci.yml）
  on: push
  steps: checkout → setup-python → pip install pytest → pytest
      │
      ├─ 全 PASSED → workflow 綠勾
      └─ 任一失敗 → step 非 0 → workflow 紅叉（品質門檻）
```

### 4.2 最小 workflow（對照沙盒）

[`sandbox/github-actions/ci.yml`](./sandbox/github-actions/ci.yml) 精簡結構：

```yaml
name: ci

on:
  push:
    branches: ["**"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest
      - name: Run pytest
        run: |
          python -m pytest modules/m5-automation/sandbox/tests/ -v
```

三個欄位先記牢：

| 欄位 | 問題 | 本範例 |
| :--- | :--- | :--- |
| `on` | 什麼時候跑？ | `push` |
| `runs-on` | 在哪跑？ | `ubuntu-latest` |
| `steps` | 跑什麼？ | checkout → Python → 裝 pytest → 跑 pytest |

### 4.3 為什麼 pytest 失敗就能當「gate」

1. pytest 失敗 → process exit code ≠ 0  
2. GitHub Actions 該 step 失敗 → job / workflow 變紅  
3. PR / 分支狀態顯示未通過 → 提醒（或設定 branch protection 擋合併）

這就是 CI 品質門檻：**壞測試結果進不了「已驗證」狀態，不靠口頭提醒。**

### 4.4 檔案必須放對位置

| 位置 | 用途 |
| :--- | :--- |
| `sandbox/github-actions/ci.yml` | 給你看的**範本** |
| **repo 根** `.github/workflows/*.yml` | GitHub **唯一**會自動執行的位置 |

啟用範本（從 `mlops-course/` 視路徑調整）：

```bash
mkdir -p .github/workflows
cp mlops-course/modules/m5-automation/sandbox/github-actions/ci.yml \
   .github/workflows/ci.yml
# 把 pytest 路徑改成你真正的 tests/ 後再 commit / push
```

> 本 repo 根目錄若已有 [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)，那是整課正式 CI；沙盒 YAML 是最小教學版，兩者目的不同。

### 4.5 本機 ↔ CI 對齊檢查清單

- [ ] 本機指令與 workflow 最後一步**相同**（或只差路徑前綴）
- [ ] Python 版本一致（本課 3.11）
- [ ] 依賴在 CI 有裝（至少 `pytest`）
- [ ] 測試路徑在 runner 上存在（相對 repo 根）
- [ ] 隨機性有固定 seed，避免 flaky

---

## 5. 和 Prefect 怎麼分工

| 工具 | 管什麼 | 典型觸發 |
| :--- | :--- | :--- |
| **pytest + GitHub Actions** | 程式／測試有沒有壞（CI） | `git push` / PR |
| **Prefect** | 訓練 pipeline 步驟怎麼跑（編排／CT） | 排程、新資料、手動 `python flow.py` |

可以一起用：CI 裡跑 pytest（含資料驗證）；Prefect flow 裡再放 `quality_gate` 擋爛模型。  
本課先把「本機 pytest → CI 同一條」打穩。

---

## 6. 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 測試檔不叫 `test_*.py` | pytest 蒐集不到 | 改檔名 |
| 只在本機跑過、CI 路徑寫錯 | 本機綠、Actions 紅 | 路徑相對 repo 根 |
| workflow 放在 sandbox 沒複製 | GitHub 永不執行 | 放到 `.github/workflows/` |
| 隨機未固定 seed | CI 偶發失敗 | `SEED = 42` |
| CI 跑超長完整訓練 | feedback 太慢 | CI 跑快測；長訓練另開 CT |

---

## 7. 檢核

1. 為什麼建議用 `python -m pytest` 而不是只打 `pytest`？  
2. `test_accuracy_rejects_empty` 用 `pytest.raises` 在測什麼？  
3. 「資料驗證測試」和「metric 單元測試」差在哪？  
4. pytest 失敗如何讓 GitHub Actions workflow 變紅？  
5. 沙盒 `ci.yml` 為什麼還要複製到 `.github/workflows/` 才會生效？

動手：

```bash
cd mlops-course/modules/m5-automation/sandbox
python -m pytest tests/ -v
```

全綠後，對照 [`github-actions/ci.yml`](./sandbox/github-actions/ci.yml) 最後一步是否同一條指令邏輯。
