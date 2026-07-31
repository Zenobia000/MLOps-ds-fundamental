# 環境安裝（SETUP）

> 目標：**一次裝好全課依賴**，之後每個模組直接開跑，不必每模組重裝。
> 全課統一 **Python 3.11**（torch / onnxruntime / feast 在 3.11 相容性最穩）。

---

## 0. 前置需求

- **Python 3.11**（用 uv 的話**不必自己準備**——uv 會依 `.python-version` 自動取得）。
- **Git**（m1 就會用到）。
- **Docker**（m4 服務化才需要；前面模組沒有也能跑）。
- 作業系統：Linux / macOS 皆可；Windows 建議用 WSL2。

> 不確定裝好沒？建完環境跑 `make check-env`，它會一次檢查 Python 版本、lock 是否最新、核心套件能不能 import。

---

## 1. 安裝方式 A：uv（推薦，最快）

[uv](https://github.com/astral-sh/uv) 是高速的 Python 套件管理器，會自動處理虛擬環境與 Python 版本。

```bash
# 1) 安裝 uv（若還沒裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 進到課程根目錄
cd mlops-course

# 3) 一次建好環境 + 裝依賴（uv 會自己建 .venv 並取得 Python 3.11）
make setup
#    等同於：uv sync --extra dev

# 4) 確認站對地方
make check-env

# （可選）連進階 HPO 套件一起裝
uv sync --extra dev --extra advanced
```

**Python 版本不用自己指定**：`mlops-course/.python-version` 寫死 3.11，
uv 會照著取得（本機沒有就自動下載），不會用到你系統上的其他版本。

**版本由 `uv.lock` 鎖死**：`uv sync` 是照 lock 檔安裝，
所以**今天裝、三個月後裝、同學的機器上裝，拿到的是同一組版本**。
這正是 m2 在教的「環境版本也要版控」——課程自己先做到。

> 要不要 `source .venv/bin/activate`？
> **跑 `make` 指令不用**——Makefile 偵測到 `.venv` 就會自動用它。
> 只有你想在終端機直接敲 `python` / `pytest` / `feast` / `mlflow` 時才需要啟用。

---

## 2. 安裝方式 B：conda

```bash
# 1) 建立並啟用環境
conda create -n mlops-course python=3.11 -y
conda activate mlops-course

# 2) 進到課程根目錄
cd mlops-course

# 3) 一次裝好全課依賴（含 notebook 品質工具）
pip install -e ".[dev]"

# （可選）連進階 HPO 套件一起裝
pip install -e ".[dev,advanced]"
```

> 用 conda 時 `make` 抓不到 `.venv`，會退回系統 `python3`。
> 請確保 conda 環境已 `activate`，或明確指定：`make nb-test PYTHON=$(which python)`。

> 兩種方式都用 `pyproject.toml` 當單一依賴來源。
>
> **但只有 uv 會用到 `uv.lock`。** conda + `pip install -e .` 走的是版本下限（`>=`），
> 解析出來的版本可能跟 lock 檔不同——想跟 CI、跟同學拿到完全一樣的版本，用 uv。
> conda 這條路留給已經習慣 conda 的人，代價是「版本可能有出入」。

---

## 3. 驗證安裝

```bash
make check-env      # 一次檢查：Python 3.11 / uv.lock 是否最新 / 核心套件可 import
```

想再逐批確認（需先 `source .venv/bin/activate`，或用 `.venv/bin/python` 取代 `python`）：

```bash
python -c "import mlflow, optuna, sklearn, pandas, numpy; print('core OK')"
python -c "import fastapi, uvicorn, bentoml; print('serving OK')"
python -c "import prefect, evidently; print('automation+monitoring OK')"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

每行印出 `OK` / 版本號就代表該批套件就緒。`cuda? False` 是正常的（見第 6 節無 GPU 後援）。

---

## 4. 啟動本地 MLflow UI（m2 開始會用）

MLflow 預設把實驗追蹤寫進當前目錄的 `mlruns/`。在**你正在跑腳本的同一個目錄**啟動 UI：

```bash
# 方式一：用 Makefile（推薦）
make mlflow-ui

# 方式二：直接呼叫
mlflow ui --backend-store-uri ./mlruns --port 5000
```

啟動後開瀏覽器到 **http://127.0.0.1:5000**，就能看到每次 run 的參數、指標、模型。

> 小提醒：MLflow UI 讀的是「啟動 UI 那個目錄底下的 `mlruns/`」。如果看不到 run，先確認你跑訓練腳本的目錄和啟動 UI 的目錄一致。

---

## 5. 各工具會用到的額外服務（用到再起，不用提前裝）

| 工具 | 何時用到 | 啟動方式 |
| :--- | :--- | :--- |
| MLflow UI | m2+ | `make mlflow-ui`（讀本地 `mlruns/`） |
| DVC remote | m2（階 3） | 用本地資料夾當 remote 即可，無需雲端帳號 |
| Feast | m3 | 預設 SQLite + 本地檔案 registry，零外部依賴 |
| Docker | m4 | 需先裝 Docker Desktop / Engine |
| Prefect | m5 | `prefect server start`（本地）或直接本地 run flow |

---

## 6. 無 GPU 後援說明（重要）

本課程**全程可在純 CPU 跑完**，不需要顯卡。

- **PyTorch（m4 階 8）**：`make setup` 會裝 CPU 版 `torch` / `torchvision`。沙盒用的是預訓練小模型（如 ResNet）做推論 + 匯出 ONNX，CPU 幾秒內完成，不需要訓練大模型。
- **Optuna（m2 階 2）**：玩具資料的 trial 數量很小，CPU 即可。
- 若 `torch.cuda.is_available()` 回傳 `False`，**完全正常**，照常往下走即可。

> 如果你**有** GPU 且想用：請依 [PyTorch 官網](https://pytorch.org/get-started/locally/) 的指令，先單獨裝對應 CUDA 版本的 `torch`，再裝其餘套件。注意這會讓你的版本偏離 `uv.lock`，屬於刻意的例外。

---

## 7. 疑難排解

| 症狀 | 可能原因 / 解法 |
| :--- | :--- |
| `pip install -e .` 卡在 torch | torch 體積大、首次下載慢；耐心等，或先單獨 `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `mlflow ui` 看不到任何 run | 啟動 UI 的目錄與跑腳本的目錄不一致；切到有 `mlruns/` 的目錄再啟動 |
| `import bentoml` 報錯 | 確認在已啟用的虛擬環境內；重跑第 3 節驗證指令 |
| Python 不是 3.11 | uv：刪掉 `.venv/` 再 `make setup`（版本由 `.python-version` 決定）；conda：`python=3.11` 重建 |
| `make setup` 說找不到 uv | 照第 1 節裝 uv，或改走第 2 節的 conda |
| `make check-env` 說 uv.lock 已過期 | 有人改過 `pyproject.toml`：跑 `uv lock` 更新並把 `uv.lock` 一起 commit |
| 版本跟同學不一樣 | 確認你走的是 uv（會讀 `uv.lock`）而非 conda + pip（只看 `>=` 下限） |
| 某模組改壞了 | 不要硬修，直接從 `checkpoints/after-m(N-1)/` 重置（見各模組 README「卡住怎麼辦」） |

裝好了就回 **[README.md](./README.md)**，從 `make m1` 開始。
