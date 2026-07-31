# =============================================================================
# Makefile — Smart Factory MLOps Capstone
# -----------------------------------------------------------------------------
# 常用指令封裝，全部從 repo 根執行（套件根 = src/）。
# 指令對應實際 src 模組與 services；先 `cp .env.example .env` 並調整變數。
# 用 `make help` 看全部指令。
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── 統一環境 ──────────────────────────────────────────────────
# capstone 有自己的 .venv 與 uv.lock（與課程 mlops-course/ 那套分開，
# 因為它是可獨立部署的生產骨架）。版本由 uv.lock 鎖死，兩個 Dockerfile
# 的 requirements-*.txt 也是從同一份 lock 產生的，三者不會漂移。
VENV        := .venv
VENV_PYTHON := $(VENV)/bin/python

# 有 .venv 就自動用它；沒有才退回 python3。可用 PYTHON=... 覆蓋。
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)

# 允許用 `make train-tabular ARGS="model=xgboost"` 傳額外參數給 python 模組。
ARGS ?=

.PHONY: help setup check-env up feast-apply train-tabular train-ts train-vision tune serve test lint monitor pipeline

help:  ## 顯示所有可用指令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## 建環境 + 裝依賴（uv sync，版本由 uv.lock 鎖定）
	@command -v uv >/dev/null || { \
	  echo "[setup] 找不到 uv。先安裝：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync --extra dev
	@echo "[setup] 完成。環境在 ./$(VENV)/（Python 3.11，依 .python-version）。"
	@echo "[setup] make 指令會自動用它，不必手動 activate。"

check-env:  ## 確認站在 capstone 的統一環境上
	@echo "[check-env] make 會用的 Python：$(PYTHON)"
	@test -x $(VENV_PYTHON) || { echo "  ✗ 找不到 ./$(VENV)/ —— 先跑 make setup"; exit 1; }
	@$(VENV_PYTHON) -c "import sys; v=sys.version_info; \
	  print(f'  Python {v.major}.{v.minor}.{v.micro}'); \
	  sys.exit(0 if (v.major, v.minor) == (3, 11) else 1)" \
	  || { echo "  ✗ 不是 Python 3.11 —— 刪掉 $(VENV)/ 再 make setup"; exit 1; }
	@uv lock --check >/dev/null 2>&1 \
	  && echo "  ✓ uv.lock 與 pyproject.toml 一致" \
	  || echo "  ! uv.lock 已過期 → 跑 uv lock 更新後 commit"
	@$(VENV_PYTHON) -c "import mlflow, feast, bentoml, torch; print('  ✓ 核心套件可 import')"
	@echo "  ✓ 環境就緒"

up:  ## 啟動本地平台（MLflow + Feast + 監控），見 docker/docker-compose.yml
	docker compose -f docker/docker-compose.yml up -d

feast-apply:  ## 套用 Feast 特徵定義（建立/更新 feature views）
	cd feature_repo && ../$(VENV)/bin/feast apply

train-tabular:  ## 訓練 XGBoost 預測性維護模型（結構化+時序）
	$(PYTHON) -m src.training.train model=xgboost data=sensors $(ARGS)

train-ts:  ## 訓練 LSTM 產能需求預測模型（時序）
	$(PYTHON) -m src.training.train model=lstm data=demand $(ARGS)

train-vision:  ## 訓練 ResNet 瑕疵檢測模型並匯出 ONNX（影像）
	$(PYTHON) -m src.training.train model=resnet data=images $(ARGS)

tune:  ## 用 Optuna 自動調參（預設 xgboost，可用 ARGS="model=resnet"）
	$(PYTHON) -m src.tuning.optuna_tuning model=xgboost $(ARGS)

serve:  ## 啟動 BentoML 推論服務（表格 + 影像雙模型 API）
	bentoml serve services.service:SmartFactoryService --reload

test:  ## 跑全部測試並輸出覆蓋率
	pytest --cov=src --cov-report=term-missing

lint:  ## 執行 ruff + black 檢查（不改檔）
	ruff check src tests
	black --check src tests

monitor:  ## 產出 Evidently 漂移報告
	$(PYTHON) -m src.monitoring.drift

pipeline:  ## 重跑 DVC 端到端管線（prepare→features→train→evaluate）
	dvc repro
