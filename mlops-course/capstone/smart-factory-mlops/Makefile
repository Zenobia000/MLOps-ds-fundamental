# =============================================================================
# Makefile — Smart Factory MLOps Capstone
# -----------------------------------------------------------------------------
# 常用指令封裝，全部從 repo 根執行（套件根 = src/）。
# 指令對應實際 src 模組與 services；先 `cp .env.example .env` 並調整變數。
# 用 `make help` 看全部指令。
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Python 直譯器；可用 `make train-tabular PYTHON=python` 覆蓋（預設 python3）。
PYTHON ?= python3

# 允許用 `make train-tabular ARGS="model=xgboost"` 傳額外參數給 python 模組。
ARGS ?=

.PHONY: help up feast-apply train-tabular train-ts train-vision tune serve test lint monitor pipeline

help:  ## 顯示所有可用指令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up:  ## 啟動本地平台（MLflow + Feast + 監控），見 docker/docker-compose.yml
	docker compose -f docker/docker-compose.yml up -d

feast-apply:  ## 套用 Feast 特徵定義（建立/更新 feature views）
	cd feature_repo && feast apply

train-tabular:  ## 訓練 XGBoost 預測性維護模型（結構化+時序）
	$(PYTHON) -m src.training.train model=xgboost data=sensors $(ARGS)

train-ts:  ## 訓練 LSTM 產能需求預測模型（時序）
	$(PYTHON) -m src.training.train model=lstm data=demand $(ARGS)

train-vision:  ## 訓練 ResNet 瑕疵檢測模型並匯出 ONNX（影像）
	$(PYTHON) -m src.training.train model=resnet data=images $(ARGS)

tune:  ## 用 Optuna 自動調參（預設 xgboost，可用 ARGS="model=resnet"）
	$(PYTHON) -m src.tuning.optuna_tuning model=xgboost $(ARGS)

serve:  ## 啟動 BentoML 推論服務（表格 + 影像雙模型 API）
	bentoml serve services.service:svc --reload

test:  ## 跑全部測試並輸出覆蓋率
	pytest --cov=src --cov-report=term-missing

lint:  ## 執行 ruff + black 檢查（不改檔）
	ruff check src tests
	black --check src tests

monitor:  ## 產出 Evidently 漂移報告
	$(PYTHON) -m src.monitoring.drift

pipeline:  ## 重跑 DVC 端到端管線（prepare→features→train→evaluate）
	dvc repro
