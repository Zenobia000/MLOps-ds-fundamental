# ──────────────────────────────────────────────────────────────
# MLOps 入門教學 repo — 常用指令封裝
# 用法：make help 看全部指令
# 設計：以「漸進解鎖」為原則，mN 只印出當前模組 README 路徑，
#       不一次攤開全部教材；capstone 留佔位、最後才解鎖。
# ──────────────────────────────────────────────────────────────

# MLflow UI 設定（可用環境變數覆蓋：make mlflow-ui MLFLOW_PORT=5001）
MLFLOW_PORT ?= 5000
MLRUNS_DIR  ?= ./mlruns

# 模組根目錄
MODULES := modules

# Python 直譯器（venv 裡通常有 python，系統環境常常只有 python3；可覆蓋）
PYTHON ?= python3

# 教學 notebook（只含 modules/；capstone/ 的另外測，見 nb-test）
NOTEBOOKS := $(shell find $(MODULES) -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' 2>/dev/null | sort)


.DEFAULT_GOAL := help

.PHONY: help setup mlflow-ui \
        nb nb-test nb-strip nb-verify \
        m1 m2 m3 m4 m5 m6 \
        checkpoint-m1 checkpoint-m2 checkpoint-m3 checkpoint-m4 checkpoint-m5 \
        capstone-up

# ── 總覽 ──────────────────────────────────────────────────────
help: ## 顯示所有可用指令
	@echo "MLOps 教學 repo — 可用指令："
	@echo ""
	@echo "  環境："
	@echo "    make setup            一次裝好全課依賴（pip install -e .）"
	@echo "    make mlflow-ui        啟動本地 MLflow UI（讀 $(MLRUNS_DIR)，port $(MLFLOW_PORT)）"
	@echo ""
	@echo "  模組導覽（只印當前模組 README 路徑，漸進解鎖）："
	@echo "    make m1 .. m6         印出該模組 README 路徑與一句話主題"
	@echo ""
	@echo "  存檔（把 workspace 複製成已知良好快照）："
	@echo "    make checkpoint-m1 .. checkpoint-m5"
	@echo ""
	@echo "  教學 notebook（m1 EDA / m2 Optuna 視覺化 / m3 洩漏對照 / m6 漂移探索）："
	@echo "    make nb               啟動 JupyterLab"
	@echo "    make nb-test          實際執行每一本，確認可從頭跑到尾"
	@echo "    make nb-strip         清掉執行輸出（commit 前跑）"
	@echo "    make nb-verify        檢查有無殘留輸出（CI 用，不改檔）"
	@echo ""
	@echo "  完整專案（最後才解鎖）："
	@echo "    make capstone-up      佔位：m6 解鎖後才啟動 capstone 環境"
	@echo ""

# ── 環境 ──────────────────────────────────────────────────────
setup: ## 一次裝好全課依賴
	pip install -e .
	@echo "[setup] 依賴安裝完成。進階 HPO 可另外裝：pip install -e \".[advanced]\""

mlflow-ui: ## 啟動本地 MLflow UI
	@echo "[mlflow-ui] 開瀏覽器到 http://127.0.0.1:$(MLFLOW_PORT)"
	mlflow ui --backend-store-uri $(MLRUNS_DIR) --port $(MLFLOW_PORT)

# ── 模組導覽（漸進解鎖：只指路，不展開全部教材）─────────────
m1: ## m1 全景 / 基礎（階 0：純 Python + Git）
	@echo "[m1] 全景 / 基礎 — 純 sklearn baseline + Git（階 0）"
	@echo "     讀這份 README：$(MODULES)/m1-foundations/README.md"

m2: ## m2 追蹤・調參・版本化（階 1 MLflow + 階 2 Optuna + 階 3 DVC）
	@echo "[m2] 追蹤・調參・版本化 — MLflow + Optuna + DVC（階 1–3）"
	@echo "     讀這份 README：$(MODULES)/m2-tracking-tuning-versioning/README.md"

m3: ## m3 特徵商店（階 4 Feast）
	@echo "[m3] 特徵商店 — Feast（階 4）"
	@echo "     讀這份 README：$(MODULES)/m3-feature-store/README.md"

m4: ## m4 服務化（階 5 Docker / 階 6 FastAPI / 階 7 BentoML / 階 8 PyTorch）
	@echo "[m4] 服務化 — FastAPI / Docker / BentoML / PyTorch+ONNX（階 5–8）"
	@echo "     讀這份 README：$(MODULES)/m4-serving/README.md"

m5: ## m5 自動化（階 9 Prefect + 階 10 GitHub Actions）
	@echo "[m5] 自動化 — Prefect + GitHub Actions（階 9–10）"
	@echo "     讀這份 README：$(MODULES)/m5-automation/README.md"

m6: ## m6 監控・治理・收尾（階 11 Evidently + 整合）
	@echo "[m6] 監控・治理・收尾 — Evidently + 治理 + 解鎖 capstone（階 11）"
	@echo "     讀這份 README：$(MODULES)/m6-monitoring-governance/README.md"

# ── 存檔：把 workspace 複製成 checkpoints/after-mN ───────────
# 共用實作：用 make 變數 N 指定模組編號，複製 workspace/ 到 checkpoints/after-mN/。
define do_checkpoint
	@mkdir -p checkpoints/after-m$(1)
	@cp -r workspace/. checkpoints/after-m$(1)/ 2>/dev/null || true
	@echo "[checkpoint-m$(1)] 已把 workspace/ 複製到 checkpoints/after-m$(1)/"
endef

checkpoint-m1: ## 把 workspace 存成 after-m1 快照
	$(call do_checkpoint,1)

checkpoint-m2: ## 把 workspace 存成 after-m2 快照
	$(call do_checkpoint,2)

checkpoint-m3: ## 把 workspace 存成 after-m3 快照
	$(call do_checkpoint,3)

checkpoint-m4: ## 把 workspace 存成 after-m4 快照
	$(call do_checkpoint,4)

checkpoint-m5: ## 把 workspace 存成 after-m5 快照
	$(call do_checkpoint,5)

# ── 教學 notebook ─────────────────────────────────────────────
# 介質選擇的判準（哪些單元該是 notebook、哪些該是腳本）見 docs/notebook-vs-script.md。
nb: ## 啟動 JupyterLab（教學 notebook 在 modules/*/sandbox/）
	@echo "[nb] 教學 notebook："
	@for f in $(NOTEBOOKS); do echo "     $$f"; done
	jupyter lab

nb-test: ## 實際執行每一本 notebook，確認全部可從頭跑到尾
	@echo "[nb-test] 執行 modules/ 的 $(words $(NOTEBOOKS)) 本 notebook"
	$(PYTHON) -m pytest --nbmake $(NOTEBOOKS) -q
	@echo ""
	@echo "[nb-test] capstone/ 的 notebook 需要 capstone 環境，另外測："
	@echo "          cd capstone/smart-factory-mlops && $(PYTHON) -m pytest --nbmake notebooks/ -q"

nb-strip: ## 清掉 notebook 的執行輸出（commit 前跑，保持 diff 乾淨）
	nbstripout $(NOTEBOOKS)
	@echo "[nb-strip] 已清除 modules/ notebook 的 outputs"

nb-verify: ## 檢查 notebook 是否殘留 outputs（CI 用，不修改檔案）
	@nbstripout --verify $(NOTEBOOKS) \
	  && echo "[nb-verify] OK：所有 notebook 皆無殘留 outputs" \
	  || (echo "[nb-verify] 失敗：有 notebook 帶著 outputs，請先跑 make nb-strip"; exit 1)

# ── 完整專案（最後才解鎖；目前為佔位）──────────────────────
capstone-up: ## 佔位：m6 解鎖後才啟動 capstone 完整環境
	@echo "[capstone-up] capstone 在 m6 才解鎖。"
	@echo "              走到 m6 後，這裡會啟動 capstone/smart-factory-mlops/ 的完整環境。"
	@echo "              現在請專注當前模組：make help 看進度。"
