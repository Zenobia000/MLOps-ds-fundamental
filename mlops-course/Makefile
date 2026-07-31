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

# ── 課程統一環境 ──────────────────────────────────────────────
# 全課共用一個 uv 建的 .venv，版本由 uv.lock 鎖死（同一個 commit → 同一組版本）。
# 這正是 m2 在教的「環境版本也要版控」，課程自己先做到。
VENV        := .venv
VENV_PYTHON := $(VENV)/bin/python

# 有 .venv 就自動用它，沒有才退回系統 python3。
# 好處：學生忘了 source activate，make 指令照樣跑在正確環境裡。
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)

# 教學 notebook（只含 modules/；capstone/ 的另外測，見 nb-test）
NOTEBOOKS := $(shell find $(MODULES) -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' 2>/dev/null | sort)

.DEFAULT_GOAL := help

.PHONY: help setup mlflow-ui \
        m1 m2 m3 m4 m5 m6 \
        workspace-m1 workspace-m2 workspace-m3 workspace-m4 workspace-m5 workspace-m6 \
        checkpoint-m1 checkpoint-m2 checkpoint-m3 checkpoint-m4 checkpoint-m5 \
        nb nb-test nb-strip nb-verify check-env \
        capstone-up

# ── 總覽 ──────────────────────────────────────────────────────
help: ## 顯示所有可用指令
	@echo "MLOps 教學 repo — 可用指令："
	@echo ""
	@echo "  環境（全課共用一個 .venv，版本由 uv.lock 鎖死）："
	@echo "    make setup            建環境 + 裝全課依賴（uv sync --extra dev）"
	@echo "    make check-env        確認站在正確環境上（裝完 / 卡住時先跑這個）"
	@echo "    make mlflow-ui        啟動本地 MLflow UI（讀 $(MLRUNS_DIR)，port $(MLFLOW_PORT)）"
	@echo ""
	@echo "  模組導覽（只印當前模組 README 路徑，漸進解鎖）："
	@echo "    make m1 .. m6         印出該模組 README 路徑與一句話主題"
	@echo ""
	@echo "  主線模板（把填空骨架解鎖進 workspace/，不覆蓋已有檔）："
	@echo "    make workspace-m1 .. workspace-m6"
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
setup: ## 一次裝好全課依賴（uv sync，版本由 uv.lock 鎖定）
	@command -v uv >/dev/null || { \
	  echo "[setup] 找不到 uv。先安裝："; \
	  echo "        curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	  echo "        （或改用 conda，見 SETUP.md 方式 B）"; exit 1; }
	uv sync --extra dev
	@echo "[setup] 完成。環境在 ./$(VENV)/，Python 版本依 .python-version（3.11）。"
	@echo "[setup] make 指令會自動用 ./$(VENV)，不必手動 activate。"
	@echo "[setup] 要在終端機直接跑 python/pytest 才需要：source $(VENV)/bin/activate"
	@echo "[setup] 進階 HPO（Ray Tune / FLAML）：uv sync --extra dev --extra advanced"

check-env: ## 確認目前站在課程統一環境上（裝完、卡住時先跑這個）
	@echo "[check-env] make 會用的 Python：$(PYTHON)"
	@test -x $(VENV_PYTHON) || { \
	  echo "  ✗ 找不到 ./$(VENV)/ —— 還沒建環境，先跑：make setup"; exit 1; }
	@$(VENV_PYTHON) -c "import sys; v=sys.version_info; \
	  print(f'  Python {v.major}.{v.minor}.{v.micro}'); \
	  sys.exit(0 if (v.major, v.minor) == (3, 11) else 1)" \
	  || { echo "  ✗ 不是 Python 3.11 —— 刪掉 $(VENV)/ 再 make setup"; exit 1; }
	@uv lock --check >/dev/null 2>&1 \
	  && echo "  ✓ uv.lock 與 pyproject.toml 一致" \
	  || echo "  ! uv.lock 已過期（pyproject.toml 改過）→ 跑 uv lock 更新後 commit"
	@$(VENV_PYTHON) -c "import mlflow, optuna, sklearn, feast, evidently; print('  ✓ 核心套件可 import')"
	@echo "  ✓ 環境就緒"

mlflow-ui: ## 啟動本地 MLflow UI
	@echo "[mlflow-ui] 開瀏覽器到 http://127.0.0.1:$(MLFLOW_PORT)"
	mlflow ui --backend-store-uri $(MLRUNS_DIR) --port $(MLFLOW_PORT)

# ── 模組導覽（漸進解鎖：只指路，不展開全部教材）─────────────
m1: ## m1 全景 / 基礎（階 0：純 Python + Git）
	@echo "[m1] 全景 / 基礎 — 純 sklearn baseline + Git（階 0）"
	@echo "     讀這份 README：$(MODULES)/m1-foundations/README.md"
	@echo "     解鎖主線模板：make workspace-m1"

m2: ## m2 追蹤・調參・版本化（階 1 MLflow + 階 2 Optuna + 階 3 DVC）
	@echo "[m2] 追蹤・調參・版本化 — MLflow + Optuna + DVC（階 1–3）"
	@echo "     讀這份 README：$(MODULES)/m2-tracking-tuning-versioning/README.md"
	@echo "     解鎖主線模板：make workspace-m2"

m3: ## m3 特徵商店（階 4 Feast）
	@echo "[m3] 特徵商店 — Feast（階 4）"
	@echo "     讀這份 README：$(MODULES)/m3-feature-store/README.md"
	@echo "     解鎖主線模板：make workspace-m3"

m4: ## m4 服務化（階 5 Docker / 階 6 FastAPI / 階 7 BentoML / 階 8 PyTorch）
	@echo "[m4] 服務化 — FastAPI / Docker / BentoML / PyTorch+ONNX（階 5–8）"
	@echo "     讀這份 README：$(MODULES)/m4-serving/README.md"
	@echo "     解鎖主線模板：make workspace-m4"

m5: ## m5 自動化（階 9 Prefect + 階 10 GitHub Actions）
	@echo "[m5] 自動化 — Prefect + GitHub Actions（階 9–10）"
	@echo "     讀這份 README：$(MODULES)/m5-automation/README.md"
	@echo "     解鎖主線模板：make workspace-m5"

m6: ## m6 監控・治理・收尾（階 11 Evidently + 整合）
	@echo "[m6] 監控・治理・收尾 — Evidently + 治理 + 解鎖 capstone（階 11）"
	@echo "     讀這份 README：$(MODULES)/m6-monitoring-governance/README.md"
	@echo "     解鎖主線模板：make workspace-m6"

# ── 解鎖：把 modules/mN-*/templates/ 填空骨架複製進 workspace/ ──
# 規則：已存在的檔案一律跳過（不覆蓋學生進度）；只新增尚未出現的路徑。
# 用法：$(call unlock_workspace,<N>,<模組目錄名>)
define unlock_workspace
	@src="$(MODULES)/$(2)/templates"; \
	if [ ! -d "$$src" ]; then \
	  echo "[workspace-m$(1)] 找不到模板目錄：$$src"; exit 1; \
	fi; \
	echo "[workspace-m$(1)] 解鎖模板 → workspace/（已存在則 skip）"; \
	mkdir -p workspace; \
	find "$$src" -type f | sort | while read -r f; do \
	  rel=$${f#$$src/}; \
	  dest="workspace/$$rel"; \
	  mkdir -p "$$(dirname "$$dest")"; \
	  if [ -e "$$dest" ]; then \
	    echo "  skip  $$rel"; \
	  else \
	    cp "$$f" "$$dest"; \
	    echo "  add   $$rel"; \
	  fi; \
	done; \
	echo "[workspace-m$(1)] 下一步：打開 workspace/ 搜尋 TODO(M$(1)- 對照 sandbox 填空"
endef

workspace-m1: ## 解鎖 M1 主線填空模板（train.py）
	$(call unlock_workspace,1,m1-foundations)

workspace-m2: ## 解鎖 M2 模板（train_tracking.py + conf/）
	@test -f workspace/train.py || (echo "[workspace-m2] 請先 make workspace-m1 並完成 M1"; exit 1)
	$(call unlock_workspace,2,m2-tracking-tuning-versioning)

workspace-m3: ## 解鎖 M3 模板（prepare_features.py + feature_repo/）
	@test -f workspace/train.py || (echo "[workspace-m3] 請先完成 M1（make workspace-m1）"; exit 1)
	$(call unlock_workspace,3,m3-feature-store)

workspace-m4: ## 解鎖 M4 模板（services/）
	@test -f workspace/train.py || (echo "[workspace-m4] 請先完成 M1（make workspace-m1）"; exit 1)
	$(call unlock_workspace,4,m4-serving)

workspace-m5: ## 解鎖 M5 模板（flow.py + tests/）
	@test -f workspace/train.py || (echo "[workspace-m5] 請先完成 M1（make workspace-m1）"; exit 1)
	$(call unlock_workspace,5,m5-automation)

workspace-m6: ## 解鎖 M6 模板（monitoring/）
	@test -f workspace/train.py || (echo "[workspace-m6] 請先完成 M1（make workspace-m1）"; exit 1)
	$(call unlock_workspace,6,m6-monitoring-governance)

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
	$(PYTHON) -m jupyterlab

nb-test: ## 實際執行每一本 notebook，確認全部可從頭跑到尾
	@echo "[nb-test] 執行 modules/ 的 $(words $(NOTEBOOKS)) 本 notebook"
	$(PYTHON) -m pytest --nbmake $(NOTEBOOKS) -q
	@echo ""
	@echo "[nb-test] capstone/ 的 notebook 需要 capstone 環境，另外測："
	@echo "          cd capstone/smart-factory-mlops && $(PYTHON) -m pytest --nbmake notebooks/ -q"

nb-strip: ## 清掉 notebook 的執行輸出（commit 前跑，保持 diff 乾淨）
	$(PYTHON) -m nbstripout $(NOTEBOOKS)
	@echo "[nb-strip] 已清除 modules/ notebook 的 outputs"

nb-verify: ## 檢查 notebook 是否殘留 outputs（CI 用，不修改檔案）
	@$(PYTHON) -m nbstripout --verify $(NOTEBOOKS) \
	  && echo "[nb-verify] OK：所有 notebook 皆無殘留 outputs" \
	  || (echo "[nb-verify] 失敗：有 notebook 帶著 outputs，請先跑 make nb-strip"; exit 1)

# ── 完整專案（最後才解鎖；目前為佔位）──────────────────────
capstone-up: ## 佔位：m6 解鎖後才啟動 capstone 完整環境
	@echo "[capstone-up] capstone 在 m6 才解鎖。"
	@echo "              走到 m6 後，這裡會啟動 capstone/smart-factory-mlops/ 的完整環境。"
	@echo "              現在請專注當前模組：make help 看進度。"
