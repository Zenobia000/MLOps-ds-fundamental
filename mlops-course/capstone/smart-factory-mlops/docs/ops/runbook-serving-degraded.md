# Runbook：推論服務降級或回 503

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 草稿
> **Owner:** Capstone 維護者
> **實例:** 每症狀一份

## 症狀（三種，嚴重程度不同）

| # | 症狀 | 嚴重度 |
| :---: | :--- | :--- |
| A | `/healthz` 回 `status: "degraded"` | 中——服務活著但功能殘缺 |
| B | 推論端點回 **503** `ServiceUnavailable` | 高——該功能完全不可用 |
| C | 回應正常，但 `model_version` 是 **`local`** | **高且隱蔽——最容易被忽略** |

---

## 症狀 C 最危險，先講它

回應是 200、預測有值、監控全綠——但 `model_version` 顯示 `local`。

這代表 **MLflow Registry 連不上，服務回退去讀本地檔案**
（見 [`../architecture/sad.md` §5.2](../architecture/sad.md#52-推論時的模型解析)）。
後果是：**你正在用一份可能過期的模型對外服務，而且沒有任何錯誤訊息。**

```bash
# 檢查目前對外服務的模型版本
curl -s -X POST http://localhost:3000/predict_maintenance \
  -H "Content-Type: application/json" \
  -d '{"readings":[{"machine_id":"m1","temperature":70,"vibration":3,"current":10}]}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('model_version =', d['model_version'])"
```

拿到 `local` → 依「registry 連不上」段處理。

> **建議設一條告警**：回應中出現 `model_version == "local"` 即通知。
> 這個回退是為了離線 / 教學能 demo 而設計的，生產環境不該長期停留在這個狀態。

---

## 診斷流程

### 步驟 1：確認是哪一個模型掛了

```bash
curl -s -X POST http://localhost:3000/healthz
# {"status":"degraded","tabular_model_loaded":true,"vision_model_loaded":false}
```

兩個模型**獨立降級**——一個掛掉另一個仍可服務。先確定範圍再往下查。

### 步驟 2：對照載入策略找斷點

服務啟動時的解析順序（`src/serving/model_loader.py`）：

```
registry (models:/<name>/<stage>) → 本地產物 → RuntimeError（該端點 503）
```

| 現象 | 斷在哪 | 檢查 |
| :--- | :--- | :--- |
| `model_version` = `Production` / `Staging` | 沒斷，正常 | — |
| `model_version` = `local` | registry 連不上 | 見下方 A |
| 該端點 503 | registry 與本地皆失敗 | 見下方 B |

### A. registry 連不上

```bash
# tracking URI 指到哪？
python3 -c "from src.utils.config import load_config; print(load_config()['mlflow']['tracking_uri'])"

# 環境變數有沒有覆蓋？
echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

# server 活著嗎（compose 部署時）
curl -s http://localhost:5000/health || echo "MLflow server 沒回應"
```

常見成因：

- `make up` 沒起，或 mlflow 容器掛了 → `docker compose -f docker/docker-compose.yml ps`
- `MLFLOW_TRACKING_URI` 指向錯的位址
- **模型根本沒有被註冊到該 stage**——很可能是品質門檻擋下了，見
  [runbook-quality-gate-failure](runbook-quality-gate-failure.md)

### B. 本地產物也不存在

```bash
ls -la models/tabular/model.xgb models/vision/model.onnx 2>&1
```

`models/` 由 DVC 管理且被 gitignore，全新 clone 後是空的。取回：

```bash
dvc pull              # 有設定 remote 時
# 或直接重訓
make train-tabular
make train-vision
```

---

## 處置

| 情況 | 動作 |
| :--- | :--- |
| MLflow server 沒起 | `make up`，等 mlflow 容器 healthy 後**重啟服務**（模型只在啟動時載入） |
| 模型沒過品質門檻所以沒註冊 | 這是保護機制正常運作。走 [品質門檻 runbook](runbook-quality-gate-failure.md)，不要繞過門檻硬註冊 |
| 本地產物遺失 | `dvc pull` 或重訓 |
| 只有 vision 掛掉 | 表格端點仍可服務。先讓 `predict_maintenance` 繼續運作，再修 vision |
| 前處理 / ONNX 版本不合 | 檢查 `onnx` 與 `onnxruntime` 版本是否與 `uv.lock` 一致（映像應由 lock 產生的 requirements 建置） |

> **重要**：模型是在服務**啟動時**載入的（`__init__`）。修好 registry 後**必須重啟服務**，
> 它不會自己重試。

---

## 完成後

- [ ] `/healthz` 回 `status: "ok"`，兩個 `*_model_loaded` 皆為 true
- [ ] 推論回應的 `model_version` **不是** `local`
- [ ] 若這次是因為 registry 不穩定，考慮加上「`model_version == local` 即告警」
- [ ] 記錄根因；若是模型未註冊，回頭確認品質門檻的判定是否合理

---

## 相關

- [`../design/api_spec.md` §5 模型版本語意](../design/api_spec.md#5-模型版本語意)
- [`../architecture/sad.md` §5.2 推論時的模型解析](../architecture/sad.md#52-推論時的模型解析)
- [ADR-001](../architecture/adr/ADR-001-bentoml-as-serving-framework.md)、[ADR-004](../architecture/adr/ADR-004-onnx-for-vision-serving.md)
