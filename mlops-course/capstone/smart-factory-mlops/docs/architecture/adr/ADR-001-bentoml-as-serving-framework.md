# ADR-001：用 BentoML 當服務化框架

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 已接受
> **Owner:** Capstone 維護者
> **實例:** 每決策一份

## 脈絡

系統要同時提供兩種推論：**表格模型**（XGBoost，CPU、毫秒級、批次）與**影像模型**（ONNX，較重、單張）。
需求：

- 一個服務同時掛兩個異質模型，不要為此起兩套服務
- 能打包成映像交付，且推論映像不該背訓練依賴
- 教學上要能讓學員在 15 分鐘內從「模型檔」走到「可 curl 的 API」

候選：FastAPI 手寫、BentoML、NVIDIA Triton、KServe、Ray Serve。

## 決策

**採用 BentoML**（`services/service.py`，`@bentoml.service` + `@bentoml.api`）。

## 理由

| 候選 | 為什麼不選 |
| :--- | :--- |
| **FastAPI 手寫** | 能做，但模型載入、批次、打包、版本全要自己寫。課程 m4 已用它建立直覺（`modules/m4-serving/sandbox/01_fastapi/`），capstone 要示範的是「ML 原生框架幫你省掉什麼」 |
| **Triton** | 高吞吐、支援 dynamic batching，但要寫 model repository 結構與 config.pbtxt，對 CPU 上的玩具規模是過度工程；且它的心智模型離「Python 函式」較遠 |
| **KServe** | K8s 原生，但強制引入 Kubernetes——與「本機 compose 就能跑完」的教學前提衝突 |
| **Ray Serve** | 適合需要複雜 DAG / 自動擴縮的場景，本專案兩個端點用不到 |

選 BentoML 的關鍵：**它的打包單位（Bento）天生包含依賴宣告**（`bentofile.yaml` 的 `requirements_txt`），
所以「模型 + 程式 + 依賴」是一起版本化的一個交付物，這正是 MLOps 要教的東西。

## 後果

**正面**

- 一個 service class 掛兩個模型，共用 config 載入與健康檢查
- `bentoml build` 產出的 Bento 自帶依賴，`Dockerfile.serve` 只需裝 `requirements-serve.txt`
- schema 直接吃 Pydantic（`src/serving/schemas.py`），驗證免自己寫

**負面 / 代價**

- BentoML 1.2 → 1.3 → 1.4 的 API 變動大（`@bentoml.service` 是新式寫法），教材需跟著升版。已用 `uv.lock` 鎖住 1.4.39
- 沒有 Triton 等級的 dynamic batching；若未來吞吐成為瓶頸需重新評估
- 多一層框架抽象，除錯時要懂 BentoML 的 runner 生命週期

**分工上的約束（重要）**

推論邏輯放 `src/serving/`（純函式、可單元測試），`services/service.py` 只做框架的殼。
這樣換框架時要改的只有殼——這個邊界是刻意為了「不被 BentoML 綁死」而設的。

## 相關

- [ADR-004](ADR-004-onnx-for-vision-serving.md)：影像模型為什麼用 ONNX 而非 TorchScript
- [`../../design/api_spec.md`](../../design/api_spec.md)：三個端點的對外契約
