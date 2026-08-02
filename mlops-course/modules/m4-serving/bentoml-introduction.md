# BentoML 入門：ML 原生模型服務框架

> 本頁回答：**BentoML 是什麼、解決什麼問題、核心概念有哪些、什麼時候該用。**  
> 動手實作請接 [`sandbox/03_bentoml/`](./sandbox/03_bentoml/README.md)；工程落地細節見 [`application-guide.md`](./application-guide.md#bentoml-怎麼應用)。

---

## 1. 一句話定位

**BentoML 是專門為機器學習模型設計的服務化框架**——你把「模型 + 推論邏輯」描述清楚，它幫你處理模型版本、API 文件、打包與容器化等 ML 服務常見瑣事。

對比 FastAPI：

| | FastAPI | BentoML |
| :--- | :--- | :--- |
| 本質 | 通用 Web 框架 | ML 服務框架 |
| 你要自己寫 | 模型載入、版本管理、Dockerfile、推論最佳化 | 主要描述 service 與 API |
| 適合 | 端點少、要接進既有後端、非純 ML 邏輯多 | 純 ML 推論、多模型多版本、要標準化部署 |

> **FastAPI 給你最大彈性但什麼都自己來；BentoML 幫你把 ML 服務的瑣事標準化。**

---

## 2. 它解決什麼工程問題

模型訓練完只是 `.pkl` 或 `.pt` 檔，產品與其他系統無法直接呼叫。服務化要把模型變成 HTTP API，但用 FastAPI 從零做起，你通常還得自己處理：

- 模型存在哪、怎麼載入、怎麼管版本
- request/response schema 與 OpenAPI 文件
- 依賴與模型 artifact 一起打包成可部署單位
- 推論吞吐（例如 batching、多 worker）

BentoML 把這些收斂成一套 **ML 原生工作流**：存模型 → 定義 service → 本地 serve → 打包 Bento → 容器化部署。

---

## 3. 核心概念（四個名詞）

```text
訓練產物 ──save──▶ Model Store ──引用──▶ Service ──build──▶ Bento ──containerize──▶ Docker image
```

### Model Store（模型倉庫）

BentoML 內建的模型儲存與版本管理。模型以 **tag** 命名（例如 `iris_clf:abc123`），可追溯、可切版本。

```python
import bentoml
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X, y)
saved = bentoml.sklearn.save_model("iris_clf", model)
# saved.tag → iris_clf:xxxxxx
```

常用 CLI：

```bash
bentoml models list          # 列出已存模型
bentoml models get iris_clf:latest   # 查看某模型 metadata
```

### Service（服務定義）

用 Python class 描述「這個服務載入哪些模型、提供哪些 API」。BentoML 1.2+ 使用 `@bentoml.service` 與 `@bentoml.api`：

```python
@bentoml.service(name="iris_classifier")
class IrisClassifier:
    bento_model = bentoml.models.get("iris_clf:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.bento_model)

    @bentoml.api
    def predict(self, features: list[float]) -> dict:
        ...
```

重點：

- **一個 class = 一個 service**
- 方法上的 **type hint** 會自動生成 JSON schema 與 Swagger UI（不必像 FastAPI 手寫 Pydantic，但仍可搭配 Pydantic）
- 模型在 **`__init__` 載入一次**，不要在每個 request 裡重載

### Bento（可部署包）

`bentoml build` 依 `bentofile.yaml` 把 **service 程式 + 模型 + 依賴** 收斂成一個可版本化的部署單位（Bento）。類似「專為 ML 服務設計的 release artifact」。

### bentofile.yaml（打包描述檔）

告訴 BentoML 要打包什麼：

```yaml
service: "service:IrisClassifier"
include:
  - "service.py"
python:
  requirements_txt: "requirements.txt"
```

對比 Docker：Dockerfile 你要逐步寫 COPY/RUN/CMD；bentofile 只描述需求，BentoML 幫你生成可部署結構。

---

## 4. 最小工作流（五個動詞）

| 步驟 | 指令 / API | 做什麼 |
| :--- | :--- | :--- |
| 1. 存模型 | `bentoml.sklearn.save_model(...)` | 把訓練產物放進 Model Store |
| 2. 定義服務 | `@bentoml.service` + `@bentoml.api` | 描述模型與推論端點 |
| 3. 本地跑 | `bentoml serve service:IrisClassifier` | 預設 `http://localhost:3000`，自帶 Swagger |
| 4. 打包 | `bentoml build` | 產出 Bento |
| 5. 容器化 | `bentoml containerize iris_classifier:latest` | 一鍵生成 Docker image |

本課 sandbox 完整流程：[`sandbox/03_bentoml/README.md`](./sandbox/03_bentoml/README.md)。

---

## 5. 與本課其他元件怎麼接

| 上游 | 接法 |
| :--- | :--- |
| **MLflow** | 從 registry 或 artifact 路徑載入模型，再 `save_model` 進 BentoML Store（或 serve 時直接 load） |
| **FastAPI / Docker** | 01、02 沙盒用 FastAPI + 手寫 Dockerfile；03 起改用 BentoML，體會「誰幫你省掉哪些步驟」 |
| **ONNX / PyTorch** | 04 沙盒：ResNet 匯出 ONNX 後，用 BentoML service + ONNX Runtime 推論 |
| **Capstone** | `smart-factory-mlops/services/` 把表格 + 影像雙模型包成單一 BentoML service |

---

## 6. 什麼時候選 BentoML

**適合：**

- 服務主體就是模型推論，端點以 `/predict` 類為主
- 需要管理 **多模型、多版本**（Model Store + tag）
- 希望 **build → containerize** 標準化，減少手寫 Dockerfile
- 需要 adaptive batching、資源描述（CPU/GPU）等推論向能力

**仍用 FastAPI 較好：**

- API 裡有大量非 ML 邏輯（權限、訂單、業務流程）
- 要把模型嵌進 **既有 Web 後端**，改動面要小
- 只有一兩個簡單端點，團隊已熟悉 FastAPI，BentoML 的抽象還沒開始值回票價

---

## 7. 常見誤解

| 誤解 | 事實 |
| :--- | :--- |
| BentoML 取代 FastAPI | BentoML 底層仍用 ASGI；它是 **ML 服務層**，不是通用 Web 框架 |
| 一定要先用 BentoML 存模型 | 也可在 service 的 `__init__` 從 pickle / MLflow / ONNX 載入；Model Store 是 **可選但推薦** 的版本化方式 |
| `.pt` 直接給 BentoML | PyTorch 權重需先載入模型結構；CV 場景本課走 **ONNX** 再 serve（見 04 沙盒） |
| build 完就等於上線 | 仍要測 `/health`、壓力測試、監控；BentoML 解決的是 **打包與服務定義**，不是 M6 的 drift 監控 |

---

## 8. 本課延伸閱讀

| 資源 | 內容 |
| :--- | :--- |
| [`sandbox/03_bentoml/`](./sandbox/03_bentoml/README.md) | iris 模型：Model Store → serve → build → containerize |
| [`sandbox/04_pytorch_onnx/serve_bento.py`](./sandbox/04_pytorch_onnx/serve_bento.py) | ONNX + BentoML 影像推論 |
| [`application-guide.md`](./application-guide.md) | M4 工程驗收、FastAPI 對照、序列化格式選型 |
| [`component-introduction.md`](../component-introduction.md) | 全課程元件速讀（含 BentoML 在生命週期中的位置） |
| Capstone `services/service.py` | 雙模型統一 API 的生產級範例 |

---

## 9. 檢核：讀完這頁你應該能回答

1. Model Store、Service、Bento、bentofile 各自扮演什麼角色？
2. 同一個 iris 模型，用 FastAPI 與 BentoML 各要多寫哪些東西？
3. `bentoml serve` 與 `bentoml build` 的差別是什麼？
4. 什麼情境你會從 FastAPI 升級到 BentoML？

動手驗證：完成 [`03_bentoml`](./sandbox/03_bentoml/README.md) 沙盒後，用 curl 打一次 `/predict`，並在 Swagger UI 試第二筆輸入。
