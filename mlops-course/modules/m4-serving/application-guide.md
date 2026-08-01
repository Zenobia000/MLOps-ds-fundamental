# M4 應用指導手冊：模型服務化與部署格式

## 這個模組解決什麼工程問題

M4 解決的是「模型怎麼被別的系統使用」。訓練完成的模型如果只存在 notebook 或本機檔案裡，對產品沒有價值。服務化的目標是把模型變成穩定、可部署、可測試、可維運的 API。

本模組的工程心法是：模型服務啟動時載入模型，請求進來時只做驗證、前處理、推論、回傳；絕對不要在 request 裡重新訓練模型。

## 哪邊應用

M4 適用在模型已訓練完成，需要被產品、後端、排程任務或其他系統呼叫的階段。只要模型要離開資料科學家的本機，就需要服務化與部署格式。

常見應用：

- 後端服務呼叫模型：用 HTTP `/predict` 取得預測結果。
- 批次或即時推論平台：將模型服務封裝成標準 container。
- 影像模型部署：將 PyTorch 模型轉成 ONNX，降低推論環境耦合。
- 多模型服務管理：用 BentoML 管模型版本與服務包裝。

## 怎麼用

使用順序是先 API 化，再容器化，最後依需求升級模型服務框架與部署格式：

1. 從已訓練模型 artifact 開始，不在 API request 內訓練。
2. 用 Pydantic 定義 request/response schema。
3. 用 FastAPI 建立 `/health` 與 `/predict`。
4. 用 Uvicorn 本機啟動並測試。
5. 用 Dockerfile 封裝依賴、模型檔與啟動命令。
6. 若要標準化 ML serving，改用 BentoML。
7. 若要跨框架或高效推論，評估 TorchScript 或 ONNX。

## 本模組元件

| 元件 | 在 MLOps 裡的職責 | 本課使用方式 |
| :--- | :--- | :--- |
| FastAPI | 建立 HTTP 推論 API | `/health`、`/predict` |
| Pydantic | 定義與驗證 request schema | 輸入欄位型別檢查 |
| Uvicorn | 啟動 ASGI web server | 本機執行 FastAPI |
| Docker | 封裝服務環境 | build image、run container |
| BentoML | ML 原生服務化框架 | service、Model Store、Bento |
| pickle / joblib | 傳統 ML 模型序列化 | 保存 sklearn 模型 |
| PyTorch / ResNet | 深度學習影像模型 | 預訓練模型服務化 |
| TorchScript / ONNX | 模型部署格式 | 跨環境推論 |

## FastAPI 怎麼應用

FastAPI 適合建立簡潔、可客製化的模型推論 API。

最小 API 結構：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = load_model_once()

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    features = [[
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width,
    ]]
    pred = model.predict(features)
    return {"prediction": int(pred[0])}
```

工程應用場景：

- 只有少量模型端點。
- 需要把模型接進既有 web service。
- 團隊熟 Python API 開發。
- 需要高度控制 request/response、auth、logging、business logic。

設計原則：

- 模型在 app 啟動時載入，不要在 request 中載入。
- 每個端點都要有明確 schema。
- `/health` 不應執行昂貴推論，只確認服務活著。
- 錯誤輸入應由 schema 先擋下。

## Pydantic 怎麼應用

Pydantic 負責把 API 輸入從「任意 JSON」變成「可驗證資料結構」。

工程應用：

- 檢查欄位是否存在。
- 檢查型別是否正確。
- 加上範圍限制，例如分數不能小於 0。
- 自動產生 OpenAPI 文件。

設計建議：

- 輸入 schema 命名為 `PredictRequest`。
- 輸出 schema 可命名為 `PredictResponse`。
- schema 應與模型訓練時的特徵順序對齊。
- 不要在 endpoint 裡手動從 dict 到處取值。

## Uvicorn 怎麼應用

Uvicorn 是啟動 FastAPI 的 server。

本機啟動：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

工程理解：

- 第一個 `app` 是 Python 檔名 `app.py`。
- 第二個 `app` 是檔案內的 FastAPI 實例變數。
- Docker container 內也通常用 Uvicorn 作為啟動命令。

## Docker 怎麼應用

Docker 把模型服務與依賴打包成 image，讓環境可攜帶。

最小 Dockerfile 結構：

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.pkl .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

常用指令：

```bash
docker build -t iris-fastapi:0.1 .
docker run --rm -p 8000:8000 iris-fastapi:0.1
```

工程應用場景：

- 將模型服務交給別人部署。
- CI/CD 要 build image。
- 本機、測試、正式環境需要一致。
- 要部署到 Kubernetes、Cloud Run、ECS 等容器平台。

設計原則：

- image 裡要包含啟動服務所需的模型檔與依賴。
- container 啟動後應能直接提供 `/health`。
- 使用固定版本依賴，避免部署時行為漂移。
- 不要把訓練資料或不必要檔案塞進 serving image。

## BentoML 怎麼應用

BentoML 是 ML 原生服務化框架。它幫你處理模型保存、服務定義、打包與容器化。

FastAPI 與 BentoML 取捨：

| 面向 | FastAPI | BentoML |
| :--- | :--- | :--- |
| 定位 | 通用 web framework | ML service framework |
| 模型管理 | 自己設計 | 內建 Model Store |
| 打包 | 自己寫 Dockerfile | Bento + containerize |
| 彈性 | 最高 | ML 情境更省事 |
| 適合 | 客製 API、整合既有後端 | 多模型、多版本、標準化 ML serving |

工程應用場景：

- 需要管理多個模型版本。
- 希望服務規格、模型 artifact、依賴一起打包。
- 需要 batch inference 或更 ML 原生的服務能力。

最小流程：

```bash
python service.py
bentoml serve service:IrisClassifier
bentoml build
bentoml containerize iris_classifier:latest
```

## 模型序列化格式怎麼選

| 格式 | 適用模型 | 優點 | 限制 |
| :--- | :--- | :--- | :--- |
| pickle / joblib | scikit-learn | 簡單快速 | 綁 Python 與套件版本 |
| TorchScript | PyTorch | 較少依賴原始 Python code | 仍偏 PyTorch 生態 |
| ONNX | PyTorch、sklearn 等 | 跨框架、跨硬體、利於推論最佳化 | 算子覆蓋與轉換需驗證 |

工程判斷：

- 內部 Python sklearn 服務：joblib 通常夠用。
- PyTorch 模型仍在 PyTorch runtime：TorchScript 可考慮。
- 要跨語言、跨硬體、低延遲推論：優先評估 ONNX。

## PyTorch、ResNet、ONNX 怎麼應用

PyTorch 適合深度學習；ResNet 是常見影像模型架構；ONNX 是部署交換格式。

典型影像服務流程：

```text
pretrained ResNet -> fine-tune last layer -> export ONNX -> validate output -> serve
```

工程注意事項：

- 匯出 ONNX 後要用同一筆 input 比對 PyTorch output 與 ONNX output。
- 前處理要和訓練時一致，例如 resize、normalize、channel order。
- 模型服務啟動時做 warmup，避免第一個請求延遲過高。
- GPU 高流量服務要考慮 dynamic batching。
- 邊緣或低延遲場景可以考慮 quantization。

## 最小落地流程

1. 從已訓練模型 artifact 開始，不在服務內訓練模型。
2. 定義 request schema 與 response schema。
3. 建立 `/health` 與 `/predict`。
4. 啟動 API，本機用 curl 或 Swagger UI 測試。
5. 寫 Dockerfile 封裝服務。
6. 用 Docker 啟動並測試 `/health`、`/predict`。
7. 若服務需求偏 ML 原生，改用 BentoML 管模型與打包。
8. 若是深度學習模型，評估 TorchScript 或 ONNX 部署格式。

## 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 每次 request 都重新訓練模型 | 延遲高、結果不穩 | 啟動時載入模型 |
| schema 與訓練特徵順序不一致 | 推論錯誤但不一定報錯 | 明確定義欄位順序 |
| Docker image 依賴未鎖版本 | 部署後行為漂移 | 固定 requirements |
| 沒有 `/health` | 部署平台無法判斷服務狀態 | 加健康檢查端點 |
| ONNX 匯出後未驗證 | 線上推論可能和原模型不同 | 比對 PyTorch 與 ONNX output |
| 把訓練流程塞進 serving image | image 大、風險高 | 訓練與服務分離 |

## 工程驗收標準

- API 啟動後 `/health` 回傳正常。
- `/predict` 能處理合法輸入並回傳可解釋結果。
- 錯誤輸入會被 schema 擋下。
- 模型只在啟動時載入一次。
- Docker container 可在乾淨環境啟動。
- 若使用 ONNX，輸出已和原模型比對。

## 你應該帶走的能力

完成 M4 後，工程師應該能把訓練好的模型轉成可被其他系統呼叫的服務，並能根據服務複雜度選擇 FastAPI、Docker、BentoML、ONNX 等工具組合。
