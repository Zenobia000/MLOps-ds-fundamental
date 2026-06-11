"""
這個檔示範什麼:
    階 6 FastAPI 的「最小可用單元」——把一個既訓好的 sklearn 模型
    包成一個 HTTP 服務,對外只開一個 POST /predict 端點。

    重點觀念:
        1. 模型在「啟動時」載入一次(這裡為了單檔自足,直接在檔內現訓 iris),
           不要每次請求都重訓——服務化的核心就是「訓練一次、服務多次」。
        2. 用 Pydantic 定義「輸入 schema」,在系統邊界驗證外部資料。
        3. 路由函式只做:解析輸入 → 呼叫模型 → 回傳 JSON。

怎麼跑:
    # 先裝依賴
    pip install "fastapi[standard]" uvicorn scikit-learn

    # 起服務(reload 方便開發時自動重載)
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

    # 開另一個終端機測試(見同層 README.md 的 curl 範例)
    curl http://localhost:8000/health
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 設定隨機種子,確保每次起服務訓出來的模型一致(可重現性)
RANDOM_SEED = 42

# iris 的三種類別名稱,用來把模型輸出的數字索引轉回人類看得懂的標籤
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# 用一個 module 層級的容器存放「啟動時載入的模型」,
# 讓路由函式可以共用同一個模型物件,不必每次請求重建。
ml_models: dict = {}


def train_iris_model() -> LogisticRegression:
    """現訓一個 iris 分類器當作「既訓好的模型」。

    真實情境中,這裡會改成「從 pickle / MLflow Registry / ONNX 載入」,
    而不是在服務內訓練。此處為了讓 sandbox 單檔可獨立執行才現訓。
    """
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
    model.fit(X, y)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命週期:服務啟動時載入模型一次,關閉時清理。

    這是 FastAPI 推薦的「啟動載入模型」寫法,避免在每個請求重複載入。
    """
    ml_models["iris"] = train_iris_model()
    yield  # ← yield 之前是「啟動」,之後是「關閉」
    ml_models.clear()


app = FastAPI(title="Iris Classifier", version="1.0.0", lifespan=lifespan)


class IrisFeatures(BaseModel):
    """輸入 schema:iris 的四個特徵(公分)。

    用 Pydantic 在系統邊界驗證外部輸入——少傳、型別錯都會被擋下,
    並回傳清晰的 422 錯誤,符合「快速失敗」原則。
    """

    sepal_length: float = Field(..., ge=0, description="花萼長度 (cm)")
    sepal_width: float = Field(..., ge=0, description="花萼寬度 (cm)")
    petal_length: float = Field(..., ge=0, description="花瓣長度 (cm)")
    petal_width: float = Field(..., ge=0, description="花瓣寬度 (cm)")


class PredictResponse(BaseModel):
    """輸出 schema:預測類別索引、人類可讀名稱、各類別機率。"""

    class_index: int
    class_name: str
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    """健康檢查端點。容器編排(Docker / K8s)會用它判斷服務是否就緒。"""
    return {"status": "ok", "model_loaded": "iris" in ml_models}


@app.post("/predict", response_model=PredictResponse)
def predict(features: IrisFeatures) -> PredictResponse:
    """接收一筆 iris 特徵,回傳預測類別與機率。"""
    model: LogisticRegression = ml_models["iris"]

    # Pydantic 物件 → 模型需要的二維陣列 (1 筆 x 4 特徵)
    x = np.array(
        [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]
    )

    class_index = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]

    return PredictResponse(
        class_index=class_index,
        class_name=CLASS_NAMES[class_index],
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, proba)},
    )
