"""
workspace/services/app.py — M4 填空：FastAPI 服務化

對照：modules/m4-serving/sandbox/01_fastapi/app.py
重點：啟動時載入「已訓好」的模型，不要在請求內重訓。

怎麼跑（在 workspace/services/）：
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

CLASS_NAMES = ["setosa", "versicolor", "virginica"]
ml_models: dict = {}

# 預期：M2 訓練後你會存一個模型檔；路徑可改
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"


def load_model(path: Path):
    """TODO(M4-1): 從 path 載入模型（joblib / pickle / mlflow）。

    對照 sandbox 是「現訓」；workspace 必須改成「載入」。
    若檔案不存在，raise FileNotFoundError 並提示先完成 M2 匯出模型。
    """
    raise NotImplementedError("TODO(M4-1): load_model")


class IrisFeatures(BaseModel):
    """輸入 schema：四個 iris 數值欄（TODO(M4-2): 幫每欄加上 ge=0 等驗證）。"""

    sepal_length: float = Field(...)
    sepal_width: float = Field(...)
    petal_length: float = Field(...)
    petal_width: float = Field(...)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO(M4-3): 啟動時 load_model 放進 ml_models["clf"]
    raise NotImplementedError("TODO(M4-3): lifespan 載入模型")
    yield
    ml_models.clear()


app = FastAPI(title="workspace-iris-service", lifespan=lifespan)


@app.get("/health")
def health():
    # TODO(M4-4): 回傳 {"status": "ok"}；若模型未載入可回 503
    raise NotImplementedError("TODO(M4-4): /health")


@app.post("/predict")
def predict(features: IrisFeatures):
    # TODO(M4-5): 組 numpy 列 → model.predict → 回傳 class index + class name
    raise NotImplementedError("TODO(M4-5): /predict")
