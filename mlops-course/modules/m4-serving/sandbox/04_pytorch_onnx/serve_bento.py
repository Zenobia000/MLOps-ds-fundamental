"""
這個檔示範什麼:
    階 8 的最後一步——把 export_onnx.py 產出的 ONNX 模型,
    用 BentoML(階 7 學過的)包成一個推論服務。

    串起整條線:
        train_resnet.py(訓)→ export_onnx.py(匯出 ONNX)→ 本檔(服務)

    重點:這裡用 onnxruntime 載入並執行 ONNX,示範「服務暖機(warmup)」——
    在服務啟動時先跑一次假輸入,把 runtime 的延遲初始化、記憶體配置先做掉,
    避免「第一個真實請求」特別慢(GPU/CPU 服務常見的冷啟動問題)。

怎麼跑:
    python train_resnet.py
    python export_onnx.py
    pip install onnxruntime          # 若尚未安裝
    bentoml serve serve_bento:ResNetONNX --reload

    curl -X POST http://localhost:3000/predict \
      -H "Content-Type: application/json" \
      -d "$(python -c 'import json,random; random.seed(0); print(json.dumps({\"image\": [[ [random.random() for _ in range(224)] for _ in range(224)] for _ in range(3)]}))')"
"""

import bentoml
import numpy as np

ONNX_PATH = "resnet18.onnx"
NUM_CLASSES = 3
IMAGE_SIZE = 224


@bentoml.service(
    name="resnet_onnx",
    resources={"cpu": "1"},
    traffic={"timeout": 20},
)
class ResNetONNX:
    """用 ONNX Runtime 服務 ResNet18 的 BentoML service。"""

    def __init__(self) -> None:
        import onnxruntime as ort

        # 載入 ONNX 模型,建立推論 session(CPU provider)
        self.session = ort.InferenceSession(
            ONNX_PATH, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # === 暖機(warmup):啟動時先跑一次假輸入 ===
        # 第一次推論常因延遲初始化而特別慢,暖機把這成本移到「服務啟動時」,
        # 讓真正的第一個使用者請求就已是熱路徑。
        dummy = np.random.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})
        print("warmup 完成,服務已就緒")

    @bentoml.api
    def predict(self, image: list) -> dict:
        """接收一張 [3][224][224] 的 RGB 影像(巢狀 list),回傳預測類別。

        真實情境會在這裡做前處理(resize / normalize);
        此處假設呼叫端已送來正確形狀,聚焦在「ONNX 推論」本身。
        """
        # list → numpy,補上 batch 維度,轉成 ONNX 需要的 float32
        x = np.asarray(image, dtype=np.float32)
        if x.ndim == 3:
            x = x[np.newaxis, ...]  # (3,224,224) → (1,3,224,224)

        logits = self.session.run(None, {self.input_name: x})[0]
        class_index = int(np.argmax(logits, axis=1)[0])

        # softmax 把 logits 轉成機率
        exp = np.exp(logits[0] - np.max(logits[0]))
        proba = (exp / exp.sum()).tolist()

        return {
            "class_index": class_index,
            "probabilities": {f"class_{i}": p for i, p in enumerate(proba)},
        }
