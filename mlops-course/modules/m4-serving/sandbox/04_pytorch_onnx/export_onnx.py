"""
這個檔示範什麼:
    階 8 的第二步——把 train_resnet.py 訓好的 PyTorch 模型匯出成 ONNX。

    為什麼要 ONNX(打包格式的取捨):
        - pickle / .pt(state_dict):綁死 Python + 同版 PyTorch,跨語言/跨框架難。
        - TorchScript:序列化計算圖,可脫離 Python 原始碼,但仍是 PyTorch 生態。
        - ONNX:開放標準中間表示,可被 ONNX Runtime / TensorRT / 各家硬體
          加速器執行,跨框架、跨語言、好做量化——服務化常見的「中立交換格式」。

    本檔也示範「動態量化(dynamic quantization)」的最小說明:
        把權重從 float32 壓成 int8,模型更小、CPU 推論更快,
        代價是少許精度損失。這是 GPU/邊緣服務常用的瘦身手法。

怎麼跑:
    python train_resnet.py        # 先有 resnet18_finetuned.pt
    python export_onnx.py
    # 產出: resnet18.onnx(以及說明用的 resnet18.quant.onnx)
"""

import torch
import torch.nn as nn
from torchvision import models

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cpu")
NUM_CLASSES = 3
IMAGE_SIZE = 224

WEIGHTS_PATH = "resnet18_finetuned.pt"
ONNX_PATH = "resnet18.onnx"
ONNX_QUANT_PATH = "resnet18.quant.onnx"


def load_trained_model() -> nn.Module:
    """重建與訓練時相同的網路結構,再載回訓好的權重。"""
    model = models.resnet18(weights=None)  # 結構即可,權重等下載自己訓的
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()  # 推論模式:關掉 dropout/batchnorm 的訓練行為
    return model


def export_to_onnx(model: nn.Module) -> None:
    """把模型匯出成 ONNX,並設定 batch 維度為動態(支援不同 batch size)。"""
    # ONNX 匯出需要一個「範例輸入」來追蹤計算圖
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        # 把 batch 維(第 0 維)設為動態,服務時才能一次吃多筆(dynamic batching)
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # 用傳統 TorchScript exporter:它輸出的 ONNX 與 onnxruntime 的動態量化
        # (quantize_dynamic 內部會跑 shape inference)相容;torch 2.x 預設的
        # dynamo exporter 輸出在某些算子上會讓量化的 shape inference 失敗。
        dynamo=False,
    )
    print(f"已匯出 ONNX: {ONNX_PATH}")


def quantize_onnx() -> None:
    """動態量化說明:把 float32 權重壓成 int8,模型更小、CPU 更快。

    需要 onnxruntime;若環境沒裝,僅印出說明不中斷流程。
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=ONNX_PATH,
            model_output=ONNX_QUANT_PATH,
            weight_type=QuantType.QInt8,  # 權重量化成 int8
        )
        print(f"已產生量化模型: {ONNX_QUANT_PATH}(更小、CPU 推論更快)")
    except ImportError:
        print("(略過量化:未安裝 onnxruntime。"
              "正式環境可 `pip install onnxruntime` 後再跑以體驗 int8 量化。)")


if __name__ == "__main__":
    model = load_trained_model()
    export_to_onnx(model)
    quantize_onnx()
