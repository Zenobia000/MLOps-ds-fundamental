"""ResNet transfer learning 瑕疵檢測模型（PyTorch / torchvision）。

場景：產線視覺瑕疵檢測（good / defect 二分類）。載入 torchvision 預訓練
ResNet18/50，凍結 backbone、替換分類頭，只微調最後一層（小資料防過擬合），
並可匯出 ONNX 供 src.serving 以 onnxruntime 推論。

無 GPU 後援：自動偵測 CUDA，無 GPU 時退回 CPU；ONNX 匯出全程在 CPU 進行。

TODO（生產化）：
    - 完整 train loop（augmentation / scheduler / mixed precision）放在 src.training。
    - 大資料微調時解凍部分 backbone，並用差分學習率。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torchvision import models

from src.utils.logging import get_logger

logger = get_logger(__name__)

ONNX_FILENAME = "model.onnx"
WEIGHTS_FILENAME = "resnet.pt"

# 支援的 backbone → (建構式, 預訓練權重列舉)。
_BACKBONES: dict[str, tuple[Any, Any]] = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


class ResNetDefectClassifier(nn.Module):
    """凍結 backbone、替換分類頭的 ResNet 瑕疵分類器。"""

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        p = dict(params or {})
        self.backbone_name = str(p.get("backbone", "resnet18"))
        self.num_classes = int(p.get("num_classes", 2))
        self.image_size = int(p.get("image_size", 224))
        pretrained = bool(p.get("pretrained", True))
        freeze = bool(p.get("freeze_backbone", True))

        if self.backbone_name not in _BACKBONES:
            raise ValueError(
                f"不支援的 backbone：{self.backbone_name}（可選 {list(_BACKBONES)}）"
            )
        ctor, weights = _BACKBONES[self.backbone_name]
        # pretrained=False 時傳 weights=None，避免無網路環境下載失敗。
        self.backbone = ctor(weights=weights if pretrained else None)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 換頭：把原本 1000 類的 fc 換成 num_classes 輸出（此層必訓練）。
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, self.num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logger.info(
            "建立 %s 瑕疵分類器（num_classes=%d, frozen=%s, device=%s）",
            self.backbone_name,
            self.num_classes,
            freeze,
            self.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.backbone(x)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """回傳需要梯度的參數（凍結 backbone 時即僅分類頭），供 optimizer 用。"""
        return [p for p in self.parameters() if p.requires_grad]

    # ── 序列化 ───────────────────────────────────────────────────────────
    def save(self, dir_path: str | Path) -> Path:
        """儲存 PyTorch 權重到 ``<dir_path>/resnet.pt``。"""
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = out_dir / WEIGHTS_FILENAME
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "backbone": self.backbone_name,
                    "num_classes": self.num_classes,
                    "image_size": self.image_size,
                    "pretrained": False,  # 載回時不需再下載
                    "freeze_backbone": False,
                },
            },
            weights,
        )
        logger.info("已儲存 ResNet 模型：%s", weights)
        return weights

    @classmethod
    def load(cls, dir_path: str | Path) -> "ResNetDefectClassifier":
        """從 ``<dir_path>/resnet.pt`` 載入模型（CPU 安全）。"""
        weights = Path(dir_path) / WEIGHTS_FILENAME
        if not weights.exists():
            raise FileNotFoundError(f"找不到模型權重：{weights}")
        ckpt = torch.load(weights, map_location="cpu")
        instance = cls(params=ckpt["config"])
        instance.load_state_dict(ckpt["state_dict"])
        instance.eval()
        logger.info("已載入 ResNet 模型：%s", weights)
        return instance

    def export_onnx(self, dir_path: str | Path, *, opset: int = 17) -> Path:
        """匯出 ONNX 到 ``<dir_path>/model.onnx``，供 onnxruntime 服務推論。

        輸出 logits 形狀 ``(batch, num_classes)``，與 src.serving.predict
        的 ``predict_defect``（good/defect 二欄）契約一致。
        """
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / ONNX_FILENAME

        self.eval()
        dummy = torch.randn(1, 3, self.image_size, self.image_size)
        torch.onnx.export(
            self.cpu(),
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        self.to(self.device)  # 還原裝置
        logger.info("已匯出 ONNX 模型：%s（opset=%d）", onnx_path, opset)
        return onnx_path
