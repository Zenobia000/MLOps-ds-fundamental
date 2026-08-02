# CV 影像處理 Pipeline：Resize → ToTensor → Normalize

> 本頁完整說明 [`train_resnet.py`](./sandbox/04_pytorch_onnx/train_resnet.py) 註解裡那三步 transform，以及為什麼 ImageNet 預訓練模型**一定要用同一組 normalize**。  
> 概念圖：[`assets/m4-cv-image-pipeline.png`](./assets/m4-cv-image-pipeline.png)  
> Backbone 架構圖：[`assets/m4-cv-backbone.png`](./assets/m4-cv-backbone.png)

---

## 1. 一句話定位

**影像進模型之前，必須變成「模型認得的數字張量」**：尺寸對、通道順序對、數值尺度對。  
這三步合起來叫 **preprocessing pipeline**；訓練與線上推論必須用**同一條**，否則會 training-serving skew。

本課骨架：

```python
transforms.Compose([
    transforms.Resize(IMAGE_SIZE),          # 1. 尺寸對齊
    transforms.ToTensor(),                  # 2. 型別 / 通道 / 尺度
    transforms.Normalize(                   # 3. ImageNet 統計標準化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

---

## 2. 整條 Pipeline（資料長什麼樣）

```text
原始影像 (PIL / ndarray)
  H×W×C, uint8, 像素約 0~255, 通道順序常是 RGB
        │
        ▼  Resize(224)
  224×224×3, 仍多是 uint8
        │
        ▼  ToTensor()
  Tensor C×H×W = 3×224×224, float32, 數值約 0~1
        │
        ▼  Normalize(ImageNet mean/std)
  Tensor 3×224×224, float32, 約略零均值（每通道各自縮放）
        │
        ▼  DataLoader / unsqueeze(0)
  Batch N×3×224×224  →  餵進 ResNet
```

| 步驟後 | 形狀 | dtype | 數值大致範圍 |
| :--- | :--- | :--- | :--- |
| 原始 | `H×W×3` | uint8 | 0 ~ 255 |
| Resize | `224×224×3` | uint8 | 0 ~ 255 |
| ToTensor | `3×224×224` | float32 | 0 ~ 1 |
| Normalize | `3×224×224` | float32 | 約 -2 ~ +2（視影像而定） |
| + batch | `N×3×224×224` | float32 | 同上 |

> PyTorch / torchvision 慣例是 **NCHW**（batch, channel, height, width），不是 OpenCV 常見的 HWC。

---

## 3. 每一步在做什麼

### 3.1 `Resize(IMAGE_SIZE)` — 尺寸對齊

```python
transforms.Resize(IMAGE_SIZE)  # IMAGE_SIZE = 224
```

| 問題 | 說明 |
| :--- | :--- |
| 為什麼要做 | ResNet 預訓練時固定吃 **224×224**；尺寸不一無法組成 batch，權重感受野也不對 |
| 輸入 | 任意高寬的 PIL Image（或相容格式） |
| 輸出 | 短邊/指定邊對齊後的影像（預設行為見 torchvision 版本文檔） |
| 實務補充 | 正式訓練常再加 `CenterCrop(224)` 或 `RandomResizedCrop(224)`；本課骨架用 Resize 示意即可 |

常見變化（進階，本沙盒未強制）：

| Transform | 用途 |
| :--- | :--- |
| `Resize(256)` + `CenterCrop(224)` | 驗證/推論常見穩妥組合 |
| `RandomResizedCrop(224)` | 訓練時資料增強 |
| `RandomHorizontalFlip` | 訓練時左右翻轉增強 |

### 3.2 `ToTensor()` — 型別、通道、尺度一次到位

```python
transforms.ToTensor()
```

它同時做三件事：

1. **HWC → CHW**：`(H, W, 3)` → `(3, H, W)`
2. **uint8 → float32**
3. **除以 255**：把 `0~255` 縮成 **`0~1`**

| 若省略會怎樣 |
| :--- |
| 模型收到 0~255 或錯誤通道順序，特徵分布全錯，準確率崩盤 |

### 3.3 `Normalize(mean, std)` — ImageNet 統計標準化

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # R, G, B
    std=[0.229, 0.224, 0.225],
)
```

公式（**每個通道各自算**）：

```text
output[c] = (input[c] - mean[c]) / std[c]
```

| 問題 | 說明 |
| :--- | :--- |
| 這組數字哪來的 | ImageNet 訓練集上算好的 **RGB 通道 mean / std** |
| 為什麼必須用 | torchvision 的 ResNet 權重是在「已 Normalize 過」的影像上訓的；你若不做或用錯統計值，等於輸入分布漂移 |
| 輸入前提 | 必須先 `ToTensor()`，數值已在 **0~1**；不要對 0~255 直接套這組 mean/std |

錯誤示範：

```python
# 錯：還沒 ToTensor 就 Normalize，或 mean/std 自己亂設
# 錯：訓練用 ImageNet normalize，推論忘了做
# 錯：訓練有 Normalize，服務端只 Resize 就送進 ONNX
```

---

## 4. 訓練 vs 推論：同一條邏輯

| 階段 | Pipeline | 差別 |
| :--- | :--- | :--- |
| **訓練** | Resize / RandomCrop + Flip + ToTensor + Normalize | 可加隨機增強，提高泛化 |
| **驗證 / 推論** | Resize(+CenterCrop) + ToTensor + Normalize | **關掉隨機**，結果可重現 |
| **必須相同** | ToTensor 尺度、Normalize 的 mean/std、最終空間尺寸、通道順序 | 否則 offline 準、上線崩 |

本課 `serve_bento.py` 註解已標明：真實情境要在 API 內做 resize / normalize；沙盒假設呼叫端已送對形狀，是為了聚焦 ONNX 推論。  
**工程上上線時，前處理要寫進服務，且與訓練一致。**

---

## 5. 和本課假資料的關係

`train_resnet.py` 主流程用的是：

```python
torch.randn(N, 3, 224, 224)  # 已是 NCHW float，跳過真實影像 pipeline
```

檔末註解才是「換成真實 CIFAR-10」時要接上的完整 pipeline：

```python
tfm = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

所以：

- **主程式**：示範 backbone 凍結 / 換頭 / 訓練迴圈  
- **檔末註解 + 本頁**：示範真實影像進模型前必須做的事  

---

## 6. 最小可跑片段（真實影像時）

```python
from torchvision import transforms
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

img = Image.open("cat.jpg").convert("RGB")  # 保證三通道 RGB
x = preprocess(img)                         # (3, 224, 224)
batch = x.unsqueeze(0)                      # (1, 3, 224, 224)
# model(batch)
```

檢查清單：

- [ ] 讀圖後 `.convert("RGB")`（避免灰階或 RGBA 通道數不對）
- [ ] Resize 到模型預期尺寸（本課 224）
- [ ] 有 `ToTensor()`（0~1 + CHW）
- [ ] Normalize 用 ImageNet mean/std（預訓練 ResNet）
- [ ] 推論前 `unsqueeze(0)` 或 DataLoader 組成 batch
- [ ] 服務端前處理與訓練端一致

---

## 7. 常見錯誤

| 錯誤 | 後果 | 修正 |
| :--- | :--- | :--- |
| 忘了 Normalize | 輸入分布偏離預訓練，分數異常 | 訓練/推論都加同一組 mean/std |
| 對 0~255 直接 Normalize | 數值尺度錯兩個數量級 | 先 `ToTensor()` |
| HWC 當 CHW 送進模型 | shape 錯或結果亂 | `ToTensor()` 或 `permute(2,0,1)` |
| 訓練有增強、上線用另一套 normalize | training-serving skew | 共用同一 preprocess 模組 |
| 灰階圖沒轉 RGB | 通道數 1 ≠ 3 | `.convert("RGB")` |

---

## 8. 檢核

1. `ToTensor()` 做了哪三件事？  
2. ImageNet 的 mean/std 為什麼不能隨便換？  
3. 為什麼訓練與 `/predict` 必須共用同一條 preprocess？  
4. 張量從原始圖到進 ResNet，形狀如何從 `H×W×3` 變成 `1×3×224×224`？

動手：把 `train_resnet.py` 檔末註解接上 CIFAR-10 子集時，確認 DataLoader 吐出的 batch shape 是 `(B, 3, 224, 224)`。
