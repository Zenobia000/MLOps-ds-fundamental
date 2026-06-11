"""
這個檔示範什麼:
    階 8 PyTorch 服務的第一步——遷移學習(transfer learning)的最小骨架。
    用 torchvision 的「預訓練 ResNet18」當 backbone,凍結它的權重,
    只換掉並訓練最後一層分類頭(classification head)。

    為什麼這樣做:
        從零訓 CNN 需要大量資料與 GPU 時間;遷移學習借用 ImageNet 上
        學到的通用視覺特徵,只在「最後一層」針對新任務微調,
        極少資料 + CPU 也能跑——非常適合教學示範。

    本檔刻意用「極小的隨機假資料」代替真實影像資料集,
    目的只在說明骨架流程(凍結 → 換頭 → 訓練 → 存權重),
    不追求準確率。要換成真資料(如 CIFAR-10 子集)的接法見檔末註解。

    *** 無 GPU 後援:全程強制使用 CPU,確保任何機器都跑得起來。 ***

怎麼跑:
    pip install -r requirements.txt
    python train_resnet.py
    # 產出: resnet18_finetuned.pt(只含 state_dict)
"""

import torch
import torch.nn as nn
from torchvision import models

# 設定隨機種子,確保假資料與初始化可重現
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# 強制使用 CPU(教學環境無 GPU 後援)
DEVICE = torch.device("cpu")

# 新任務的類別數(示範用,假設要分 3 類)
NUM_CLASSES = 3

# 假資料規模:極小,只為跑通流程
NUM_FAKE_SAMPLES = 16
IMAGE_SIZE = 224  # ResNet 預期輸入 224x224 的 RGB 圖

WEIGHTS_PATH = "resnet18_finetuned.pt"


def build_model() -> nn.Module:
    """載入預訓練 ResNet18,凍結 backbone,只換最後一層分類頭。"""
    # 1. 載入在 ImageNet 上預訓練好的權重
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. 凍結所有 backbone 參數:requires_grad=False → 反向傳播不更新它們
    for param in model.parameters():
        param.requires_grad = False

    # 3. 換掉最後一層全連接層(原本 1000 類 → 改成我們的 NUM_CLASSES)。
    #    新建的層 requires_grad 預設為 True,所以「只有這一層會被訓練」。
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model.to(DEVICE)


def make_fake_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """產生一批極小的隨機假影像與標籤,純粹為了跑通訓練迴圈。"""
    images = torch.randn(NUM_FAKE_SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (NUM_FAKE_SAMPLES,))
    return images.to(DEVICE), labels.to(DEVICE)


def train() -> None:
    """最小訓練骨架:只更新分類頭,跑幾步示範流程。"""
    model = build_model()
    model.train()

    criterion = nn.CrossEntropyLoss()
    # 只把「需要梯度的參數」(也就是新分類頭)交給 optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    print(f"可訓練參數張量數量: {len(trainable)} (應只有分類頭的 weight/bias)")

    images, labels = make_fake_batch()

    # 跑幾個 step 示範「loss 會動」即可,不追求收斂
    for step in range(3):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss={loss.item():.4f}")

    # 只存 state_dict(權重),不存整個物件——較小、較可攜
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"已存權重: {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()


# --- 要換成真實資料(CIFAR-10 子集)時,把 make_fake_batch 換成: ---
#
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset
#
# tfm = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),
#     transforms.ToTensor(),
#     # ImageNet 預訓練模型要用同一組 normalize 統計值
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
# full = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
# subset = Subset(full, range(256))            # 只取 256 筆,CPU 也能跑
# loader = DataLoader(subset, batch_size=32, shuffle=True)
# 然後把訓練迴圈改成 for images, labels in loader: ...
