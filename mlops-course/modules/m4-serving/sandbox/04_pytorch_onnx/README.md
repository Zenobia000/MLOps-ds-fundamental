# 04 · PyTorch 預訓練 + ONNX:CV 模型服務化(階 8)

> 本沙盒把三步串成一條線:**遷移學習訓練 → 匯出 ONNX → 用 BentoML 服務**。
> 全程強制 CPU(無 GPU 後援),用極小假資料跑通流程,不追求準確率。

---

## 為什麼這一站從 sklearn 跳到 PyTorch?

前三站服務的是「結構化資料 + 小模型」。真實世界還有「影像 / 大型深度模型」,
它們的打包與服務有不同的取捨——這一站讓你體會 CV 模型的服務化關鍵字。

## 三步流程

```
train_resnet.py     →   export_onnx.py      →   serve_bento.py
(凍結 backbone,        (PyTorch → ONNX,        (onnxruntime 推論,
 只訓最後一層)           + int8 量化說明)         + 啟動暖機 warmup)
        │                      │                        │
 resnet18_finetuned.pt   resnet18.onnx           POST /predict
```

## 0. 裝依賴

```bash
pip install -r requirements.txt
```

> torch/torchvision 第一次會下載預訓練權重(約幾十 MB),需要網路。

## 1. 訓練(遷移學習骨架)

```bash
python train_resnet.py
# step 0: loss=...  →  已存權重: resnet18_finetuned.pt
```

學到的觀念:**凍結 backbone**(`requires_grad=False`)+ **換最後一層**
(`model.fc = nn.Linear(...)`),只訓練分類頭。

## 2. 匯出 ONNX(+ 量化說明)

```bash
python export_onnx.py
# 已匯出 ONNX: resnet18.onnx
# 已產生量化模型: resnet18.quant.onnx
```

學到的觀念:**動態 batch 維**(服務時可一次吃多筆 = dynamic batching 的前提)、
**int8 量化**(模型更小、CPU 推論更快,精度略降)。

## 3. 用 BentoML 服務 ONNX

```bash
bentoml serve serve_bento:ResNetONNX --reload
# warmup 完成,服務已就緒  →  http://localhost:3000
```

學到的觀念:**warmup 暖機**——啟動時先跑一次假輸入,把冷啟動延遲移到啟動階段。

---

## 打包格式三選一(服務化必懂的取捨)

| 格式 | 優點 | 缺點 | 何時用 |
| :--- | :--- | :--- | :--- |
| pickle / `.pt` state_dict | 簡單、原生 | 綁死 Python + 同版框架 | 同生態內快速存取 |
| TorchScript | 脫離 Python 原始碼 | 仍是 PyTorch 生態 | PyTorch 部署、行動端 |
| **ONNX** | 跨框架/語言、好量化、多硬體加速 | 算子覆蓋偶有限制 | 異質環境、邊緣、加速器 |

## GPU 服務關鍵字(本沙盒只示範心智模型,正式環境再深入)

- **dynamic batching**:把多個請求湊成一個 batch 一起算,提高 GPU 吞吐(已在 ONNX 設動態 batch 維)。
- **量化(quantization)**:float32 → int8/fp16,模型小、推論快、省記憶體。
- **warmup 暖機**:啟動先跑假輸入,避免第一個真實請求踩冷啟動延遲。

---

## 檢核

- [ ] 你能說出為什麼要凍結 backbone、只訓最後一層?
- [ ] ONNX 相比 pickle 的主要好處是什麼?
- [ ] warmup 解決的是哪一種延遲問題?
