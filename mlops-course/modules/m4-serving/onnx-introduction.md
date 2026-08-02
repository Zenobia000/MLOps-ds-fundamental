# ONNX 入門：中立模型格式、ONNX Runtime、TensorRT

> 本頁回答：**ONNX 是什麼、跟 pickle/TorchScript 差在哪、和 ONNX Runtime / TensorRT 是什麼關係。**  
> 動手匯出見 [`sandbox/04_pytorch_onnx/export_onnx.py`](./sandbox/04_pytorch_onnx/export_onnx.py)；服務見 [`serve_bento.py`](./sandbox/04_pytorch_onnx/serve_bento.py)。  
> 概念圖：[`assets/m4-onnx-runtime-tensorrt.png`](./assets/m4-onnx-runtime-tensorrt.png)

---

## 1. 一句話定位

**ONNX（Open Neural Network Exchange）是模型的「中立交換格式」**——把計算圖與權重存成 `.onnx`，訓練框架（PyTorch 等）與推論引擎（ONNX Runtime、TensorRT 等）之間不必綁死同一套 runtime。

類比：

| 角色 | 類比 |
| :--- | :--- |
| `.onnx` 檔 | PDF：大家都能讀的中立文件 |
| PyTorch / TensorFlow | Word / Google Docs：各自編輯環境 |
| ONNX Runtime / TensorRT | 不同品牌的 PDF 閱讀器／加速印表機 |

> **ONNX 本身通常不負責「最快推論」；它負責「換引擎時不用重訓、少改碼」。**  
> 真正跑推論的是 **Runtime / 加速器**。

---

## 2. ONNX 是什麼（技術上）

一個 `.onnx` 檔大致包含：

| 內容 | 意義 |
| :--- | :--- |
| **計算圖（graph）** | 一串運算子（Conv、MatMul、ReLU…）怎麼接 |
| **權重（initializers）** | 訓練好的參數 |
| **輸入 / 輸出定義** | 名稱、形狀、型別（本課：`input` → `logits`） |
| **opset version** | 運算子規格版本（本課 `opset_version=17`） |

訓練端（本課）：

```text
PyTorch model  ──torch.onnx.export──▶  resnet18.onnx
```

服務端（本課）：

```text
resnet18.onnx  ──onnxruntime.InferenceSession──▶  預測結果
```

本課 `export_onnx.py` 還把 **batch 維設成動態**（`dynamic_axes`），讓服務端一次可吃多筆，對應 dynamic batching 的前提。

---

## 3. 三種打包格式怎麼選

對應 `export_onnx.py` 開頭的取捨：

| 格式 | 本質 | 優點 | 限制 | 何時用 |
| :--- | :--- | :--- | :--- | :--- |
| **pickle / `.pt` state_dict** | Python / PyTorch 原生序列化 | 簡單、還原訓練結構方便 | 綁 Python + 套件版本 | 同生態內存權重、繼續微調 |
| **TorchScript** | PyTorch 計算圖序列化 | 可少依賴原始 `.py` | 仍偏 PyTorch 生態 | 仍在 PyTorch runtime 部署 |
| **ONNX** | 開放中間表示 | 跨框架、跨語言、好接加速器與量化 | 部分算子匯出/對齊需驗證 | 異質環境、CPU/GPU 加速、邊緣 |

一句判斷：

- 還要在 Python + PyTorch 裡改模型 → `.pt` / TorchScript 夠用  
- 要給 C++/其他語言、ONNX Runtime、TensorRT、邊緣裝置 → **匯出 ONNX**

---

## 4. ONNX vs ONNX Runtime vs TensorRT

這三個最容易混在一起——層次不同：

```text
訓練框架 (PyTorch)
        │ export
        ▼
   ┌─────────┐
   │  ONNX   │  ← 格式 / 檔案（.onnx）
   └────┬────┘
        │ 載入並執行
        ├──────────────────────┐
        ▼                      ▼
 ONNX Runtime              TensorRT
 （通用推論引擎）         （NVIDIA GPU 優化引擎）
        │                      │
        ▼                      ▼
   CPU / GPU / …          NVIDIA GPU（高度最佳化）
```

| 名稱 | 是什麼 | 吃什麼 | 本課有沒有用到 |
| :--- | :--- | :--- | :--- |
| **ONNX** | **格式**（中立模型檔） | — | 有：`resnet18.onnx` |
| **ONNX Runtime（ORT）** | **推論引擎**：讀 `.onnx` 並執行 | `.onnx` | 有：`serve_bento.py`、`quantize_dynamic` |
| **TensorRT** | **NVIDIA 的高效能推論引擎** | 可從 ONNX 建 engine，或自家流程 | 本課不跑；正式 NVIDIA GPU 服務常見 |

關係整理：

1. **ONNX ≠ Runtime**  
   ONNX 是檔案格式；Runtime 才是「跑起來」的程式庫。

2. **ONNX Runtime 是「通用執行器」**  
   同一份 `.onnx` 可在 CPU（`CPUExecutionProvider`）、CUDA、其他 EP 上跑。本課強制 CPU provider，確保無 GPU 也能教。

3. **TensorRT 是「NVIDIA 上的加速器路線」**  
   常見路徑：`.onnx` → TensorRT 解析/最佳化 → 產生 engine → GPU 上極低延遲推論。  
   它不是 ONNX 的替代格式名稱，而是**另一個執行後端**（常以 ONNX 當入口）。

4. **誰負責量化**  
   - 本課：`onnxruntime.quantization.quantize_dynamic` → `resnet18.quant.onnx`（INT8，偏 CPU）  
   - TensorRT：可在建 engine 時做 FP16/INT8 等 GPU 向最佳化（正式環境議題）

---

## 5. 資料流（本課完整一條線）

```text
train_resnet.py          export_onnx.py              serve_bento.py
PyTorch + 微調頭    →    resnet18.onnx (+ quant)  →  ORT InferenceSession
.pt state_dict           中立圖檔                     POST /predict
```

| 步驟 | 產物 | 工具 |
| :--- | :--- | :--- |
| 訓練 | `resnet18_finetuned.pt` | PyTorch |
| 匯出 | `resnet18.onnx` | `torch.onnx.export` |
| （可選）量化 | `resnet18.quant.onnx` | ONNX Runtime quantization |
| 服務 | HTTP 預測 | BentoML + ONNX Runtime |

匯出後工程上應做的驗證（本課心智模型，正式必做）：

```text
同一筆 input → PyTorch 輸出  vs  ONNX Runtime 輸出  → 數值應對齊
```

---

## 6. 什麼時候選誰

| 情境 | 建議 |
| :--- | :--- |
| 教學 / CPU / 跨機器可重現 | **ONNX + ONNX Runtime**（本課路線） |
| 要 INT8 縮小、加速 CPU | ORT dynamic / static quantization |
| NVIDIA GPU、要極致延遲與吞吐 | **ONNX → TensorRT**（或 Triton + TensorRT） |
| 仍深度綁 PyTorch 動態圖除錯 | 先別急著上 TensorRT；先驗證 ONNX 數值 |

口訣：

> **用 ONNX 解耦訓練框架；用 Runtime/TensorRT 解耦「在哪顆晶片上跑多快」。**

---

## 7. 常見誤解

| 誤解 | 事實 |
| :--- | :--- |
| 「裝了 ONNX 就能推論」 | ONNX 是格式；還要 **ONNX Runtime** 或 **TensorRT** 等引擎 |
| 「ONNX Runtime = TensorRT」 | ORT 通用；TensorRT 主打 NVIDIA GPU 深度最佳化 |
| 「匯出成功 = 上線正確」 | 必須做 **數值對齊驗證**；算子覆蓋也可能失敗 |
| 「有 ONNX 就一定比 PyTorch 快」 | 不一定；速度取決於後端（ORT EP / TensorRT / 硬體） |

---

## 8. 檢核

1. ONNX、ONNX Runtime、TensorRT 各自是「格式」還是「引擎」？  
2. 為什麼服務化常把 PyTorch 轉成 ONNX，而不是直接 pickle 整個 model？  
3. 本課 `serve_bento.py` 用的是哪一個執行引擎？  
4. TensorRT 通常如何接到 ONNX 流程？

動手：跑完 `export_onnx.py` 後，確認產出 `resnet18.onnx`，再用 `serve_bento.py` 以 ORT 載入推論。
