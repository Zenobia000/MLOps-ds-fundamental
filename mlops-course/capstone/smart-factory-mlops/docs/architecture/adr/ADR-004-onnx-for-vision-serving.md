# ADR-004：影像模型以 ONNX 形式服務，不送 PyTorch 權重

> **版本:** v1.0 | **更新:** 2026-08-03 | **狀態:** 已接受
> **Owner:** Capstone 維護者
> **實例:** 每決策一份

## 脈絡

瑕疵檢測用 torchvision 的預訓練 ResNet 微調。訓練產出是 `.pt`（`state_dict`），
但服務端要的是「能穩定、輕量、快速推論的東西」。

問題：**推論容器要不要裝 PyTorch？** 裝了容器會胖到 6 GB 以上、冷啟動變慢；
不裝就得換一種模型格式。

候選：直接送 `.pt` + 服務端裝 torch、TorchScript、ONNX + ONNX Runtime。

## 決策

**訓練完匯出 ONNX**（`src/models/vision/model.py` 的 `export_onnx()`，opset 17），
服務端用 **ONNX Runtime** 推論（`src/serving/model_loader.py` 的 `load_vision_session`）。
**推論映像不裝 torch。**

## 理由

| 候選 | 為什麼不選 |
| :--- | :--- |
| 送 `.pt` + 服務裝 torch | 推論容器要背整個 PyTorch（含 CUDA stub），映像從 ~1.9 GB 膨脹到 ~6.9 GB；且 `state_dict` 需要模型類別定義才能載入，服務端得 import 訓練期的程式碼，訓練與服務耦合 |
| TorchScript | 解決了「需要類別定義」的問題，但仍需 torch runtime，映像大小沒省到；跨框架能力也不如 ONNX |

ONNX 的三個實際好處：

1. **服務端零 torch 依賴**——實測 `Dockerfile.serve` 1.9 GB vs `Dockerfile.train` 6.9 GB
2. **訓練/服務解耦**——ONNX 是自帶計算圖的檔案，服務端不需要知道模型是怎麼定義的
3. **可量化**——ONNX Runtime 的動態量化（INT8）能再縮體積、加速 CPU 推論，
   課程 m4 沙盒有可執行的對照示範

## 後果

**正面**

- 推論映像小、冷啟動快、攻擊面小
- 換訓練框架不影響服務端（只要還能匯出 ONNX）
- 前處理明確化：因為服務端沒有 torchvision 的 transform，前處理必須自己寫成
  `src/serving/predict.py` 的 `preprocess_image`——**這反而是好事**，前處理變成可測試的明碼，
  而不是藏在框架裡的隱含行為

**負面 / 代價**

- **前處理必須手動對齊**。訓練用 torchvision transform、服務用 numpy，兩邊算得不一樣就會靜默降準。
  這是本決策最大的風險，靠 `tests/unit/` 的前處理測試守住
- 匯出這一步會壞。實例：torch 2.13 起 `torch.onnx.export` 預設走 dynamo，
  **需要額外的 `onnxscript` 套件**，少了它會 `ModuleNotFoundError`。
  已顯式宣告在 `pyproject.toml`（見 [`../../README.md`](../../README.md) 依賴說明）
- 動態軸與 opset 版本要對齊，換 opset 可能讓舊的量化流程失效

## 相關

- [ADR-001](ADR-001-bentoml-as-serving-framework.md)：服務框架
- [`../sad.md` §7 部署視圖](../sad.md#7-部署視圖)：兩個映像的大小對照
