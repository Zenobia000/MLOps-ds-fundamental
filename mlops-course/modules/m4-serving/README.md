# M4 · 模型服務化(Serving)

> 對應技能階梯 **階 5–8**:Docker / FastAPI / BentoML / PyTorch+ONNX。

---

## 1. 本模組學什麼

把「訓好的模型」變成「別人能呼叫的服務」。本模組依「先熟悉、再升級」的節奏走:
**FastAPI(階 6)→ Docker 包起來(階 5)→ 升級 BentoML(階 7)→ PyTorch 預訓練 + ONNX(階 8)**。
學完你能說清楚:模型怎麼打包(pickle/ONNX/TorchScript)、服務框架怎麼選、GPU 服務有哪些關鍵字。

### 先建立兩個背景概念

**打包格式(Serialization)——模型存成什麼樣的檔?**

| 格式 | 一句話 | 取捨 |
| :--- | :--- | :--- |
| pickle / joblib | sklearn 等的原生序列化 | 簡單,但綁死 Python + 同版套件 |
| TorchScript | PyTorch 計算圖序列化 | 脫離 Python 原始碼,仍在 PyTorch 生態 |
| ONNX | 開放標準中間表示 | 跨框架/語言、好量化、多硬體;算子偶有覆蓋限制 |

**服務框架選型——用什麼把模型開成 API?**

| 框架 | 定位 | 何時選 |
| :--- | :--- | :--- |
| FastAPI | 通用 web 框架,彈性最大、什麼都自己來 | 端點少、要塞進既有 web app |
| BentoML | ML 原生,內建模型管理/打包/批次最佳化 | 純 ML 服務、要量產多模型多版本 |

**GPU 服務關鍵字(階 8 會碰到的心智模型)**

- **dynamic batching**:把多個請求湊成一批一起算,拉高 GPU 吞吐。
- **量化(quantization)**:float32 → int8/fp16,模型小、推論快、省記憶體,精度略降。
- **warmup 暖機**:啟動先跑假輸入,避免第一個真實請求踩冷啟動延遲。

---

## 2. 沙盒步驟(Layer 1:照編號逐個跑,一次一工具)

每個沙盒都「單檔自足、現訓玩具模型」,彼此不 import。各資料夾內有自己的 README 與 `requirements.txt`。

| 編號 | 資料夾 | 學的動詞 | 怎麼起 |
| :--- | :--- | :--- | :--- |
| 01 | `sandbox/01_fastapi/` | 一個 `POST /predict` 包既有模型 | `uvicorn app:app --port 8000` |
| 02 | `sandbox/02_docker/` | `build` / `run` / `-p` 把 01 容器化 | `docker build -f Dockerfile -t iris-fastapi:0.1 ..` |
| 03 | `sandbox/03_bentoml/` | service + runner + `bentofile` + `serve` | `bentoml serve service:IrisClassifier` |
| 04 | `sandbox/04_pytorch_onnx/` | 預訓練 ResNet → ONNX → Bento 服務 | 見該資料夾 README 三步 |

> 順序有意義:先用最熟悉的 FastAPI 建立直覺,再用 Docker 學「不改程式換環境」,
> 再用 BentoML 看「ML 原生框架幫你省掉哪些瑣事」,最後用 PyTorch+ONNX 面對 CV 模型的服務取捨。

---

## 3. 整合任務(Layer 2:接到 `workspace/`)

回到 `../../workspace/`,把你前面模組(M2 訓練 / M3 特徵)產出的模型包成可呼叫 API。

**TODO 提示:**

- [ ] 在 `workspace/services/` 新增一個 FastAPI 或 BentoML 服務,**載入 M2 訓好的模型**(從 pickle 或 MLflow Registry),不要在服務內重訓。
- [ ] 定義輸入 schema:欄位對齊 M3 Feast 的 feature view,在邊界驗證。
- [ ] 加 `/health` 端點與容器 `HEALTHCHECK`,為 M5 自動化部署鋪路。
- [ ] 寫一個 `Dockerfile`(參考 `sandbox/02_docker/`),把服務容器化並 `-p` 對外。
- [ ] (進階)若 workspace 有影像子場景,走 04 的 ONNX 流程並在服務啟動時 warmup。

> 正解對照:`../../checkpoints/after-m4/`。

---

## 4. 卡住怎麼辦

- 想要乾淨起點:從上一模組快照重置 workspace
  ```bash
  cp -r ../../checkpoints/after-m3/* ../../workspace/
  ```
- 做完想對答案:比對本模組快照 `../../checkpoints/after-m4/`。
- 沙盒本身跑不起來:沙盒範例「就是可跑的正解」,照各資料夾 README 的指令逐行跑;
  Docker / BentoML 沒裝先看各 `requirements.txt` 與 README 的安裝段。

---

## 5. 檢核題

1. pickle、TorchScript、ONNX 三種打包格式各自的最大優點與限制是什麼?
2. 同一個模型,什麼情況你會選 FastAPI、什麼情況選 BentoML?各舉一個理由。
3. Docker `run` 時 `-p 8000:8000` 的兩個 8000 分別代表什麼?少了這行會怎樣?
4. 遷移學習為什麼要「凍結 backbone、只訓最後一層」?這樣省下了什麼?
5. dynamic batching、量化、warmup 各自解決 GPU 服務的什麼問題?
