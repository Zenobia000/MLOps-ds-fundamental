# 03 · BentoML:升級到 ML 原生服務框架(階 7)

> 本沙盒只學一個動詞:**用 BentoML service 服務同一個 iris 模型**,
> 並體會它跟 FastAPI 的差異。模型一樣現訓 iris,單檔自足。

---

## FastAPI vs BentoML:差在哪?

| 面向 | FastAPI(01) | BentoML(本沙盒) |
| :--- | :--- | :--- |
| 定位 | 通用 web 框架 | **ML 原生**服務框架 |
| 模型管理 | 自己 load pickle | 內建 **Model Store**(有版本/可追溯) |
| 輸入 schema | 自己寫 Pydantic | type hint 自動生成 |
| 打包部署 | 自己寫 Dockerfile | `bentoml build` + `containerize` 一鍵生 image |
| 推論最佳化 | 自己實作 | 內建 **adaptive batching**、runner 並行 |
| 何時選它 | 端點少、要塞進既有 web app | 純 ML 服務、要量產多模型/多版本 |

> 一句話:**FastAPI 給你最大彈性但什麼都自己來;BentoML 幫你把 ML 服務的瑣事標準化。**

## 1. 裝依賴

```bash
pip install -r requirements.txt
```

## 2. 先把模型存進 Model Store

```bash
python service.py
# 模型已存入 BentoML Model Store: iris_clf:xxxxxx
```

確認存進去了:

```bash
bentoml models list
```

## 3. 起服務

```bash
bentoml serve service:IrisClassifier --reload
# 預設跑在 http://localhost:3000
```

互動文件在 `http://localhost:3000`(BentoML 自動生成 Swagger UI)。

## 4. 測

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
# {"class_index":0,"class_name":"setosa","probabilities":{...}}
```

## 5.(選配)打包成 Bento → Docker image

```bash
bentoml build                                   # 依 bentofile.yaml 打包
bentoml list                                    # 看打包結果
bentoml containerize iris_classifier:latest     # 一鍵生成 Docker image
```

對比 02:你不用親手寫 Dockerfile,BentoML 從 bentofile 自動生成。

---

## 學到的最小心智模型

| 動詞 | 指令 / 程式 | 為什麼 |
| :--- | :--- | :--- |
| 存模型 | `bentoml.sklearn.save_model` | 進 Model Store,有版本 |
| 定義服務 | `@bentoml.service` + `@bentoml.api` | 描述模型與端點 |
| 起服務 | `bentoml serve` | 本地跑起來 |
| 打包 | `bentoml build` | 收斂成可部署 Bento |
| 生 image | `bentoml containerize` | 一鍵 Docker 化 |

> 進階(先不教):adaptive batching 參數、多模型組合、Yatai/雲端部署。
> 下一步(04):把 sklearn 換成 PyTorch 預訓練 CV 模型,並走 ONNX 流程。
