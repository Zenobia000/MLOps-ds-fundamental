# 01 · FastAPI:把模型包成一個 POST /predict(階 6)

> 本沙盒只學一個動詞:**用 FastAPI 開一個 HTTP 端點服務既有模型**。
> 模型在檔內現訓 iris,讓你不必準備任何外部檔案就能跑起來。

---

## 1. 裝依賴

```bash
pip install -r requirements.txt
```

## 2. 起服務

```bash
# 在本資料夾(01_fastapi/)執行
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- `app:app` = `app.py` 檔裡的 `app` 物件。
- `--reload` 改檔自動重載(只在開發時用,正式服務不要開)。
- 看到 `Uvicorn running on http://0.0.0.0:8000` 就成功了。

## 3. 用瀏覽器看自動產生的文件

FastAPI 會自動生成互動式 API 文件,打開:

```
http://localhost:8000/docs
```

可以直接在頁面上按 "Try it out" 送請求,不必寫 curl。

## 4. 用 curl 測

健康檢查:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

預測(送一筆 setosa 典型特徵):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      }'
# {"class_index":0,"class_name":"setosa","probabilities":{...}}
```

故意送錯(少一個欄位),看 FastAPI 如何回 422 驗證錯誤:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1}'
# 422 Unprocessable Entity + 清楚指出缺哪些欄位
```

---

## 學到的最小心智模型

| 動詞 | 在 app.py 哪裡 | 為什麼 |
| :--- | :--- | :--- |
| 啟動載入模型一次 | `lifespan` | 訓練一次、服務多次 |
| 定義輸入 schema | `IrisFeatures(BaseModel)` | 邊界驗證、快速失敗 |
| 開一個端點 | `@app.post("/predict")` | 對外的呼叫介面 |
| 健康檢查 | `@app.get("/health")` | 給容器/編排器探活 |

> 下一步(02_docker):同一支服務不必改任何程式,直接用 Dockerfile 打包成 image。
