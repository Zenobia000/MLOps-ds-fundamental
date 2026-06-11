# 02 · Docker:把 FastAPI 服務裝進容器(階 5)

> 本沙盒只學一個動詞:**把 01 的服務 `build` 成 image,再 `run` 起來**。
> 程式碼一行都不用改——容器化的價值就是「不改程式、換環境也能跑」。

---

## 為什麼 build context 是上一層 (`..`)

01 的程式在 `../01_fastapi/`,而 Docker 只能複製「build context 範圍內」的檔案。
所以我們把 context 指到上一層,Dockerfile 內用 `01_fastapi/...` 的相對路徑複製。

## 1. Build(打包成 image)

```bash
# 在本資料夾(02_docker/)執行
docker build -f Dockerfile -t iris-fastapi:0.1 ..
```

- `-f Dockerfile` 指定要用的 Dockerfile。
- `-t iris-fastapi:0.1` 給 image 取名 + 版本標籤。
- 結尾的 `..` 是 build context(上一層),Dockerfile 才看得到 `01_fastapi/`。

看 image 建好了沒:

```bash
docker images | grep iris-fastapi
```

## 2. Run(起容器),`-p` 做埠對應

```bash
docker run --rm -p 8000:8000 iris-fastapi:0.1
```

- `-p 8000:8000` = `主機埠:容器埠`。少了這行,服務只活在容器裡,外面連不到。
- `--rm` 容器停掉就自動刪除,沙盒練習不留垃圾。
- 要背景跑加 `-d`:`docker run -d --rm -p 8000:8000 iris-fastapi:0.1`

## 3. 測(跟 01 完全一樣)

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## 4. 收尾

```bash
# 找到容器 ID 後停掉(若沒加 --rm 才需要)
docker ps
docker stop <CONTAINER_ID>
```

---

## 學到的最小心智模型

| 動詞 | 指令 | 為什麼 |
| :--- | :--- | :--- |
| 打包 | `docker build -t name:tag .` | 把程式+依賴+環境凍結成 image |
| 執行 | `docker run --rm name:tag` | 從 image 起一個容器 |
| 埠對應 | `-p 主機:容器` | 讓外面連得到容器內服務 |
| 探活 | `HEALTHCHECK` | 編排器判斷容器是否健康 |

> 進階(本沙盒先不教,之後再回來):multi-stage build 縮小 image、
> docker compose 一次起多個服務、非 root 使用者強化安全。
