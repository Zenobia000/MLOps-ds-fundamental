# Docker 入門：常見指令、生命週期、Dockerfile、Image、Compose

> 本頁回答：**Dockerfile / Image / Container / Compose 怎麼分、容器生命週期長什麼樣、日常指令怎麼下。**  
> 動手實作請接 [`sandbox/02_docker/`](./sandbox/02_docker/README.md)；工程落地見 [`application-guide.md`](./application-guide.md#docker-怎麼應用)。

概念圖：[`assets/m4-docker-lifecycle.png`](./assets/m4-docker-lifecycle.png)

---

## 1. 一句話定位

**Docker 把「程式 + 依賴 + 執行環境」凍成可移植單位**，讓同一份服務在本機、CI、正式環境行為一致。

四個名詞先分清楚：

| 名詞 | 是什麼 | 類比 |
| :--- | :--- | :--- |
| **Dockerfile** | 建 image 的配方（文字檔） | 食譜 |
| **Image** | 依配方建好的唯讀範本 | 燒錄好的安裝光碟 |
| **Container** | 從 image 跑起來的程序實例 | 正在跑的程式 |
| **Compose** | 用 YAML 一次定義、啟動多個服務 | 整桌菜的出餐順序 |

```text
Dockerfile ──docker build──▶ Image ──docker run──▶ Container
                                      ▲
                              docker compose up
                              （可一次起多個 container）
```

---

## 2. 容器生命週期

```text
                    docker build
Dockerfile ──────────────────────▶ Image（存在本機）
                                      │
                                      │ docker run
                                      ▼
                              Created → Running
                                      │
                         docker stop  │  docker pause / unpause
                                      ▼
                                   Stopped
                                      │
                         docker start │（可再回到 Running）
                                      │
                         docker rm    ▼
                                   Deleted
```

| 狀態 | 意義 | 常用指令 |
| :--- | :--- | :--- |
| Image 已建好 | 還沒跑 | `docker images` |
| Created | 容器已建立、尚未啟動 | `docker create`（少用；多半直接 `run`） |
| Running | 服務在跑 | `docker run` / `docker start` |
| Stopped | 停了但還在本機 | `docker stop` → `docker start` 可重啟 |
| Deleted | 容器刪掉；image 還在 | `docker rm`；加 `--rm` 則 stop 後自動刪 |

本課最小路徑：

```bash
docker build -f Dockerfile -t iris-fastapi:0.1 ..   # 建 image
docker run --rm -p 8000:8000 iris-fastapi:0.1      # 起 container；停掉就刪
docker ps                                          # 看正在跑的
docker stop <CONTAINER_ID>                         # 停掉
```

> **Image 刪不掉容器；Container 刪不掉 Image。**  
> 清空間：`docker image prune` / `docker container prune`（先確認沒要保留的東西）。

---

## 3. Dockerfile：配方怎麼寫

對照本課 [`sandbox/02_docker/Dockerfile`](./sandbox/02_docker/Dockerfile)：

| 指令 | 做什麼 | 本課例子 |
| :--- | :--- | :--- |
| `FROM` | 選基底 image | `python:3.11-slim` |
| `ENV` | 環境變數 | `PYTHONUNBUFFERED=1` |
| `WORKDIR` | 容器內工作目錄 | `/app` |
| `COPY` | 把本機檔案拷進 image | `requirements.txt`、`app.py` |
| `RUN` | **建置時**執行（結果寫進 image layer） | `pip install -r requirements.txt` |
| `EXPOSE` | 宣告埠（文件用途；對外仍要 `-p`） | `8000` |
| `HEALTHCHECK` | 探活 | 打 `/health` |
| `CMD` | **容器啟動時**預設命令 | `uvicorn app:app --host 0.0.0.0 --port 8000` |

寫 Dockerfile 的兩個實務習慣：

1. **先 COPY requirements → RUN pip，再 COPY 程式碼**  
   依賴沒變時可重用 layer cache，改程式不必重裝套件。
2. **`CMD` 綁 `0.0.0.0`**  
   綁 `127.0.0.1` 時只有容器內部連得到，外面 `-p` 也進不來。

`RUN` vs `CMD`：

| | 何時執行 | 結果 |
| :--- | :--- | :--- |
| `RUN` | `docker build` 時 | 寫進 image |
| `CMD` / `ENTRYPOINT` | `docker run` 時 | 每次啟動容器都跑 |

---

## 4. Image：建、看、標、刪

| 目的 | 指令 |
| :--- | :--- |
| 建 image | `docker build -t name:tag .` |
| 指定 Dockerfile + context | `docker build -f Dockerfile -t iris-fastapi:0.1 ..` |
| 列 image | `docker images` 或 `docker image ls` |
| 加標籤 | `docker tag iris-fastapi:0.1 iris-fastapi:latest` |
| 推遠端（需 registry） | `docker push myregistry/iris-fastapi:0.1` |
| 拉 image | `docker pull python:3.11-slim` |
| 刪 image | `docker rmi iris-fastapi:0.1` |
| 清未使用 image | `docker image prune` |

命名慣例：`名稱:標籤`，例如 `iris-fastapi:0.1`。沒寫 tag 預設是 `latest`（正式環境建議用明確版本號）。

**Build context**：`docker build` 最後那個路徑（`.` 或 `..`）決定 Docker 能看到哪些檔案。  
本課 02 用 `..`，是因為要複製隔壁的 `01_fastapi/`。

---

## 5. Container：日常操作指令

| 目的 | 指令 |
| :--- | :--- |
| 前景啟動 | `docker run --rm -p 8000:8000 iris-fastapi:0.1` |
| 背景啟動 | `docker run -d --name iris-api -p 8000:8000 iris-fastapi:0.1` |
| 列執行中 | `docker ps` |
| 列全部（含停掉的） | `docker ps -a` |
| 看 log | `docker logs -f iris-api` |
| 進容器 shell | `docker exec -it iris-api bash`（或 `sh`） |
| 停 | `docker stop iris-api` |
| 再啟動 | `docker start iris-api` |
| 重啟 | `docker restart iris-api` |
| 刪容器 | `docker rm iris-api` |
| 強制刪執行中 | `docker rm -f iris-api` |

常用 `run` 旗標：

| 旗標 | 意思 |
| :--- | :--- |
| `-d` | 背景跑（detached） |
| `--rm` | 容器結束後自動刪除（沙盒練習好用） |
| `-p 主機:容器` | 埠對應；少了這行外面連不到 |
| `--name` | 幫容器取名，之後不用記 ID |
| `-e KEY=value` | 傳環境變數 |
| `-v 主機路徑:容器路徑` | 掛載 volume（開發熱重載、持久化資料） |
| `--network` | 指定網路 |

測本課服務（容器跑起來後）：

```bash
curl http://localhost:8000/health
```

---

## 6. Docker Compose：一次編排多服務

當你不只一個容器（例如：API + Redis + DB），手打多個 `docker run` 容易亂。  
**Compose 用一支 `compose.yaml`（或 `docker-compose.yml`）描述服務關係，一條指令起停。**

> 本課 `02_docker` 沙盒先練單容器；Compose 是下一步心智模型，這裡給最小可用範例。

最小 `compose.yaml` 概念：

```yaml
services:
  api:
    build:
      context: ..
      dockerfile: 02_docker/Dockerfile
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 3s
      retries: 3
```

| 目的 | 指令（Compose V2：`docker compose`） |
| :--- | :--- |
| 建並啟動 | `docker compose up --build` |
| 背景啟動 | `docker compose up -d --build` |
| 看狀態 | `docker compose ps` |
| 看 log | `docker compose logs -f` |
| 停並移除容器 | `docker compose down` |
| 連 volume 一起清 | `docker compose down -v` |

`docker run` vs Compose：

| | `docker run` | Compose |
| :--- | :--- | :--- |
| 服務數 | 通常一個 | 多個服務、網路、依賴 |
| 設定存放 | 散在指令列 | `compose.yaml` 可版控 |
| 適合 | 沙盒、單服務驗證 | 本機多服務、接近正式拓撲 |

---

## 7. 常見指令速查（一頁表）

```bash
# --- Image ---
docker build -t name:tag .              # 建
docker images                           # 列
docker rmi name:tag                     # 刪
docker image prune                      # 清懸空 image

# --- Container 生命週期 ---
docker run --rm -p 8000:8000 name:tag   # 建+跑（停後刪）
docker run -d --name app -p 8000:8000 name:tag
docker ps / docker ps -a                # 列
docker logs -f app                      # log
docker exec -it app sh                  # 進容器
docker stop app / docker start app      # 停 / 啟
docker rm app                           # 刪容器

# --- Compose ---
docker compose up -d --build
docker compose ps
docker compose logs -f
docker compose down
```

除錯口訣：

1. 容器有沒有在跑？→ `docker ps`
2. 埠有沒有對？→ `-p 主機:容器`，容器內服務綁 `0.0.0.0`
3. 程式有沒有掛？→ `docker logs`
4. Image 是不是舊的？→ 改 Dockerfile 後記得重新 `build`

---

## 8. 與本課怎麼接

| 位置 | 角色 |
| :--- | :--- |
| [`sandbox/02_docker/`](./sandbox/02_docker/README.md) | 單服務：`build` → `run` → `-p` → `HEALTHCHECK` |
| [`sandbox/01_fastapi/`](./sandbox/01_fastapi/README.md) | 被裝進容器的應用 |
| [`bentoml-introduction.md`](./bentoml-introduction.md) | 之後可用 `bentoml containerize` 少寫 Dockerfile |
| Capstone | 正式服務常用 Compose / K8s；本頁先把單機 Docker 打穩 |

---

## 9. 檢核：讀完這頁你應該能回答

1. Dockerfile、Image、Container 三者差在哪？誰是配方、誰是範本、誰是正在跑的實例？
2. `docker build` 跟 `docker run` 各發生在生命週期哪一步？
3. `-p 8000:8000` 左邊、右邊分別是什麼？少了會怎樣？
4. 什麼時候該從單次 `docker run` 升級到 Compose？

動手驗證：完成 [`02_docker`](./sandbox/02_docker/README.md)——`build` → `run` → `curl /health`。
