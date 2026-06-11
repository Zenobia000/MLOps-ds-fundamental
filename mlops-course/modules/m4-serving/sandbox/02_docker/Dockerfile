# 這個 Dockerfile 示範什麼:
#   階 5 Docker 的「最小可用單元」——把 01_fastapi 的服務容器化。
#   一個 image = 「程式 + 依賴 + 執行環境」全部凍結在一起,
#   到哪台機器都跑出一樣的結果(解決「在我電腦上明明可以」)。
#
# 怎麼跑(在本資料夾 02_docker/ 執行,build context 指到上一層好複製 01 的程式):
#   docker build -f Dockerfile -t iris-fastapi:0.1 ..
#   docker run --rm -p 8000:8000 iris-fastapi:0.1
#   curl http://localhost:8000/health
#
# 注意 build context 是上一層 (..),因為要把同層的 01_fastapi/ 程式一起複製進去。

# 1. 選一個「瘦身版」官方 Python 基底,固定版本確保可重現
FROM python:3.11-slim

# 2. 設定環境變數:不寫 .pyc、輸出不緩衝(log 即時可見)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. 工作目錄
WORKDIR /app

# 4. 先只複製 requirements 再裝依賴 —— 利用 layer cache:
#    只要 requirements 沒變,改程式碼時不會重裝套件,build 快很多。
COPY 01_fastapi/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. 再複製應用程式碼(01_fastapi/app.py)
COPY 01_fastapi/app.py ./app.py

# 6. 宣告服務埠(文件用途;實際對外仍要 docker run -p)
EXPOSE 8000

# 7. 容器健康檢查:定期打 /health,讓編排器知道服務是否還活著
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

# 8. 啟動指令:注意 host 要綁 0.0.0.0,否則容器外連不進來
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
