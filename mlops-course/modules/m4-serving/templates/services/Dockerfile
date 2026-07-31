# workspace/services/Dockerfile — M4 填空
# 對照：modules/m4-serving/sandbox/02_docker/Dockerfile
#
# 在 workspace/services/ 建置：
#   docker build -t workspace-iris:0.1 .
#   docker run --rm -p 8000:8000 workspace-iris:0.1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# TODO(M4-6): 先複製 requirements（可自建一份最小 fastapi/uvicorn/scikit-learn/joblib）
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# TODO(M4-7): 加上 HEALTHCHECK 打 /health（對照 sandbox Dockerfile）
# HEALTHCHECK ...

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
