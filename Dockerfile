FROM python:3.11-slim

WORKDIR /app

# 安装必要的运行时动态链接库（支持 libgl/pillow 图像解码）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖 (保留 ddddocr，同时引入轻量级中文点选高精度识别库 rapidocr-onnxruntime)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pillow \
    ddddocr \
    rapidocr-onnxruntime

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
