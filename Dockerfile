FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装Python依赖
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ .

# 复制其他必要文件
COPY Agent/ ./Agent/
COPY Models/ ./Models/
COPY Tools/ ./Tools/
COPY Utils/ ./Utils/
COPY data/ ./data/
COPY prompts/ ./prompts/

# 复制数据库文件（如果存在）
COPY portfolio.db ./portfolio.db
COPY calibration.db ./calibration.db
COPY stock_pools.db ./stock_pools.db

# 创建必要的目录
RUN mkdir -p logs data

# 设置环境变量
ENV PYTHONPATH=/app
ENV PORT=8000

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]