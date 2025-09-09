# Render 全栈部署指南

## 单一服务部署方案

### 1. 项目结构调整
```
Auto-GPT-Stock/
├── backend/
├── frontend/
├── Dockerfile
├── render.yaml
└── start.sh
```

### 2. 创建 Dockerfile
```dockerfile
FROM node:18-slim as frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# 安装Python依赖
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ ./backend/
COPY *.db ./

# 复制前端构建结果
COPY --from=frontend-builder /app/frontend/dist ./static

# 安装nginx用于服务前端
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

# 配置nginx
COPY deployment/nginx.conf /etc/nginx/nginx.conf

EXPOSE 10000

COPY deployment/start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]
```

### 3. Nginx 配置
创建 `deployment/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    
    server {
        listen 10000;
        
        # 前端静态文件
        location / {
            root /app/static;
            try_files $uri $uri/ /index.html;
        }
        
        # API代理到后端
        location /api/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

### 4. 启动脚本
创建 `deployment/start.sh`:
```bash
#!/bin/bash

# 启动后端API服务
cd /app/backend
uvicorn app:app --host 127.0.0.1 --port 8000 &

# 启动nginx服务前端
nginx -g "daemon off;" &

# 等待所有后台进程
wait
```

### 5. Render配置文件
创建 `render.yaml`:
```yaml
services:
  - type: web
    name: stock-analysis
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: ANTHROPIC_AUTH_TOKEN
        sync: false
      - key: ANTHROPIC_BASE_URL
        value: https://anyrouter.top
```
